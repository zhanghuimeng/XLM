# Copied from xnli.py
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import copy
import time
import json
from collections import OrderedDict
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, matthews_corrcoef, mean_squared_error, mean_absolute_error

from ..optim import get_optimizer
from ..utils import concat_batches, truncate, to_cuda, ScoreRecorder, get_embedding_per_token
from ..data.dataset import ParallelDataset
from ..data.loader import load_binarized, set_dico_parameters

logger = getLogger()


class QE:

    def __init__(self, embedder, params):
        """
        Initialize QE trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        # Do not need modification
        self._embedder = embedder
        self.params = params
        self.score_recorder = None

    def get_iterator(self, splt):
        """
        Get a monolingual data iterator.
        """
        # Why do we need it?
        # Modified splt according to QE
        assert splt in self.params.data_split
        return self.data[splt]['x'].get_iterator(
            shuffle=(splt == 'train'),
            group_by_size=self.params.group_by_size,
            return_indices=True
        )

    def run(self, task):
        """
        Run XNLI training / evaluation.
        """
        # Not modified yet
        # Added task
        params = self.params

        # load data
        self.data = self.load_data(task)
        if not self.data['dico'] == self._embedder.dico:
            raise Exception(("Dictionary in evaluation data (%i words) seems different than the one " +
                             "in the pretrained model (%i words). Please verify you used the same dictionary, " +
                             "and the same values for max_vocab and min_count.") % (len(self.data['dico']), len(self._embedder.dico)))

        # Score Recorder
        if task == "DA" or task == "HTER":
            self.score_recorder = ScoreRecorder(['pearson', 'rmse', 'mae'], 'pearson', params.patience)
        else:
            self.score_recorder = ScoreRecorder(['mcc', 'acc', 'f1'], 'mcc', params.patience)

        # embedder
        self.embedder = copy.deepcopy(self._embedder)
        self.embedder.cuda()

        if task == "DA" or task == "HTER":
            # projection layer
            self.proj = nn.Sequential(*[
                nn.Dropout(params.dropout),
                nn.Linear(self.embedder.out_dim, 1)
            ]).cuda()
        elif task == "TAG_SRC":
            # source projection layer
            self.proj = nn.Sequential(*[
                nn.Dropout(params.dropout),
                nn.Linear(self.embedder.out_dim, 1),
                nn.Sigmoid()
            ]).cuda()
        elif task == "TAG_TGT":
            # target projection layer
            self.proj = nn.Sequential(*[
                nn.Dropout(params.dropout),
                nn.Linear(self.embedder.out_dim, 1),
                nn.Sigmoid()
            ]).cuda()
        elif task == "TAG_GAP":
            # target gap projection layer
            self.proj = nn.Sequential(*[
                nn.Dropout(params.dropout),
                nn.Linear(self.embedder.out_dim * 2, 1),
                nn.Sigmoid()
            ]).cuda()

        # optimizers
        self.optimizer_e = get_optimizer(list(self.embedder.get_parameters(params.finetune_layers)), params.optimizer_e)
        self.optimizer_p = get_optimizer(self.proj.parameters(), params.optimizer_p)

        # train and evaluate the model
        for epoch in range(params.n_epochs):

            # update epoch
            self.epoch = epoch

            # training
            logger.info("%s - Training epoch %i ..." % (task, epoch))
            self.train(task=task)

            # evaluation
            logger.info("%s - Evaluating epoch %i ..." % (task, epoch))
            with torch.no_grad():
                self.eval(task=task)

            logger.info("Patience: %d" % self.score_recorder.fuss)
            if not self.score_recorder.check():
                break

    def train(self, task):
        """
        Finetune for one epoch on the QE training set.
        """
        # Modified
        params = self.params
        self.embedder.train()
        self.proj.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        iterator = self.get_iterator('train')
        lang_id1 = params.lang2id[params.lang[0]]
        lang_id2 = params.lang2id[params.lang[1]]

        while True:

            # batch
            try:
                batch = next(iterator)
            except StopIteration:
                break
            (sent1, len1), (sent2, len2), idx = batch  # idx is index in original data
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang_id1,
                sent2, len2, lang_id2,
                params.pad_index,
                params.eos_index,
                reset_positions=False
            )
            y = self.data['train']['y'][idx]
            bs = len(len1)

            # cuda
            x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)

            # loss
            if task == "DA" or task == "HTER":
                output = self.proj(self.embedder.get_embeddings(x, lengths, positions, langs))
                output = output.squeeze(1)
                loss = F.mse_loss(output, y.float())
            else:
                embeddings = self.embedder.get_all_embeddings(x, lengths, positions, langs)
                slen, bs, _ = embeddings.size()
                # logger.info(embeddings.size())
                # logger.info("bs=%d" % bs)
                # logger.info("slen=%d" % slen)
                # put all tags in a column
                output_list = []
                y_list = []
                for i in range(bs):
                    # def get_embedding_per_token(): return cls, source, sep, target, sep
                    cls, src, sep1, tgt, sep2 = get_embedding_per_token(
                        dico=self.data['dico'],
                        x=x[:,i].t(),
                        embeddings=embeddings[:,i,:],
                        mode="first"
                    )

                    # real length
                    len_src = len(src)
                    len_tgt = len(tgt)

                    if task == "TAG_SRC":
                        y_list.append(y[i][:len_src])
                        output = self.proj(src)
                        output = output.squeeze(1)
                        output_list.append(output)
                    elif task == "TAG_TGT":
                        y_list.append(y[i][:len_tgt])
                        output = self.proj(tgt)
                        output = output.squeeze(1)
                        output_list.append(output)
                    elif task == "TAG_GAP":
                        y_list.append(y[i][:len_tgt + 1])
                        top_half = torch.cat([sep1.view(1, -1), tgt], 0)
                        bottom_half = torch.cat([tgt, sep2.view(1, -1)], 0)
                        input = torch.cat([top_half, bottom_half], 1)
                        output = self.proj(input)
                        output = output.squeeze(1)
                        output_list.append(output)

                y_list = torch.cat(y_list, 0)
                output_list = torch.cat(output_list, 0)
                loss_function = torch.nn.BCELoss()
                loss = loss_function(output_list, y_list.float())

            # backward / optimization
            self.optimizer_e.zero_grad()
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_p.step()

            # update statistics
            ns += bs
            nw += lengths.sum().item()
            losses.append(loss.item())

            # log
            if ns % (100 * bs) < bs:
                logger.info("%s - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (task, self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
                nw, t = 0, time.time()
                losses = []

            # epoch size
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def eval(self, task):
        """
        Evaluate on XNLI validation and test sets.
        """
        # Modified
        params = self.params
        self.embedder.eval()
        self.proj.eval()

        scores = OrderedDict({'epoch': self.epoch})

        for splt in self.params.data_split[1:]:

                pred = []
                gold = []

                lang_id1 = params.lang2id[params.lang[0]]
                lang_id2 = params.lang2id[params.lang[1]]

                for batch in self.get_iterator(splt):

                    # batch
                    (sent1, len1), (sent2, len2), idx = batch
                    x, lengths, positions, langs = concat_batches(
                        sent1, len1, lang_id1,
                        sent2, len2, lang_id2,
                        params.pad_index,
                        params.eos_index,
                        reset_positions=False
                    )
                    y = self.data[splt]['y'][idx]

                    # cuda
                    x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)

                    # forward
                    if task == "DA" or task == "HTER":
                        output = self.proj(self.embedder.get_embeddings(x, lengths, positions, langs))
                        predictions = output.squeeze(1)
                    else:
                        # TODO
                        pass

                    # update statistics
                    if task == "DA" or task == "HTER":
                        pred.append(predictions.cpu().numpy())
                        gold.append(y.cpu().numpy())
                    else:
                        # TODO
                        pass

                if task == "DA" or task == "HTER":
                    gold = np.concatenate(gold)
                    pred = np.concatenate(pred)
                else:
                    # TODO
                    pass

                # print and debug
                if task == "DA" or task == "HTER":
                    with open(os.path.join(params.dump_path, 'pred_epoch_%d.pred' % self.epoch), 'w') as f:
                        for h in pred:
                            f.write('%f\n' % h)

                # compute scores
                if task == "DA" or task == "HTER":
                    pearson = pearsonr(pred, gold)[0]
                    rmse = sqrt(mean_squared_error(pred, gold))
                    mae = mean_absolute_error(pred, gold)
                    self.score_recorder.update('pearson', pearson)
                    self.score_recorder.update('rmse', rmse)
                    self.score_recorder.update('mae', mae)
                else:
                    # TODO
                    pass

                # print
                if task == "DA" or task == "HTER":
                    logger.info("QE - %s - %s - Epoch %i - Pearson: %.6f" % (task, splt, self.epoch, pearson))
                    logger.info("QE - %s - %s - Epoch %i - RMSE: %.6f" % (task, splt, self.epoch, rmse))
                    logger.info("QE - %s - %s - Epoch %i - MAE: %.6f" % (task, splt, self.epoch, mae))

        logger.info("__log__:%s" % json.dumps(str(scores)))
        return scores

    def load_data(self, task):
        """
        Load QE task 1 or task 2 data.
        """
        # Not modified yet
        params = self.params
        data = {splt: {} for splt in params.data_split}

        if task == "DA":
            dpath = os.path.join(params.data_path, 'task1')
        else:
            dpath = os.path.join(params.data_path, 'task2')
        filename = "-".join(params.lang)

        for splt in params.data_split:
            # load data and dictionary
            data1 = load_binarized(os.path.join(dpath, '%s.%s.s1.pth' % (filename, splt)), params)
            data2 = load_binarized(os.path.join(dpath, '%s.%s.s2.pth' % (filename, splt)), params)
            data['dico'] = data.get('dico', data1['dico'])  # why only data1?

            # set dictionary parameters
            set_dico_parameters(params, data, data1['dico'])
            set_dico_parameters(params, data, data2['dico'])

            # create dataset
            data[splt]['x'] = ParallelDataset(
                data1['sentences'], data1['positions'],
                data2['sentences'], data2['positions'],
                params
            )

            # load labels
            if task == "DA" or task == "HTER":
                with open(os.path.join(dpath, '%s.%s.label' % (filename, splt)), 'r') as f:
                    labels = [float(l.rstrip()) for l in f]
                data[splt]['y'] = torch.FloatTensor(labels)
                assert len(data[splt]['x']) == len(data[splt]['y'])
            else:
                whole_filename = "%s.%s.%s.label"
                if task == "TAG_SRC":
                    whole_filename = whole_filename % (filename, splt, "src_tags")
                elif task == "TAG_TGT":
                    whole_filename = whole_filename % (filename, splt, "tgt_tags")
                elif task == "TAG_GAP":
                    whole_filename = whole_filename % (filename, splt, "gap_tags")
                with open(os.path.join(dpath, whole_filename), 'r') as f:
                    labels = []
                    for l in f:
                        labels.append([])
                        for token in l.rstrip().split(' '):
                            labels[-1].append(int(token))
                labels = [torch.LongTensor(l) for l in labels]
                data[splt]['y'] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=2)
                assert len(data[splt]['x']) == len(data[splt]['y'])
        return data
