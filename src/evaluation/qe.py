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

import torch
from torch import nn
import torch.nn.functional as F

from ..optim import get_optimizer
from ..utils import concat_batches, truncate, to_cuda
from ..data.dataset import ParallelDataset
from ..data.loader import load_binarized, set_dico_parameters

logger = getLogger()


class QE:

    def __init__(self, embedder, scores, params):
        """
        Initialize QE trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        # Do not need modification
        self._embedder = embedder
        self.params = params
        self.scores = scores

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

        # embedder
        self.embedder = copy.deepcopy(self._embedder)
        self.embedder.cuda()

        # projection layer
        self.proj = nn.Sequential(*[
            nn.Dropout(params.dropout),
            nn.Linear(self.embedder.out_dim, 3)
        ]).cuda()

        # optimizers
        self.optimizer_e = get_optimizer(list(self.embedder.get_parameters(params.finetune_layers)), params.optimizer_e)
        self.optimizer_p = get_optimizer(self.proj.parameters(), params.optimizer_p)

        # train and evaluate the model
        for epoch in range(params.n_epochs):

            # update epoch
            self.epoch = epoch

            # training
            logger.info("XNLI - Training epoch %i ..." % epoch)
            self.train()

            # evaluation
            logger.info("XNLI - Evaluating epoch %i ..." % epoch)
            with torch.no_grad():
                scores = self.eval()
                self.scores.update(scores)

    def train(self):
        """
        Finetune for one epoch on the XNLI English training set.
        """
        # Not modified yet
        params = self.params
        self.embedder.train()
        self.proj.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        iterator = self.get_iterator('train', 'en')
        lang_id = params.lang2id['en']

        while True:

            # batch
            try:
                batch = next(iterator)
            except StopIteration:
                break
            (sent1, len1), (sent2, len2), idx = batch
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang_id,
                sent2, len2, lang_id,
                params.pad_index,
                params.eos_index,
                reset_positions=False
            )
            y = self.data['en']['train']['y'][idx]
            bs = len(len1)

            # cuda
            x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)

            # loss
            output = self.proj(self.embedder.get_embeddings(x, lengths, positions, langs))
            loss = F.cross_entropy(output, y)

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
                logger.info("XNLI - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
                nw, t = 0, time.time()
                losses = []

            # epoch size
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def eval(self):
        """
        Evaluate on XNLI validation and test sets, for all languages.
        """
        # Not modified yet
        params = self.params
        self.embedder.eval()
        self.proj.eval()

        scores = OrderedDict({'epoch': self.epoch})

        for splt in ['valid', 'test']:

            for lang in XNLI_LANGS:
                if lang not in params.lang2id:
                    continue

                lang_id = params.lang2id[lang]
                valid = 0
                total = 0

                for batch in self.get_iterator(splt, lang):

                    # batch
                    (sent1, len1), (sent2, len2), idx = batch
                    x, lengths, positions, langs = concat_batches(
                        sent1, len1, lang_id,
                        sent2, len2, lang_id,
                        params.pad_index,
                        params.eos_index,
                        reset_positions=False
                    )
                    y = self.data[lang][splt]['y'][idx]

                    # cuda
                    x, y, lengths, positions, langs = to_cuda(x, y, lengths, positions, langs)

                    # forward
                    output = self.proj(self.embedder.get_embeddings(x, lengths, positions, langs))
                    predictions = output.data.max(1)[1]

                    # update statistics
                    valid += predictions.eq(y).sum().item()
                    total += len(len1)

                # compute accuracy
                acc = 100.0 * valid / total
                scores['xnli_%s_%s_acc' % (splt, lang)] = acc
                logger.info("XNLI - %s - %s - Epoch %i - Acc: %.1f%%" % (splt, lang, self.epoch, acc))

        logger.info("__log__:%s" % json.dumps(scores))
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
                # TODO
                pass

        return data
