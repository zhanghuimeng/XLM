from logging import getLogger
import os
import copy
import time
import json
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from math import sqrt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, matthews_corrcoef, mean_squared_error, mean_absolute_error

from src.fp16 import network_to_half
from apex.fp16_utils import FP16_Optimizer

from ..utils import get_optimizer, concat_batches, truncate, to_cuda
from ..data.dataset import Dataset, ParallelDataset
from ..data.loader import load_binarized, set_dico_parameters

from tensorboardX import SummaryWriter

logger = getLogger()

class QE:

    def __init__(self, embedder, scores, params):
        """
        Initialize QE trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        self.params = params
        self.scores = scores
        self.writer = SummaryWriter()

    def get_iterator(self, splt):
        """
        Build data iterator.
        data[splt]['x']是数据，data[splt]['y']是标签。
        这个Dataset是自己实现的，而且没有继承pytorch的Dataset。。
        """
        return self.data[splt]['x'].get_iterator(
            shuffle=(splt == 'train'),
            return_indices=True,
            group_by_size=self.params.group_by_size
        )

    def run(self):
        """
        Run QE training / evaluation.
        """
        params = self.params

        # task parameters
        params.out_features = 1
        self.is_classif = False
        # QE的源语言和目标语言
        self.langs = params.transfer_task.split('-')
        assert(len(self.langs) == 2)

        # load data
        self.data = self.load_data(params.qe_task_path)
        if not self.data['dico'] == self._embedder.dico:
            raise Exception(("Dictionary in evaluation data (%i words) seems different than the one " +
                             "in the pretrained model (%i words). Please verify you used the same dictionary, " +
                             "and the same values for max_vocab and min_count.") % (len(self.data['dico']), len(self._embedder.dico)))

        # embedder
        # 这个embedder大概就是transformer
        # 这个cuda大概就是普通的把模型放到cuda上的意思
        self.embedder = copy.deepcopy(self._embedder)
        self.embedder.cuda()

        # projection layer
        # 若干个dropout + linear层（这个我可以改的……）
        # 我认识到这个后面应该加一层softmax了……
        self.proj = nn.Sequential(*[
            nn.Dropout(params.dropout),
            nn.Linear(self.embedder.out_dim, params.out_features)
        ]).cuda()

        # float16
        if params.fp16:
            assert torch.backends.cudnn.enabled
            self.embedder.model = network_to_half(self.embedder.model)
            self.proj = network_to_half(self.proj)

        # optimizer
        # 看来可以比较简单地解决fine-tune哪些的问题……
        self.optimizer = get_optimizer(
            list(self.embedder.get_parameters(params.finetune_layers)) +
            list(self.proj.parameters()),
            params.optimizer
        )
        if params.fp16:
            self.optimizer = FP16_Optimizer(self.optimizer, dynamic_loss_scale=True)

        self.n_elapsed_batches = 0

        # train and evaluate the model
        for epoch in range(params.n_epochs):

            # update epoch
            self.epoch = epoch

            # training
            logger.info("QE - Training epoch %i ..." % (epoch))
            self.train()

            # evaluation
            logger.info("QE - Evaluating epoch %i ..." % (epoch))
            with torch.no_grad():
                scores = self.eval()
                # update是啥？
                self.scores.update(scores)

    def train(self):
        """
        Finetune for one epoch on the training set.
        """
        params = self.params
        # 这两句的意思是：设为训练模式（不是开始训练）
        self.embedder.train()
        self.proj.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        bn = 0  # number of batch
        t = time.time()

        iterator = self.get_iterator('train')
        # 我猜QE的语言得自己设置……
        lang_id1 = params.lang2id[self.langs[0]]
        lang_id2 = params.lang2id[self.langs[1]]

        while True:

            # batch
            try:
                batch = next(iterator)
            except StopIteration:
                break
            # 目前不是很懂这个idx是啥……（看来是在原数据里的index，要这么去查label）
            # （图啥……）
            (sent1, len1), (sent2, len2), idx = batch
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
            # 不知道要不要分成不同的语言（要）
            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang_id1,
                sent2, len2, lang_id2,
                params.pad_index,
                params.eos_index,
                reset_positions=False
            )
            y = self.data['train']['y'][idx]
            bs = len(lengths)

            # cuda
            x, y, lengths = to_cuda(x, y, lengths)

            # loss（直接求MSE loss）
            # 这个取的也是第一列……
            output = self.proj(self.embedder.get_embeddings(x, lengths, positions=None, langs=None))
            # 这个F的写法很神奇，实际上是functional
            # out_feature已经是1了，这里是把多余的维度都去掉了
            loss = F.mse_loss(output.squeeze(1), y.float())

            # backward / optimization（这啥）
            self.optimizer.zero_grad()
            if params.fp16:
                self.optimizer.backward(loss)
            else:
                loss.backward()
            self.optimizer.step()

            # update statistics
            # numpy.ndarray.item：复制到标准python标量
            ns += bs
            nw += lengths.sum().item()
            losses.append(loss.item())

            # log
            # if ns != 0 and ns % (10 * bs) < bs:
            #     logger.info(
            #         "QE - %s-%s - Epoch %s - Train iter %7i - %.1f words/s - %s Loss: %.4f"
            #         % (self.langs[0], self.langs[1], self.epoch, ns, nw / (time.time() - t), 'MSE', sum(losses) / len(losses))
            #     )
            #     nw, t = 0, time.time()
            #     losses = []
            # 上面这堆我不是很懂，所以换成每个batch输出一次好了
            # 但是我不知道每个epoch有几个batch……
            logger.info("QE - %s - Epoch %s - Batch %5i - Loss: %.4f" % (params.transfer_task, self.epoch, bn, loss))

            # 问题：一个epoch有几个batch？（不知道）epoch是在哪里设置的？（run的时候）
            self.writer.add_scalars('data/train', {'loss': loss}, self.n_elapsed_batches)

            bn += 1
            self.n_elapsed_batches += 1
            # epoch size
            # 为啥这个要我设置（ok，也可以不设置……）
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def eval(self):
        """
        Evaluate on QE validation and test sets, for all languages.
        """
        params = self.params
        task = params.transfer_task
        self.embedder.eval()
        self.proj.eval()

        scores = OrderedDict({'epoch': self.epoch})

        for splt in ['valid', 'test']:

            pred = []  # predicted values
            gold = []  # real values

            lang_id1 = params.lang2id[self.langs[0]]
            lang_id2 = params.lang2id[self.langs[1]]

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
                output = self.proj(self.embedder.get_embeddings(x, lengths, positions, langs))
                predictions = output.squeeze(1)

                # update statistics
                pred.append(predictions.cpu().numpy())
                gold.append(y.cpu().numpy())

            gold = np.concatenate(gold)
            pred = np.concatenate(pred)

            # 打印出来好debug==
            with open(os.path.join(params.dump_path, 'pred_epoch_%d.hter' % self.epoch), 'w') as f:
                for h in pred:
                    f.write('%f\n' % h)

            pearson = pearsonr(pred, gold)[0]
            rmse = sqrt(mean_squared_error(pred, gold))
            mae = mean_absolute_error(pred, gold)

            scores['%s_%s_pearson' % (task, splt)] = pearson
            scores['%s_%s_rmse' % (task, splt)] = rmse
            scores['%s_%s_mae' % (task, splt)] = mae

            self.writer.add_scalars('data/%s' % splt, {'pearson': pearson,
                                                      'rmse': rmse,
                                                      'mae': mae}, self.n_elapsed_batches)

        # 似乎value是浮点数的时候就不行……
        logger.info("__log__:%s" % json.dumps(str(scores)))
        return scores

    def load_data(self, task):
        """
        Load pair regression/classification bi-sentence tasks
        我只要load双语regression task就可以啦
        """
        params = self.params
        data = {splt: {} for splt in ['train', 'valid', 'test']}
        dpath = os.path.join(params.data_path, 'eval', task)

        self.n_sent = 2

        for splt in ['train', 'valid', 'test']:

            # fix：只有XNLI的是valid，其他的是dev
            splt2 = splt
            if splt == "valid":
                splt2 = "dev"

            # load data and dictionary
            data1 = load_binarized(os.path.join(dpath, '%s.s1.pth' % splt2), params)
            data2 = load_binarized(os.path.join(dpath, '%s.s2.pth' % splt2), params)
            data['dico'] = data.get('dico', data1['dico'])

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
            # 看来data[splt]['x']是数据，data[splt]['y']是label。。
            # read labels from file
            with open(os.path.join(dpath, '%s.label' % splt2), 'r') as f:
                lines = [l.rstrip() for l in f]
            assert all(0 <= float(x) <= 1 for x in lines)
            y = [float(l) for l in lines]
            data[splt]['y'] = torch.FloatTensor(y)
            assert len(data[splt]['x']) == len(data[splt]['y'])

        # compute weights for weighted training
        self.weights = None

        return data
