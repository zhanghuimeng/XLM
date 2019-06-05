# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse

from src.utils import bool_flag, initialize_exp
from src.evaluation.qe import QE
from src.model.embedder import SentenceEmbedder


TASKS = ["2017de_en", "2017en_de"]


# parse parameters
parser = argparse.ArgumentParser(description='Train on QE 2017/2018 tasks')

# main parameters
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")

# float16
parser.add_argument("--fp16", type=bool_flag, default=False,
                    help="Run model with float16")

# evaluation task / pretrained model
parser.add_argument("--transfer_task", type=str, default="",
                    help="Transfer task, example: 'en-de' ")
parser.add_argument("--qe_task_path", type=str, default="", help="example: WMT17/sentence_level/en_de")
parser.add_argument("--model_path", type=str, default="",
                    help="Model location")

# data
parser.add_argument("--data_path", type=str, default="",
                    help="Data path")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--min_count", type=int, default=0,
                    help="Minimum vocabulary count")

# batch parameters
parser.add_argument("--max_len", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--group_by_size", type=bool_flag, default=False,
                    help="Sort sentences by size during the training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch")
parser.add_argument("--max_batch_size", type=int, default=0,
                    help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
parser.add_argument("--tokens_per_batch", type=int, default=-1,
                    help="Number of tokens per batch")

# model / optimization
parser.add_argument("--finetune_layers", type=str, default='0:_1',
                    help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
parser.add_argument("--weighted_training", type=bool_flag, default=False,
                    help="Use a weighted loss during training")
parser.add_argument("--dropout", type=float, default=0,
                    help="Fine-tuning dropout")
parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                    help="Optimizer")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of epochs")
parser.add_argument("--epoch_size", type=int, default=-1,
                    help="Epoch size (-1 for full pass over the dataset)")
parser.add_argument("--loss_type", type=str, default="mse", help="Loss type (only mse and xent now)")

# debug
parser.add_argument("--debug_train", type=bool_flag, default=False,
                    help="Use valid sets for train sets (faster loading)")
parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                    help="Debug multi-GPU / multi-node within a SLURM job")

# parse parameters
params = parser.parse_args()
if params.tokens_per_batch > -1:
    params.group_by_size = True

# check parameters
assert os.path.isdir(params.data_path)
assert os.path.isfile(params.model_path)

# reload pretrained model
embedder = SentenceEmbedder.reload(params.model_path, params)

# reload langs from pretrained model
# 我猜这里是语言和id的互相映射（而不是word2id！！）
params.n_langs = embedder.pretrain_params['n_langs']
params.id2lang = embedder.pretrain_params['id2lang']
params.lang2id = embedder.pretrain_params['lang2id']

# initialize the experiment / build sentence embedder
logger = initialize_exp(params)
scores = {}

# prepare trainers / evaluators
qe = QE(embedder, scores, params)

# run
qe.run()
