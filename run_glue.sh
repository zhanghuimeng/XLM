#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="0"
python glue-xnli.py \
    --exp_name test_glue_mlm_tlm \
    --dump_path ./dumped/ \
    --model_path mlm_tlm_xnli15_1024.pth \
    --data_path ./data/processed/XLM15 \
    --transfer_tasks MNLI,QNLI,QQP,SST-2,STS-B,SST-2 \
    --optimizer adam,lr=0.000005 \
    --batch_size 8 \
    --n_epochs 250 \
    --epoch_size 20000 \
    --max_len 256 \
    --max_vocab 95000
