#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES="0"
python qe.py \
    --exp_name test_qe_mlm_tlm \
    --dump_path ./dumped/ \
    --model_path mlm_tlm_xnli15_1024.pth \
    --data_path ./data/processed/XLM15 \
    --qe_task_path QE/WMT17/sentence_level/en_de \
    --transfer_task en-de \
    --optimizer adam,lr=0.00001 \
    --batch_size 10 \
    --n_epochs 250 \
    --epoch_size -1 \
    --max_len 256 \
    --max_vocab 95000
