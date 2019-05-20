#!/usr/bin/env bash

set -e

# data paths
MAIN_PATH=$PWD
OUTPATH=$PWD/data
PROCESSED_PATH=$PWD/data/processed/XLM15
CODES_PATH=$MAIN_PATH/codes_xnli_15
VOCAB_PATH=$MAIN_PATH/vocab_xnli_15
URLPATH=https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2F

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

# install tools
./install-tools.sh

# create directories
# rm -r $OUTPATH
mkdir -p $OUTPATH

# QE

qe_tasks="QE/WMT17/sentence_level/en_de QE/WMT17/sentence_level/de_en"

for task in $qe_tasks
do
  if [ ! -d $PROCESSED_PATH/eval/$task ]; then
    mkdir -p $PROCESSED_PATH/eval/$task
  else
    rm -r $PROCESSED_PATH/eval/$task/*
  fi
    for splt in train dev test.2017
  do
#      sed '1d' $OUTPATH/$task/${splt}.src | cut -f1 | $REPLACE_UNICODE_PUNCT | \
#        $NORM_PUNC -l en | $REM_NON_PRINT_CHAR > $PROCESSED_PATH/eval/$task/${splt}.x
#      sed '1d' $OUTPATH/$task/${splt}.mt | cut -f2 > $PROCESSED_PATH/eval/$task/${splt}.y
      split=$splt
      if [ "$split" == "test.2017" ]; then
        split="test"
      fi
#      paste $PROCESSED_PATH/eval/$task/${splt}.x $PROCESSED_PATH/eval/$task/${splt}.y $OUTPATH/$task/${splt}.hter \
#        > $PROCESSED_PATH/eval/$task/${split}.xlm.tsv
#      rm $PROCESSED_PATH/eval/$task/${splt}.x $PROCESSED_PATH/eval/$task/${splt}.y
      paste $OUTPATH/$task/${splt}.src $OUTPATH/$task/${splt}.mt $OUTPATH/$task/${splt}.hter \
        > $PROCESSED_PATH/eval/$task/${split}.xlm.tsv
  done
done
#sed '1d' $OUTPATH/SST-2/test.tsv | cut -f2 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR > $OUTPATH/SST-2/test.xlm.tsv
#rm $OUTPATH/*SST-2.zip*

# Get BPE codes and vocab
wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $MAIN_PATH
wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $MAIN_PATH

# apply BPE codes and binarize the QE corpora

for task in $qe_tasks
do
  for splt in train dev test
  do
    FPATH=$PROCESSED_PATH/eval/${task}/${splt}.xlm.tsv
    # 分别读出三列，分别BPE
    cut -f1 $FPATH > ${FPATH}.f1
    $FASTBPE applybpe $PROCESSED_PATH/eval/$task/${splt}.s1 ${FPATH}.f1 $CODES_PATH
    python preprocess.py $VOCAB_PATH $PROCESSED_PATH/eval/$task/${splt}.s1
    rm ${FPATH}.f1

    cut -f2 $FPATH > ${FPATH}.f2
    $FASTBPE applybpe $PROCESSED_PATH/eval/$task/${splt}.s2 ${FPATH}.f2 $CODES_PATH
    python preprocess.py $VOCAB_PATH $PROCESSED_PATH/eval/$task/${splt}.s2
    rm ${FPATH}.f2

    cut -f3 $FPATH > $PROCESSED_PATH/eval/$task/${splt}.label
  done
done
