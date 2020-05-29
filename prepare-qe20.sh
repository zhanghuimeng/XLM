#!/usr/bin/env bash

# data paths
DATA_PATH=$PWD/data/qe20
TASK1_PATH=$DATA_PATH/sentence-da
TASK2_PATH=$DATA_PATH/word-pe
TASK1_OUTPATH=$DATA_PATH/processed/task1
TASK2_OUTPATH=$DATA_PATH/processed/task2

# model paths
MODEL_PATH=$PWD/pre-model
CODES_PATH=$PWD/pre-model/codes_xnli_15
VOCAB_PATH=$PWD/pre-model/vocab_xnli_15

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast
CUT_GAP=$TOOLS_PATH/cut_word_gap_labels.py
PSEUDO_DATA=$TOOLS_PATH/create_pseudo_y.py
MARK=$TOOLS_PATH/mark_word_piece.py

# install tools
./install-tools.sh

# create directories
mkdir -p $TASK1_OUTPATH
mkdir -p $TASK2_OUTPATH

LG1=(en en)
LG2=(de zh)

wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $MODEL_PATH
wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $MODEL_PATH

# Task1
echo "*** Preparing Task 1 set ****"
for i in $(seq 0 1); do
    lg1="${LG1[$i]}"
    lg2="${LG2[$i]}"

    # Cut out source sentence
    if [ $lg1 == "en" ]
    then
        sed '1d' "$TASK1_PATH/$lg1-$lg2/train.$lg1$lg2.df.short.tsv" | cut -f2 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.train.s1.tok"
        sed '1d' "$TASK1_PATH/$lg1-$lg2/dev.$lg1$lg2.df.short.tsv" | cut -f2 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.dev.s1.tok"
        sed '1d' "$TASK1_PATH/$lg1-$lg2/test.$lg1$lg2.df.short.tsv" | cut -f2 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.test.s1.tok"
    else
        sed '1d' "$TASK1_PATH/$lg1-$lg2/train.$lg1$lg2.df.short.tsv" | cut -f2 | $TOKENIZE $lg1 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.train.s1.tok"
        sed '1d' "$TASK1_PATH/$lg1-$lg2/dev.$lg1$lg2.df.short.tsv" | cut -f2 | $TOKENIZE $lg1 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.dev.s1.tok"
        sed '1d' "$TASK1_PATH/$lg1-$lg2/test.$lg1$lg2.df.short.tsv" | cut -f2 | $TOKENIZE $lg1 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.test.s1.tok"
    fi
    # Apple BPE and preprocess
    $FASTBPE applybpe "$TASK1_OUTPATH/$lg1-$lg2.train.s1" "$TASK1_OUTPATH/$lg1-$lg2.train.s1.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK1_OUTPATH/$lg1-$lg2.train.s1"
    $FASTBPE applybpe "$TASK1_OUTPATH/$lg1-$lg2.dev.s1" "$TASK1_OUTPATH/$lg1-$lg2.dev.s1.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK1_OUTPATH/$lg1-$lg2.dev.s1"
    $FASTBPE applybpe "$TASK1_OUTPATH/$lg1-$lg2.test.s1" "$TASK1_OUTPATH/$lg1-$lg2.test.s1.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK1_OUTPATH/$lg1-$lg2.test.s1"

    # Cut out translation sentence
    if [ $lg2 == "en" ]
    then
        sed '1d' "$TASK1_PATH/$lg1-$lg2/train.$lg1$lg2.df.short.tsv" | cut -f3 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.train.s2.tok"
        sed '1d' "$TASK1_PATH/$lg1-$lg2/dev.$lg1$lg2.df.short.tsv" | cut -f3 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.dev.s2.tok"
        sed '1d' "$TASK1_PATH/$lg1-$lg2/test.$lg1$lg2.df.short.tsv" | cut -f3 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.test.s2.tok"
    else
        sed '1d' "$TASK1_PATH/$lg1-$lg2/train.$lg1$lg2.df.short.tsv" | cut -f3 | $TOKENIZE $lg2 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.train.s2.tok"
        sed '1d' "$TASK1_PATH/$lg1-$lg2/dev.$lg1$lg2.df.short.tsv" | cut -f3 | $TOKENIZE $lg2 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.dev.s2.tok"
        sed '1d' "$TASK1_PATH/$lg1-$lg2/test.$lg1$lg2.df.short.tsv" | cut -f3 | $TOKENIZE $lg2 | python $LOWER_REMOVE_ACCENT > "$TASK1_OUTPATH/$lg1-$lg2.test.s2.tok"
    fi
    # Apple BPE and preprocess
    $FASTBPE applybpe "$TASK1_OUTPATH/$lg1-$lg2.train.s2" "$TASK1_OUTPATH/$lg1-$lg2.train.s2.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK1_OUTPATH/$lg1-$lg2.train.s2"
    $FASTBPE applybpe "$TASK1_OUTPATH/$lg1-$lg2.dev.s2" "$TASK1_OUTPATH/$lg1-$lg2.dev.s2.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK1_OUTPATH/$lg1-$lg2.dev.s2"
    $FASTBPE applybpe "$TASK1_OUTPATH/$lg1-$lg2.test.s2" "$TASK1_OUTPATH/$lg1-$lg2.test.s2.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK1_OUTPATH/$lg1-$lg2.test.s2"

    # Cut out label
    sed '1d' "$TASK1_PATH/$lg1-$lg2/train.$lg1$lg2.df.short.tsv" | cut -f7 > "$TASK1_OUTPATH/$lg1-$lg2.train.label"
    sed '1d' "$TASK1_PATH/$lg1-$lg2/dev.$lg1$lg2.df.short.tsv" | cut -f7 > "$TASK1_OUTPATH/$lg1-$lg2.dev.label"
    python $PSEUDO_DATA --s1 "$TASK1_OUTPATH/$lg1-$lg2.test.s1" --s2 "$TASK1_OUTPATH/$lg1-$lg2.test.s2" \
        --type "DA" --name "$lg1-$lg2" --output "$TASK1_OUTPATH"

    # Clean
    rm -rf "$TASK1_OUTPATH"/*.tok
done

# Task2
echo "*** Preparing Task 2 set ****"
for i in $(seq 0 1); do
    lg1="${LG1[$i]}"
    lg2="${LG2[$i]}"

    # TOKENIZE source sentence
    if [ $lg1 == "en" ]
    then
        cat "$TASK2_PATH/$lg1-$lg2/train/train.src" | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.train.s1.tok"
        cat "$TASK2_PATH/$lg1-$lg2/dev/dev.src" | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.dev.s1.tok"
        cat "$TASK2_PATH/$lg1-$lg2/test/test.src" | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.test.s1.tok"
    else
        cat "$TASK2_PATH/$lg1-$lg2/train/train.src" | $TOKENIZE $lg1 | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.train.s1.tok"
        cat "$TASK2_PATH/$lg1-$lg2/dev/dev.src" | $TOKENIZE $lg1 | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.dev.s1.tok"
        cat "$TASK2_PATH/$lg1-$lg2/test/test.src" | $TOKENIZE $lg1 | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.test.s1.tok"
    fi
    # Apple BPE and preprocess
    $FASTBPE applybpe "$TASK2_OUTPATH/$lg1-$lg2.train.s1" "$TASK2_OUTPATH/$lg1-$lg2.train.s1.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK2_OUTPATH/$lg1-$lg2.train.s1"
    $FASTBPE applybpe "$TASK2_OUTPATH/$lg1-$lg2.dev.s1" "$TASK2_OUTPATH/$lg1-$lg2.dev.s1.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK2_OUTPATH/$lg1-$lg2.dev.s1"
    $FASTBPE applybpe "$TASK2_OUTPATH/$lg1-$lg2.test.s1" "$TASK2_OUTPATH/$lg1-$lg2.test.s1.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK2_OUTPATH/$lg1-$lg2.test.s1"
    # Mark wordpiece ranges
    python $MARK --input "$TASK2_OUTPATH/$lg1-$lg2.train.s1" --output "$TASK2_OUTPATH/$lg1-$lg2.train.s1.bperange"
    python $MARK --input "$TASK2_OUTPATH/$lg1-$lg2.dev.s1" --output "$TASK2_OUTPATH/$lg1-$lg2.dev.s1.bperange"
    python $MARK --input "$TASK2_OUTPATH/$lg1-$lg2.test.s1" --output "$TASK2_OUTPATH/$lg1-$lg2.test.s1.bperange"

    # TOKENIZE translation sentence
    if [ $lg2 == "en" ]
    then
        cat "$TASK2_PATH/$lg1-$lg2/train/train.mt" | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.train.s2.tok"
        cat "$TASK2_PATH/$lg1-$lg2/dev/dev.mt" | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.dev.s2.tok"
        cat "$TASK2_PATH/$lg1-$lg2/test/test.mt" | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.test.s2.tok"
    else
        cat "$TASK2_PATH/$lg1-$lg2/train/train.mt" | $TOKENIZE $lg2 | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.train.s2.tok"
        cat "$TASK2_PATH/$lg1-$lg2/dev/dev.mt" | $TOKENIZE $lg2 | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.dev.s2.tok"
        cat "$TASK2_PATH/$lg1-$lg2/test/test.mt" | $TOKENIZE $lg2 | python $LOWER_REMOVE_ACCENT > "$TASK2_OUTPATH/$lg1-$lg2.test.s2.tok"
    fi
    # Apple BPE and preprocess
    $FASTBPE applybpe "$TASK2_OUTPATH/$lg1-$lg2.train.s2" "$TASK2_OUTPATH/$lg1-$lg2.train.s2.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK2_OUTPATH/$lg1-$lg2.train.s2"
    $FASTBPE applybpe "$TASK2_OUTPATH/$lg1-$lg2.dev.s2" "$TASK2_OUTPATH/$lg1-$lg2.dev.s2.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK2_OUTPATH/$lg1-$lg2.dev.s2"
    $FASTBPE applybpe "$TASK2_OUTPATH/$lg1-$lg2.test.s2" "$TASK2_OUTPATH/$lg1-$lg2.test.s2.tok" $CODES_PATH
    python preprocess.py $VOCAB_PATH "$TASK2_OUTPATH/$lg1-$lg2.test.s2"
    # Mark wordpiece ranges
    python $MARK --input "$TASK2_OUTPATH/$lg1-$lg2.train.s2" --output "$TASK2_OUTPATH/$lg1-$lg2.train.s2.bperange"
    python $MARK --input "$TASK2_OUTPATH/$lg1-$lg2.dev.s2" --output "$TASK2_OUTPATH/$lg1-$lg2.dev.s2.bperange"
    python $MARK --input "$TASK2_OUTPATH/$lg1-$lg2.test.s2" --output "$TASK2_OUTPATH/$lg1-$lg2.test.s2.bperange"

    # Parse Label
    cp "$TASK2_PATH/$lg1-$lg2/train/train.hter" "$TASK2_OUTPATH/$lg1-$lg2.train.label"
    cp "$TASK2_PATH/$lg1-$lg2/dev/dev.hter" "$TASK2_OUTPATH/$lg1-$lg2.dev.label"
    python $PSEUDO_DATA --s1 "$TASK2_PATH/$lg1-$lg2/test/test.src" --s2 "$TASK2_PATH/$lg1-$lg2/test/test.mt" \
        --type "HTER" --name "$lg1-$lg2" --output "$TASK2_OUTPATH"
    python $CUT_GAP --input "$TASK2_PATH/$lg1-$lg2/train/train.source_tags" \
        --word_output "$TASK2_OUTPATH/$lg1-$lg2.train.src_tags.label"
    python $CUT_GAP --input "$TASK2_PATH/$lg1-$lg2/dev/dev.source_tags" \
        --word_output "$TASK2_OUTPATH/$lg1-$lg2.dev.src_tags.label"
    python $CUT_GAP --input "$TASK2_PATH/$lg1-$lg2/train/train.tags" --cut \
        --word_output "$TASK2_OUTPATH/$lg1-$lg2.train.tgt_tags.label" \
        --gap_output "$TASK2_OUTPATH/$lg1-$lg2.train.gap_tags.label"
    python $CUT_GAP --input "$TASK2_PATH/$lg1-$lg2/dev/dev.tags" --cut \
        --word_output "$TASK2_OUTPATH/$lg1-$lg2.dev.tgt_tags.label" \
        --gap_output "$TASK2_OUTPATH/$lg1-$lg2.dev.gap_tags.label"
    python $PSEUDO_DATA --s1 "$TASK2_PATH/$lg1-$lg2/test/test.src" --s2 "$TASK2_PATH/$lg1-$lg2/test/test.mt" \
        --type "TAG" --name "$lg1-$lg2" --output "$TASK2_OUTPATH"

    # Clean
    rm -rf "$TASK2_OUTPATH"/*.tok
done
