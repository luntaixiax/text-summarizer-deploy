#!/bin/sh
python train.py \
    --BATCH_SIZE=16 \
    --EPOCHS=1 \
    --OUTPUT_DIR='YOUR-OUTPUT-DIR-HERE/cmm-summarizer' \
    --HF_TOKEN='YOUR-HF-TOKEN-HERE' \
    --HF_HUB_REPO='HF-ACCT-NAME/HF-REPO-NAME-HERE'