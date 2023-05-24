#!/bin/sh
python train.py \
    --BATCH_SIZE=16 \
    --EPOCHS=1 \
    --OUTPUT_DIR='D:/LargeDatasets/_PretrainedModels/cmm-summarizer' \
    --HF_TOKEN='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' \
    --HF_HUB_REPO='luntaixia/cnn-summarizer'