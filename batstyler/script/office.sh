CUDA_VISIBLE_DEVICES=0 \
    python batstyler/train.py \
        --config-file batstyler/config/office.yaml \
        --trainer PromptLearnerTrainer \
        --output batstyler/output/office/rn50 \
        MODEL.BACKBONE.NAME RN50 \
        OPTIM.MAX_EPOCH 100 \
        OPTIM.LR 0.1