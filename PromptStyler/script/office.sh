CUDA_VISIBLE_DEVICES=0 \
    python PromptStyler/train.py \
        --config-file PromptStyler/config/office.yaml \
        --trainer PromptStylerTrainer \
        --output-dir PromptStyler/output/office/rn50 \
        MODEL.BACKBONE.NAME RN50