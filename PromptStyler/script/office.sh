CUDA_VISIBLE_DEVICES=1 \
    python PromptStyler/train.py \
        --config-file PromptStyler/config/office.yaml \
        --trainer StyleTrainer \
        --output-dir PromptStyler/output/office/rn50 \
        MODEL.BACKBONE.NAME RN50 \
        OPTIM.MAX_EPOCH 100 \
        OPTIM.LR 0.002

CUDA_VISIBLE_DEVICES=1 \
    python PromptStyler/train.py \
        --config-file PromptStyler/config/office.yaml \
        --trainer PromptStylerLPTrainer \
        --output-dir PromptStyler/output/office/rn50 \
        --source-domains none \
        --target-domains art clipart product real_world \
        --root /home/xuxiusheng/deeplearning/data \
        OPTIM.MAX_EPOCH 30 \
        MODEL.BACKBONE.NAME RN50