import sys, subprocess

# 5e-6 1e-5 2e-5
def main():
    # subprocess.call(
    #     " CUDA_VISIBLE_DEVICES=3 python train.py \
    # --lr 3e-5 \
    # --task QA \
    # --model Bert \
    # --epoch 16 \
    # --batch_size 4 \
    # --update_every 3",
    #     shell=True,
    # )
    subprocess.call(
        " CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed 42 \
    --lr 5e-6 \
    --task RE \
    --model DebertaV2-512 \
    --epoch 16 \
    --batch_size 1 \
    --update_every 8",
        shell=True,
    )


if __name__ == "__main__":
    main()
    # print("### Finished ###")
