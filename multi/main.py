import sys, subprocess

# 5e-6 1e-5 2e-5
def main():
    subprocess.call(
        " CUDA_VISIBLE_DEVICES=1 python train.py \
    --lr 5e-6 \
    --task RE \
    --model Roberta \
    --epoch 16 \
    --batch_size 1 \
    --update_every 8",
        shell=True,
    )


if __name__ == "__main__":
    main()
