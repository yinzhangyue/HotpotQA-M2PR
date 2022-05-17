import sys, subprocess

# 5e-6 1e-5 2e-5
def main():
    # subprocess.call(
    #     "CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
    # --nproc_per_node=2 GPU_train.py \
    # --lr 1e-5 \
    # --epoch 16 \
    # --batch_size 1 \
    # --update_every 16 \
    # --Debug",
    #     shell=True,
    # )
    subprocess.call(
        "CUDA_VISIBLE_DEVICES=6 python multi_train.py \
    --lr 1e-5 \
    --epoch 16 \
    --batch_size 1 \
    --update_every 16",
        shell=True,
    )


if __name__ == "__main__":
    main()
