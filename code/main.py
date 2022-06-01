import sys, subprocess

# 5e-6 1e-5 2e-5
def main():
    # subprocess.call(
    #     " CUDA_VISIBLE_DEVICES=0 python train.py --task RE --lr 5e-6",
    #     shell=True,
    # )
    subprocess.call(
        " CUDA_VISIBLE_DEVICES=5 python train.py \
            --task QA --lr 5e-6 --batch_size 4\
            --train_path ../HotpotQA/hotpot-train.json \
            --dev_path ../HotpotQA/hotpot-train.json",
        shell=True,
    )


if __name__ == "__main__":
    main()
