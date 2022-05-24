import sys, subprocess

# 5e-6 1e-5 2e-5
def main():
    subprocess.call(
        " CUDA_VISIBLE_DEVICES=0 python validation.py",
        shell=True,
    )


if __name__ == "__main__":
    main()
