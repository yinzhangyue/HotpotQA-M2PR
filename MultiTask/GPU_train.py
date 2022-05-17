from ast import parse
import torch
import torch.distributed as dist
import torch.optim as optim

import numpy as np
from transformers import RobertaTokenizerFast

import fastNLP
from fastNLP import DistTrainer, get_local_rank
from fastNLP import Trainer, Tester
from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP.core.losses import LossBase
from fastNLP import SequentialSampler
from fastNLP import WarmupCallback, SaveModelCallback, GradientClipCallback

import argparse
from multi_preprocess import HotpotQAPipe
from multi_metric import commonLoss, MultiMetric
from multi_model import ARobertaForMulti
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    ##### Set Seed ####
    set_seed(args.seed)
    # Distribute training
    dist.init_process_group("nccl")

    # 先让主进程(rank==0)先执行，进行数据处理，预训模型参数下载等操作，然后保存cache
    print("###########{}###########".format(get_local_rank()))
    if get_local_rank() != 0:
        dist.barrier()

    ##### Loading DATA ####
    Sentence_token = "</e>"
    DOC_token = "</d>"
    if args.Debug:
        # Debug
        Hotpot_train_path = "/remote-home/share/zyyin/Hotpot/DATA_multi_task/debug.json"
        Hotpot_dev_path = "/remote-home/share/zyyin/Hotpot/DATA_multi_task/debug.json"
    else:
        # more
        Hotpot_train_path = "/remote-home/share/zyyin/Hotpot/DATA_multi_task/hotpot_train_multi_more.json"
        Hotpot_dev_path = (
            "/remote-home/share/zyyin/Hotpot/DATA_multi_task/hotpot_dev_multi_more.json"
        )

    # Roberta
    modelname = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(modelname)
    tokenizer.add_tokens([Sentence_token, DOC_token])
    qamodel = ARobertaForMulti.from_pretrained(modelname)

    # Databundle
    databundle = HotpotQAPipe(tokenizer=tokenizer).process_from_file(
        paths=Hotpot_train_path
    )
    traindata = databundle.get_dataset("train")
    databundle = HotpotQAPipe(tokenizer=tokenizer).process_from_file(
        paths=Hotpot_dev_path
    )
    devdata = databundle.get_dataset("train")

    qamodel.resize_token_embeddings(len(tokenizer))
    # read pretrained model
    # bestmodelpath=""
    # bestmodel=torch.load(bestmodelpath).state_dict()
    # deepms2.load_state_dict(bestmodel)
    if get_local_rank() == 0:
        dist.barrier()  # 主进程执行完后，其余进程开始读取cache

    if not args.Debug:
        name = "Distribute-lr-{}-update_every-{}-{}-checkpoints".format(
            args.lr, args.update_every, modelname
        )
        save_path = "./save_models/{}".format(name)

        callbacks_all = []
        callbacks_all.append(GradientClipCallback(clip_value=1))
        callbacks_all.append(SaveModelCallback(save_dir=save_path, top=3))

        # tester = Tester(
        #     model=qamodel,
        #     device=args.device,
        #     data=devdata,
        #     loss=loss,
        #     metrics=metrics,
        #     batch_size=args.batch_size,
        # )
        # tester.test()
        # trainer

        trainer = DistTrainer(
            model=qamodel,
            train_data=traindata,
            dev_data=devdata,
            save_path=save_path,
            loss=commonLoss(loss="loss"),
            metrics=MultiMetric(tokenizer=tokenizer),
            callbacks_all=callbacks_all,
            optimizer=optim.AdamW(qamodel.parameters(), lr=args.lr),
            n_epochs=args.epoch,
            batch_size_per_gpu=args.batch_size,
            update_every=args.update_every,
            dev_batch_size=args.batch_size,
            fp16=True,
            use_tqdm=True,
            # sampler=SequentialSampler(),
            # validate_every=2048,
        )
        trainer.train()
    else:
        trainer = DistTrainer(
            model=qamodel,
            train_data=traindata,
            dev_data=devdata,
            loss=commonLoss(loss="loss"),
            metrics=MultiMetric(tokenizer=tokenizer),
            optimizer=optim.AdamW(qamodel.parameters(), lr=args.lr),
            n_epochs=256,
            batch_size_per_gpu=args.batch_size,
            update_every=args.update_every,
            dev_batch_size=args.batch_size,
            fp16=True,
            validate_every=64,
            use_tqdm=True,
            device="cuda",
        )
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--lr",
        default=3e-5,
        type=float,
        help="learning rate",
    )
    parser.add_argument("--warmupsteps", default=1000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--update_every", default=8, type=int)
    parser.add_argument("--epoch", default=16, type=int)
    parser.add_argument("--Debug", action="store_true")
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()
    main(args)
