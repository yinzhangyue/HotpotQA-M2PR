import os
import numpy as np
import torch
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed

from transformers import RobertaTokenizerFast


import fastNLP
from fastNLP import MSELoss, Const
from fastNLP import Trainer, Tester
from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP import WarmupCallback, SaveModelCallback, GradientClipCallback

import argparse
from multi_preprocess import HotpotQAPipe
from multi_metric import commonLoss, MultiMetric
from multi_model import ARobertaForMulti
from ds_trainer import DS_Trainer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_argument():

    parser = argparse.ArgumentParser(description="HOTPOT")

    # data
    # cuda
    parser.add_argument(
        "--with_cuda",
        default=False,
        action="store_true",
        help="use CPU in case there's no GPU support",
    )
    parser.add_argument(
        "--use_ema",
        default=False,
        action="store_true",
        help="whether use exponential moving average",
    )

    # train
    parser.add_argument(
        "-b", "--batch_size", default=32, type=int, help="mini-batch size (default: 32)"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="use deepspeed mixture of experts (moe)",
    )

    parser.add_argument(
        "--ep-world-size", default=1, type=int, help="(moe) expert parallel world size"
    )
    parser.add_argument(
        "--num-experts", default=1, type=int, help="(moe) number of total experts"
    )
    parser.add_argument(
        "--top-k", default=1, type=int, help="(moe) gating top 1 and 2 supported"
    )
    parser.add_argument(
        "--min-capacity",
        default=0,
        type=int,
        help="(moe) minimum capacity of an expert regardless of the capacity_factor",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default=None,
        type=str,
        help="(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="(moe) create separate moe param groups, required when using ZeRO w. MoE",
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def create_moe_param_groups(model):
    from deepspeed.moe.utils import is_moe_param

    params_with_weight_decay = {"params": [], "name": "weight_decay_params"}
    moe_params_with_weight_decay = {
        "params": [],
        "moe": True,
        "name": "weight_decay_moe_params",
    }

    for module_ in model.modules():
        moe_params_with_weight_decay["params"].extend(
            [
                p
                for n, p in list(module_._parameters.items())
                if p is not None and is_moe_param(p)
            ]
        )
        params_with_weight_decay["params"].extend(
            [
                p
                for n, p in list(module_._parameters.items())
                if p is not None and not is_moe_param(p)
            ]
        )

    return params_with_weight_decay, moe_params_with_weight_decay


def main(args):
    ##### Set Seed ####
    set_seed(args.seed)

    ##### Loading DATA ####
    Sentence_token = "</e>"
    DOC_token = "</d>"

    # Debug
    Hotpot_train_path = "/remote-home/share/zyyin/Hotpot/DATA_multi_task/debug.json"
    Hotpot_dev_path = "/remote-home/share/zyyin/Hotpot/DATA_multi_task/debug.json"

    # # more
    # Hotpot_train_path = "/remote-home/share/zyyin/Hotpot/DATA_multi_task/hotpot_train_multi_more.json"
    # Hotpot_dev_path = (
    #     "/remote-home/share/zyyin/Hotpot/DATA_multi_task/hotpot_dev_multi_more.json"
    # )
    # Roberta
    modelname = "roberta-large"

    # Databundle
    tokenizer = RobertaTokenizerFast.from_pretrained(modelname)
    tokenizer.add_tokens([Sentence_token, DOC_token])
    databundle = HotpotQAPipe(tokenizer=tokenizer).process_from_file(
        paths=Hotpot_train_path
    )
    traindata = databundle.get_dataset("train")

    databundle = HotpotQAPipe(tokenizer=tokenizer).process_from_file(
        paths=Hotpot_dev_path
    )
    devdata = databundle.get_dataset("train")

    net = ARobertaForMulti.from_pretrained(modelname)
    net.resize_token_embeddings(len(tokenizer))
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    if args.moe_param_group:
        parameters = create_moe_param_groups(net)

    # Initialize DeepSpeed to use the following features
    # 1) Distributed model
    # 2) Distributed data loader
    # 3) DeepSpeed optimizer
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=traindata
    )

    fp16 = model_engine.fp16_enabled()
    print(f"fp16={fp16}")

    loss = commonLoss(loss="loss")
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    metrics = MultiMetric(tokenizer=tokenizer)
    callback = []

    trainer = DS_Trainer(
        model=net,
        train_data=traindata,
        dev_data=devdata,
        loss=loss,
        metrics=metrics,
        callbacks=callback,
        optimizer=optimizer,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        update_every=16,
        dev_batch_size=args.batch_size,
        fp16=True,
        validate_every=128,
    )
    trainer.train()


if __name__ == "__main__":
    deepspeed.init_distributed()
    args = add_argument()
    if args.moe:
        deepspeed.utils.groups.initialize(ep_size=args.ep_world_size)
    main(args)
