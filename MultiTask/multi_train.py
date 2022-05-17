import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from torch.nn import CrossEntropyLoss
import torch.optim as optim

import os
import numpy as np
import json
import wandb

from transformers import (
    BertPreTrainedModel,
    BertModel,
    BertLayer,
    BertConfig,
    BertTokenizerFast,
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaTokenizerFast,
)
from transformers.models.albert import (
    AlbertTokenizer,
    AlbertTokenizerFast,
    AlbertModel,
    AlbertForQuestionAnswering,
    AlbertPreTrainedModel,
)

import fastNLP
from fastNLP import MSELoss, Const
from fastNLP import Trainer, Tester
from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP.core.losses import LossBase
from fastNLP import SequentialSampler
from fastNLP import WarmupCallback, SaveModelCallback, GradientClipCallback
from wandbcallback import WandbCallback
import argparse
from multi_preprocess import HotpotQAPipe
from multi_metric import commonLoss, MultiMetric
from multi_model import ARobertaForMulti


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    ##### Set Seed ####
    set_seed(args.seed)

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

    loss = commonLoss(loss="loss")
    optimizer = optim.AdamW(qamodel.parameters(), lr=args.lr)
    metrics = MultiMetric(
        tokenizer=tokenizer,
    )
    if not args.Debug:
        name = "Multi-lr-{}-update_every-{}-{}-checkpoints".format(
            args.lr, args.update_every, modelname
        )
        save_path = "./save_models/{}".format(name)

        callback = []
        callback.append(GradientClipCallback(clip_value=1))
        callback.append(SaveModelCallback(save_dir=save_path, top=3))
        # callback.append(
        #     WandbCallback(
        #         project="HotpotQAspansentence",
        #         name=name,
        #         config={
        #             "lr": args.lr,
        #             "seed": args.seed,
        #             "warmupsteps": args.warmupsteps,
        #             "Batch_size": args.batch_size,
        #             "epoch": args.epoch,
        #             "update_every": args.update_every,
        #         },
        #     )
        # )

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

        trainer = Trainer(
            model=qamodel,
            train_data=traindata,
            device=args.device,
            dev_data=devdata,
            save_path=save_path,
            loss=loss,
            metrics=metrics,
            callbacks=callback,
            optimizer=optimizer,
            n_epochs=args.epoch,
            batch_size=args.batch_size,
            update_every=args.update_every,
            dev_batch_size=args.batch_size,
            fp16=True,
            # sampler=SequentialSampler(),
            # validate_every=2048,
        )
        trainer.train()
    else:
        callback = []

        trainer = Trainer(
            model=qamodel,
            train_data=traindata,
            device=args.device,
            dev_data=devdata,
            loss=loss,
            metrics=metrics,
            callbacks=callback,
            optimizer=optimizer,
            n_epochs=args.epoch,
            batch_size=args.batch_size,
            update_every=args.update_every,
            dev_batch_size=args.batch_size,
            fp16=True,
            validate_every=128,
        )
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("--device", default="cuda:0", type=str)
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

    args = parser.parse_args()
    main(args)
