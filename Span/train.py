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
from preprocess import HotpotQAPipe, HotpotQATestPipe
from model import (
    ARobertaForQuestionAnswering,
    ABertForQuestionAnswering,
    AAlbertForQuestionAnswering,
)
from metrics import commonLoss, SpanSentenceMetric, DocselectionMetric
from docselection_model import ARobertaForDocselection, AAlbertForDocselection
from dynamicmaskingmodel import (
    Guomasking_ARobertaForQuestionAnswering,
    Guomasking_AAlbertForQuestionAnswering,
)
from electra import AElectraForQuestionAnswering
from transformers.models.electra import (
    ElectraTokenizerFast,
    ElectraConfig,
    ElectraModel,
)
import argparse


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
        Hotpot_train_path = "../DATA/debug.json"
        Hotpot_dev_path = "../DATA/debug.json"
    else:
        # more
        Hotpot_train_path = "../DATA/hotpot_train_multi_more.json"
        Hotpot_dev_path = "../DATA/hotpot_dev_multi_more.json"

        # all
        # Hotpot_train_path = "../DATA/hotpot_train_multi_all.json"
        # Hotpot_dev_path = "../DATA/hotpot_dev_multi_all.json"

    if args.model == "Albert":
        # Albert
        # modelname = "albert-large-v1"
        modelname = "albert-xxlarge-v1"
        # modelname = "mfeb/albert-xxlarge-v2-squad2"
        tokenizer = AlbertTokenizerFast.from_pretrained(modelname)
        tokenizer.add_tokens([Sentence_token, DOC_token])

    elif args.model == "Roberta":
        # Roberta
        modelname = "roberta-large"
        tokenizer = RobertaTokenizerFast.from_pretrained(modelname)
        tokenizer.add_tokens([Sentence_token, DOC_token])

    elif args.model == "Electra":
        # Electra
        modelname = "google/electra-large-discriminator"
        tokenizer = ElectraTokenizerFast.from_pretrained(modelname)
        tokenizer.add_tokens([Sentence_token, DOC_token])

    print("##### {}:{} {}#####".format(args.task, modelname, args.DM))

    # Databundle
    databundle = HotpotQAPipe(tokenizer=tokenizer).process_from_file(
        paths=Hotpot_train_path
    )
    traindata = databundle.get_dataset("train")

    databundle = HotpotQAPipe(tokenizer=tokenizer).process_from_file(
        paths=Hotpot_dev_path
    )
    devdata = databundle.get_dataset("train")

    if args.task == "QA":
        if args.DM is False:
            if args.model == "Albert":
                # Albert Model
                qamodel = AAlbertForQuestionAnswering.from_pretrained(modelname)
            elif args.model == "Roberta":
                # Roberta Model
                qamodel = ARobertaForQuestionAnswering.from_pretrained(modelname)
            elif args.model == "Electra":
                # Roberta Model
                qamodel = AElectraForQuestionAnswering.from_pretrained(modelname)
        else:
            if args.model == "Albert":
                # Albert Model
                qamodel = Guomasking_AAlbertForQuestionAnswering.from_pretrained(
                    modelname
                )
            elif args.model == "Roberta":
                # Roberta Model
                qamodel = Guomasking_ARobertaForQuestionAnswering.from_pretrained(
                    modelname
                )

        metrics = SpanSentenceMetric(
            tokenizer=tokenizer,
            type_logits="type_logits",
            start_logits="start_logits",
            end_logits="end_logits",
            sentence_predictions="sentence_predictions",
            sentence_labels="sentence_labels",
            input_ids="input_ids",
            answer="answer",
            sentence_num="sentence_num",
        )

    elif args.task == "RE":
        if args.model == "Albert":
            # Albert Model
            qamodel = AAlbertForDocselection.from_pretrained(modelname)
        elif args.model == "Roberta":
            # Roberta Model
            # qamodel = ARobertaForDocselection.from_pretrained(modelname)
            checkpoint = "/remote-home/zyyin/Experiment/Hotpot/teacherforcing/roberta-large-lr5e-06-bz4-ws0/2021-10-16-05-31-14-902180/epoch-3_step-1080585_acc-0.952554.pt"
            qamodel = ARobertaForDocselection.from_pretrained(modelname)

            qamodel.resize_token_embeddings(len(tokenizer))
            checkpoint = torch.load(checkpoint).state_dict()
            qamodel.load_state_dict(checkpoint)
        metrics = DocselectionMetric(
            tokenizer=tokenizer,
            selected_pair="selected_pair",
            gold_doc_pair="gold_doc_pair",
            doc_num="doc_num",
        )

    qamodel.resize_token_embeddings(len(tokenizer))
    # read pretrained model
    # bestmodelpath=""
    # bestmodel=torch.load(bestmodelpath).state_dict()
    # deepms2.load_state_dict(bestmodel)

    loss = commonLoss(loss="loss")
    optimizer = optim.AdamW(qamodel.parameters(), lr=args.lr)
    if not args.Debug:
        name = "{}-lr-{}-update_every-{}-{}-checkpoints".format(
            args.task, args.lr, args.update_every, modelname
        )
        if args.DM is True:
            name = "DM-" + name
        save_path = "../checkpoints/{}".format(name)

        callback = []
        # callback.append(WarmupCallback(args.warmupsteps))
        callback.append(GradientClipCallback(clip_value=1))
        # Wandb
        callback.append(
            WandbCallback(
                project="HotpotQAspansentence",
                name=name,
                config={
                    "lr": args.lr,
                    "seed": args.seed,
                    "warmupsteps": args.warmupsteps,
                    "Batch_size": args.batch_size,
                    "epoch": args.epoch,
                    "update_every": args.update_every,
                },
            )
        )
        callback.append(SaveModelCallback(save_dir=save_path, top=3))
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
            # validate_every=10,
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
            validate_every=16,
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
    parser.add_argument("--warmupsteps", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--update_every", default=8, type=int)
    parser.add_argument("--epoch", default=16, type=int)
    parser.add_argument("--model", default="Albert", type=str)
    parser.add_argument("--task", default="QA", type=str)
    parser.add_argument("--DM", action="store_true")
    parser.add_argument("--Debug", action="store_true")

    args = parser.parse_args()
    main(args)
