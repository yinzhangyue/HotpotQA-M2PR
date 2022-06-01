import torch
import torch.optim as optim

import os
import numpy as np
import ipdb

# import wandb

from transformers import (
    RobertaTokenizer,
    RobertaTokenizerFast,
    AlbertTokenizerFast,
    DebertaTokenizerFast,
    DebertaV2TokenizerFast,
)
from fastNLP import WarmupCallback, SaveModelCallback, GradientClipCallback
from fastNLP import Trainer

# from wandbcallback import WandbCallback
from preprocessV2 import HotpotQAPipe, HotpotQATestPipe
from model import (
    ARobertaForQuestionAnswering,
    ABertForQuestionAnswering,
    AAlbertForQuestionAnswering,
    ADebertaV1ForQuestionAnswering,
    ADebertaV2ForQuestionAnswering,
)
from metrics import commonLoss, SpanSentenceMetric, DocselectionMetric
from docselection_model import (
    ARobertaForDocselection,
    ADebertaV2ForDocselection,
    ADebertaV2ForDocselection512,
)
from electra import AElectraForQuestionAnswering
import argparse
from transformers import AutoTokenizer


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
        Hotpot_train_path = "/home/ma-user/work/yxwang/HotpotData/data/debug.json"
        Hotpot_dev_path = "/home/ma-user/work/yxwang/HotpotData/data/debug.json"
        # Hotpot_train_path = "../DATA/single.json"
        # Hotpot_dev_path = "../DATA/single.json"

    else:
        # more
        Hotpot_train_path = "/home/ma-user/work/yxwang/HotpotData/HotpotQAGitee/DATA/hotpot_train_multi_more.json"
        Hotpot_dev_path = "/home/ma-user/work/yxwang/HotpotData/HotpotQAGitee/DATA/hotpot_dev_multi_more.json"
        
        # all
        # Hotpot_train_path = "/home/ma-user/work/yxwang/HotpotData/data/hotpot_train_multi_all.json"
        # Hotpot_dev_path = "/home/ma-user/work/yxwang/HotpotData/data/hotpot_dev_multi_all.json"

    if args.model == "Albert":
        # Albert
        # modelname = "albert-large-v1"
        modelname = "albert-xxlarge-v1"
        tokenizer = AlbertTokenizerFast.from_pretrained(modelname)
        tokenizer.add_tokens([Sentence_token, DOC_token])
    elif args.model == "Bert":
        # Bert
        modelname = "bert-large-cased"
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        tokenizer.add_tokens([Sentence_token, DOC_token])

    elif args.model == "Roberta":
        # Roberta
        modelname = "roberta-large"
        tokenizer = RobertaTokenizerFast.from_pretrained(modelname)
        tokenizer.add_tokens([Sentence_token, DOC_token])

    elif args.model == "DebertaV1":
        # DebertaV1
        # modelname = "microsoft/deberta-base"
        modelname = "microsoft/deberta-xlarge"
        tokenizer = DebertaTokenizerFast.from_pretrained(modelname)
        tokenizer.add_tokens([Sentence_token, DOC_token])

    elif args.model == "DebertaV2":
        # DebertaV1
        # modelname = "microsoft/deberta-base"
        modelname = "microsoft/deberta-v2-xxlarge"
        tokenizer = DebertaV2TokenizerFast.from_pretrained(modelname)
        tokenizer.add_tokens([Sentence_token, DOC_token])

    elif args.model == "DebertaV2-512":
        # DebertaV1
        # modelname = "microsoft/deberta-base"
        modelname = "microsoft/deberta-v2-xlarge"
        tokenizer = DebertaV2TokenizerFast.from_pretrained(modelname)
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
        if args.model == "Albert":
            # Albert Model
            qamodel = AAlbertForQuestionAnswering.from_pretrained(modelname)
        elif args.model == "Roberta":
            # Roberta Model
            # checkpoint = "/remote-home/zyyin/Experiment/Hotpot/MultiTask/save_models/lr-5e-06-update_every-16-roberta-base-checkpoints/2021-11-02-13-18-53-501964/epoch-7_step-631295_D_em-0.933776.pt"
            # print(checkpoint)
            qamodel = ARobertaForQuestionAnswering.from_pretrained(modelname)
            # checkpoint = torch.load(checkpoint).state_dict()
            # qamodel.load_state_dict(checkpoint)

        elif args.model == "Electra":
            # Roberta Model
            qamodel = AElectraForQuestionAnswering.from_pretrained(modelname)
        elif args.model == "Bert":
            qamodel = ABertForQuestionAnswering.from_pretrained(modelname)
        elif args.model == "DebertaV1":
            qamodel = ADebertaV1ForQuestionAnswering.from_pretrained(modelname)
        elif args.model == "DebertaV2":
            qamodel = ADebertaV2ForQuestionAnswering.from_pretrained(modelname)

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
        if args.model == "DebertaV2":
            qamodel = ADebertaV2ForDocselection.from_pretrained(modelname)

        if args.model == "DebertaV2-512":
            qamodel = ADebertaV2ForDocselection512.from_pretrained(modelname)
            qamodel.config.max_position_embeddings = 1024
            
        elif args.model == "Roberta":
            # Roberta Model
            qamodel = ARobertaForDocselection.from_pretrained(modelname)
            # max_pos=1026
            # new_pos_embed = qamodel.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, 1024)
            # new_pos_embed[:514]=qamodel.roberta.embeddings.position_embeddings.weight[:]
            # new_pos_embed[514:]=qamodel.roberta.embeddings.position_embeddings.weight[2:]
            # qamodel.roberta.embeddings.position_embeddings.weight.data=new_pos_embed
            # qamodel.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

        metrics = DocselectionMetric(
            tokenizer=tokenizer,
            selected_pair="selected_pair",
            gold_doc_pair="gold_doc_pair",
            doc_num="doc_num",
        )

    qamodel.resize_token_embeddings(len(tokenizer))

    # tokenizer.save_pretrained("Roberta_vob")
    # qamodel.save_pretrained("Roberta_config")
    # read pretrained model
    # bestmodelpath=""
    # bestmodel=torch.load(bestmodelpath).state_dict()
    # deepms2.load_state_dict(bestmodel)

    loss = commonLoss(loss="loss")
    optimizer = optim.AdamW(qamodel.parameters(), lr=args.lr)

    if not args.Debug:
        name = "New-{}-seed-{}-lr-{}-update_every-{}-{}-checkpoints".format(
            args.task, args.seed, args.lr, args.update_every, modelname.replace("/", "")
        )
        if args.DM is True:
            name = "DM-" + name
        save_path = "../checkpoints/{}".format(name)
        print("Save path", name)
        callback = []
        callback.append(WarmupCallback(args.warmupsteps))
        callback.append(GradientClipCallback(clip_value=1))
        # # Wandb
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
            n_epochs=500,
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
    parser.add_argument("--warmupsteps", default=0.1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--update_every", default=8, type=int)
    parser.add_argument("--epoch", default=16, type=int)
    parser.add_argument("--model", default="Albert", type=str)
    parser.add_argument("--task", default="QA", type=str)
    parser.add_argument("--DM", action="store_true")
    parser.add_argument("--Debug", action="store_true")

    args = parser.parse_args()
    main(args)
