import torch
from torch.random import seed
import torch.optim as optim

import numpy as np
from fastNLP import Trainer, Tester
from fastNLP import WarmupCallback, SaveModelCallback, GradientClipCallback

from preprocess import HotpotREPipe, HotpotQAPipe

from transformers import DebertaV2TokenizerFast

from metrics import commonLoss, SpanSentenceMetric, DocselectionMetric
from DebertaV2Retriever import DebertaV2Retriever
from DebertaV2Reader import DebertaV2Reader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="QA")
parser.add_argument("--train_path", type=str, default="../HotpotQA/hotpot_train_v1.1.json")
parser.add_argument("--dev_path", type=str, default="../HotpotQA/hotpot_dev_distractor_v1.json")

parser.add_argument("--Debug", action="store_true")
parser.add_argument("--lr", default=5e-6, type=float, help="learning rate")
parser.add_argument("--warmupsteps", default=0.1, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--update_every", default=4, type=int)
parser.add_argument("--epoch", default=16, type=int)

args = parser.parse_args()

device = "cuda:0"
seed = 42
####################
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)

# Special Token
Sentence_token = "</e>"
DOC_token = "</d>"


def main():
    if args.task == "RE":
        if args.Debug:
            train_path = "../HotpotQA/debug.json"
            dev_path = "../HotpotQA/debug.json"
        else:
            train_path = args.train_path
            dev_path = args.dev_path
        re_tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-base")
        re_model = DebertaV2Retriever.from_pretrained("microsoft/deberta-v3-base")
        metrics = DocselectionMetric(save_filename="output.json")

        databundle = HotpotREPipe(tokenizer=re_tokenizer).process_from_file(paths=train_path)
        traindata = databundle.get_dataset("train")
        databundle = HotpotREPipe(tokenizer=re_tokenizer).process_from_file(paths=dev_path)
        devdata = databundle.get_dataset("train")

        loss = commonLoss(loss="loss")
        optimizer = optim.AdamW(re_model.parameters(), lr=args.lr)
        callback = []
        name = "Task-{}-lr-{}-update_every-{}-{}-checkpoints".format(args.task, args.lr, args.update_every, "deberta-v3-base")
        print(name)
        save_path = "../checkpoints/{}".format(name)

        if args.Debug:
            trainer = Trainer(
                model=re_model,
                train_data=traindata,
                device=device,
                dev_data=devdata,
                loss=loss,
                metrics=metrics,
                callbacks=callback,
                optimizer=optimizer,
                n_epochs=512,
                batch_size=args.batch_size,
                update_every=args.update_every,
                dev_batch_size=args.batch_size,
                fp16=True,
                validate_every=16,
            )
            trainer.train()
        else:
            callback.append(WarmupCallback(args.warmupsteps))
            callback.append(GradientClipCallback(clip_value=1))
            callback.append(SaveModelCallback(save_dir=save_path, top=3))

            trainer = Trainer(
                model=re_model,
                train_data=traindata,
                device=device,
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
            )
            trainer.train()

    if args.task == "QA":
        if args.Debug:
            train_path = "../HotpotQA/hotpot-debug.json"
            dev_path = "../HotpotQA/hotpot-debug.json"
        else:
            train_path = args.train_path
            dev_path = args.dev_path
        qa_tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")
        qa_tokenizer.add_tokens([Sentence_token, DOC_token])
        qa_model = DebertaV2Reader.from_pretrained("microsoft/deberta-v3-large")
        qa_model.resize_token_embeddings(len(qa_tokenizer))
        qa_model.config.max_position_embeddings = 1024
        metrics = SpanSentenceMetric(selected_title_list_filename="selected_title.json", save_filename="pred.json", tokenizer=qa_tokenizer)

        databundle = HotpotQAPipe(tokenizer=qa_tokenizer).process_from_file(paths=train_path)
        traindata = databundle.get_dataset("train")
        databundle = HotpotQAPipe(tokenizer=qa_tokenizer).process_from_file(paths=dev_path)
        devdata = databundle.get_dataset("train")

        loss = commonLoss(loss="loss")
        optimizer = optim.AdamW(qa_model.parameters(), lr=args.lr)
        name = "Task-{}-lr-{}-update_every-{}-{}-checkpoints".format(args.task, args.lr, args.update_every, "deberta-v2-xxlarge")
        print(name)

        save_path = "../checkpoints/{}".format(name)

        callback = []
        # qa_tokenizer.save_pretrained("DebertaV2_vob")
        if args.Debug:
            trainer = Trainer(
                model=qa_model,
                train_data=traindata,
                device=device,
                dev_data=devdata,
                loss=loss,
                metrics=metrics,
                callbacks=callback,
                optimizer=optimizer,
                n_epochs=512,
                batch_size=args.batch_size,
                update_every=args.update_every,
                dev_batch_size=args.batch_size,
                fp16=True,
                validate_every=16,
            )
            trainer.train()
        else:
            callback.append(WarmupCallback(args.warmupsteps))
            callback.append(GradientClipCallback(clip_value=1))
            callback.append(SaveModelCallback(save_dir=save_path, top=3))
            trainer = Trainer(
                model=qa_model,
                train_data=traindata,
                device=device,
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
            )
            trainer.train()


if __name__ == "__main__":
    main()
