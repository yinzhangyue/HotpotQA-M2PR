import torch
from torch.random import seed
import torch.optim as optim

import numpy as np
from fastNLP import Trainer, Tester
from fastNLP import WarmupCallback, SaveModelCallback, GradientClipCallback

from preprocess import HotpotPipe, HotpotQATestPipe
from transformers import (
    RobertaModel,
    RobertaConfig,
    RobertaTokenizerFast,
)
from transformers import (
    DebertaV2Model,
    DebertaV2Config,
    DebertaV2TokenizerFast,
)

from metrics import commonLoss, SpanSentenceMetric, DocselectionMetric
from RobertaRetriever import ARobertaRetriever
from DebertaV2Retriever import ARobertaRetriever
from DebertaV2Reader import ADebertaV2Reader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="QA")
parser.add_argument("--Hotpot_train_path_all", type=str, default="HotpotQA/hotpot_train_v1.1.json")
parser.add_argument("--Hotpot_dev_path_all", type=str, default="HotpotQA/hotpot_dev_distractor_v1.json")

parser.add_argument("--Debug", action="store_true")
parser.add_argument("--lr", default=5e-6, type=float, help="learning rate")
parser.add_argument("--warmupsteps", default=0.1, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--update_every", default=4, type=int)
parser.add_argument("--epoch", default=16, type=int)

parser.add_argument("--Selection_save_filename", type=str, default="Roberta_selection.json")
parser.add_argument("--Selected_titles_path", type=str, default="selected_titles.json")
parser.add_argument("--Output_save_filename", type=str, default="dev_pred.json")
config = parser.parse_args()

device = "cuda:0"
seed = 42
####################
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)
# train Data
Hotpot_train_path_all = config.Hotpot_train_path_all
Hotpot_dev_path_all = config.Hotpot_dev_path_all

if config.Debug:
    Hotpot_train_path_all = "../DATA/debug.json"
    Hotpot_dev_path_all = "../DATA/debug.json"

Selection_save_filename = config.Selection_save_filename
Selected_titles_path = config.Selected_titles_path
Output_save_filename = config.Output_save_filename
# Special Token
Sentence_token = "</e>"
DOC_token = "</d>"


def main():
    if config.mode == "RE":
        # RE: Roberta
        re_tokenizer = RobertaTokenizerFast(
            vocab_file="RobertaConfigs/roberta_vocab.json",
            merges_file="RobertaConfigs/roberta_merges.txt",
            tokenizer_file="RobertaConfigs/roberta_tokenizer.json",
        )
        re_tokenizer.add_tokens([Sentence_token, DOC_token])

        re_model = ARobertaForDocselection.from_pretrained("roberta-large")
        re_model.resize_token_embeddings(len(re_tokenizer))

        metrics = DocselectionMetric(save_filename=Selection_save_filename)

        databundle = HotpotPipe(tokenizer=re_tokenizer).process_from_file(paths=Hotpot_train_path_all)
        traindata = databundle.get_dataset("train")
        databundle = HotpotPipe(tokenizer=re_tokenizer).process_from_file(paths=Hotpot_dev_path_all)
        devdata = databundle.get_dataset("train")

        loss = commonLoss(loss="loss")
        optimizer = optim.AdamW(re_model.parameters(), lr=config.lr)
        callback = []
        name = "Task-{}-lr-{}-update_every-{}-{}-checkpoints".format(config.mode, config.lr, config.update_every, "roberta-large")
        save_path = "../checkpoints/{}".format(name)

        if config.Debug:
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
                batch_size=config.batch_size,
                update_every=config.update_every,
                dev_batch_size=config.batch_size,
                fp16=True,
                validate_every=16,
            )
            trainer.train()
        else:
            callback.append(WarmupCallback(config.warmupsteps))
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
                n_epochs=config.epoch,
                batch_size=config.batch_size,
                update_every=config.update_every,
                dev_batch_size=config.batch_size,
                fp16=True,
            )
            trainer.train()

    if config.mode == "QA":
        # QA: DebertaV2
        # qa_tokenizer = DebertaV2TokenizerFast(vocab_file="DebertaV2Configs/spm.model")
        qa_tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v2-xlarge")
        qa_tokenizer.add_tokens([Sentence_token, DOC_token])

        qa_model = ADebertaV2ForQuestionAnswering.from_pretrained("microsoft/deberta-v2-xlarge")
        qa_model.resize_token_embeddings(len(qa_tokenizer))
        qa_model.config.max_position_embeddings = 1024

        metrics = SpanSentenceMetric(
            selected_title_list_filename=Selected_titles_path,
            save_filename=Output_save_filename,
            tokenizer=qa_tokenizer,
        )

        databundle = HotpotPipe(tokenizer=qa_tokenizer).process_from_file(paths=Hotpot_train_path_all)
        traindata = databundle.get_dataset("train")
        databundle = HotpotPipe(tokenizer=qa_tokenizer).process_from_file(paths=Hotpot_dev_path_all)
        devdata = databundle.get_dataset("train")

        loss = commonLoss(loss="loss")
        optimizer = optim.AdamW(qa_model.parameters(), lr=config.lr)
        name = "Task-{}-lr-{}-update_every-{}-{}-checkpoints".format(config.mode, config.lr, config.update_every, "deberta-v2-xxlarge")
        save_path = "../checkpoints/{}".format(name)

        callback = []
        # qa_tokenizer.save_pretrained("DebertaV2_vob")
        if config.Debug:
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
                batch_size=config.batch_size,
                update_every=config.update_every,
                dev_batch_size=config.batch_size,
                fp16=True,
                validate_every=16,
            )
            trainer.train()
        else:
            callback.append(WarmupCallback(config.warmupsteps))
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
                n_epochs=config.epoch,
                batch_size=config.batch_size,
                update_every=config.update_every,
                dev_batch_size=config.batch_size,
                fp16=True,
            )
            trainer.train()


if __name__ == "__main__":
    main()
