import torch
from torch.random import seed
import numpy as np
from fastNLP import Tester

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
from re_model import ARobertaForDocselection
from code2.DebertaV2Reader import ADebertaV2ForQuestionAnswering
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="QA")
parser.add_argument("--Hotpot_dev_path_more", type=str, default="../DATA/hotpot_dev_multi_more.json")
parser.add_argument("--Hotpot_dev_path_all", type=str, default="../DATA/hotpot_dev_multi_all.json")

parser.add_argument("--Selection_save_filename", type=str, default="Roberta_selection.json")
parser.add_argument("--Selected_titles_path", type=str, default="selected_titles.json")
parser.add_argument("--Output_save_filename", type=str, default="dev_pred.json")


config = parser.parse_args()

device = "cuda:6"
seed = 42
####################
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)
# Dev Data
Hotpot_dev_path_more = config.Hotpot_dev_path_more
Hotpot_dev_path_all = config.Hotpot_dev_path_all

Selection_save_filename = config.Selection_save_filename
Selected_titles_path = config.Selected_titles_path
Output_save_filename = config.Output_save_filename
# Checkpoint
re_checkpoint = ""
qa_checkpoint = ""


def main():
    if config.mode == "RE":
        # RE: Roberta
        re_tokenizer = RobertaTokenizerFast(
            vocab_file="RobertaConfigs/roberta_vocab.json",
            merges_file="RobertaConfigs/roberta_merges.txt",
            tokenizer_file="RobertaConfigs/roberta_tokenizer.json",
        )
        re_tokenizer.add_tokens([Sentence_token, DOC_token])

        re_model = ARobertaForDocselection(config=RobertaConfig.from_pretrained("RobertaConfigs/roberta_config.json"))
        re_model.resize_token_embeddings(len(re_tokenizer))

        metrics = DocselectionMetric(save_filename=Selection_save_filename)
        re_model.load_state_dict(torch.load(re_checkpoint).state_dict())

        databundle = HotpotPipe(tokenizer=re_tokenizer).process_from_file(paths=Hotpot_dev_path_more)
        devdata = databundle.get_dataset("train")

        loss = commonLoss(loss="loss")

        tester = Tester(
            model=re_model,
            data=devdata,
            device=device,
            metrics=metrics,
            batch_size=1,
        )
        tester.test()

    if config.mode == "QA":
        # QA: DebertaV2
        qa_tokenizer = DebertaV2TokenizerFast.from_pretrained("DebertaV2_vob")
        # qa_tokenizer.add_tokens([Sentence_token, DOC_token])

        qa_model = ADebertaV2ForQuestionAnswering(config=DebertaV2Config.from_pretrained("DebertaV2Configs/config.json"))
        qa_model.resize_token_embeddings(len(qa_tokenizer))
        qa_model.config.max_position_embeddings = 1024
        metrics = SpanSentenceMetric(
            selected_title_list_filename=Selected_titles_path,
            save_filename=Output_save_filename,
            tokenizer=qa_tokenizer,
        )
        qa_model.load_state_dict(torch.load(qa_checkpoint).state_dict())

        databundle = HotpotPipe(tokenizer=qa_tokenizer).process_from_file(paths=Hotpot_dev_path_all)
        devdata = databundle.get_dataset("train")

        loss = commonLoss(loss="loss")

        tester = Tester(
            model=qa_model,
            data=devdata,
            device=device,
            metrics=metrics,
            batch_size=1,
        )
        tester.test()


if __name__ == "__main__":
    main()
