from transformers import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaTokenizerFast,
)

import torch
import torch.nn.functional as F
from torch.random import seed
import torch.optim as optim

import numpy as np
from fastNLP import Tester

import fastNLP

from preprocess import HotpotQAPipe, HotpotQATestPipe
from model import (
    ARobertaForQuestionAnswering,
    ABertForQuestionAnswering,
    AAlbertForQuestionAnswering,
)
from metrics import commonLoss, SpanSentenceMetric, DocselectionMetric
from docselection_model import ARobertaForDocselection

device = "cuda:0"
BATCH_SIZE = 1
seed = 2021
####################
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)
# dev data
# checkpoint = "../checkpoints/lr-5e-06-update_every-2-albert-xxlarge-v1-checkpoints/2021-09-29-11-53-11-016480/epoch-14_step-315658_em-0.692443.pt"
checkpoint = "/remote-home/zyyin/Experiment/Hotpot/Two_phase/checkpoints/lr-5e-06-update_every-4-albert-xxlarge-v1-checkpoints/2021-09-29-11-57-28-479919/epoch-15_step-338205_em-0.691360.pt"

Sentence_token = "</e>"
DOC_token = "</d>"


Hotpot_dev_path = "../selection/Roberta_selection_save_file1.json"
# Hotpot_dev_path = "../DATA/debug.json"
# Hotpot_dev_path = "../DATA/hotpot_dev_multi_more.json"


print(Hotpot_dev_path)
mode = "QA"

if mode == "RE":
    # # Albert
    # modelname = "albert-xxlarge-v1"
    # tokenizer = AlbertTokenizerFast.from_pretrained(modelname)
    # tokenizer.add_tokens([Sentence_token, DOC_token])
    # qamodel = AAlbertForQuestionAnswering.from_pretrained(modelname)

    # Roberta
    modelname = "roberta-large"
    tokenizer = RobertaTokenizerFast.from_pretrained(modelname)
    tokenizer.add_tokens([Sentence_token, DOC_token])
    qamodel = ARobertaForDocselection.from_pretrained(modelname)

    qamodel.resize_token_embeddings(len(tokenizer))

    # metrics = SpanSentenceMetric(tokenizer=tokenizer)
    doc_select_3_save_filename = "../selection/Roberta-large-doc_select_3_train.json"
    save_filename = "../selection/Roberta-large-selectiondict_train.json"

    metrics = DocselectionMetric(
        doc_select_3_save_filename=doc_select_3_save_filename,
        save_filename=save_filename,
    )

    # FastNLP
    checkpoint = torch.load(checkpoint).state_dict()
    qamodel.load_state_dict(checkpoint)

    # devdata
    # databundle = HotpotQATestPipe(tokenizer=tokenizer).process_from_file(
    #     paths=Hotpot_dev_path
    # )

elif mode == "QA":
    # Roberta
    modelname = "roberta-large"
    tokenizer = RobertaTokenizerFast.from_pretrained(modelname)
    tokenizer.add_tokens([Sentence_token, DOC_token])
    qamodel = ARobertaForQuestionAnswering.from_pretrained(modelname)

    # Albert
    modelname = "albert-xxlarge-v1"
    tokenizer = AlbertTokenizerFast.from_pretrained(modelname)
    tokenizer.add_tokens([Sentence_token, DOC_token])
    qamodel = AAlbertForQuestionAnswering.from_pretrained(modelname)

    qamodel.resize_token_embeddings(len(tokenizer))
    metrics = SpanSentenceMetric(tokenizer)

    # FastNLP
    checkpoint = torch.load(checkpoint).state_dict()
    qamodel.load_state_dict(checkpoint)


print("##### {} #####".format(modelname))
databundle = HotpotQATestPipe(tokenizer=tokenizer).process_from_file(
    paths=Hotpot_dev_path
)
devdata = databundle.get_dataset("train")

loss = commonLoss(loss="loss")

# tester
tester = Tester(
    model=qamodel,
    data=devdata,
    device=device,
    metrics=metrics,
    batch_size=BATCH_SIZE,
)
tester.test()
