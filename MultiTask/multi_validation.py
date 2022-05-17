from fastNLP.core.callback import Callback
from transformers import BertPreTrainedModel, BertModel, BertLayer
from transformers import (
    RobertaPreTrainedModel,
    RobertaModel,
    BertTokenizerFast,
    RobertaTokenizerFast,
)
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import json
from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP.core.losses import LossBase
from multi_preprocess import HotpotQATestPipe
from torch.nn import CosineSimilarity
import fastNLP
from transformers import RobertaConfig
from multi_model import ARobertaForMulti
from multi_metric import commonLoss, MultiMetric
from multi_preprocess import HotpotQAPipe
from fastNLP import Tester
import torch.optim as optim


####################Const
seed = 42
lr = 1e-6
warmupsteps = 5000
N_epochs = 500
batchsize = 1
device = "cuda:7"
####################
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)


# ################Albertdata
# Hotpot_train_path = "/remote-home/share/zyyin/Hotpot/DATA_multi_task/DATA_albert/hotpot_train_multi_albert.json"
# Hotpot_dev_path="/remote-home/share/zyyin/Hotpot/DATA_multi_task/DATA_albert/hotpot_dev_multi_albert.json"
# modelname="albert-xxlarge-v1"
# tokenizer=AlbertTokenizerFast.from_pretrained(modelname)
# ##########model
# qamodel=AAlbertForQuestionAnswering.from_pretrained(modelname)

# data
Hotpot_train_path = (
    "/remote-home/share/zyyin/Hotpot/DATA_multi_task/hotpot_train_multi_more.json"
)
Hotpot_dev_path = (
    "/remote-home/share/zyyin/Hotpot/DATA_multi_task/hotpot_dev_multi_more.json"
)

modelname = "roberta-base"
Sentence_token = "</e>"
DOC_token = "</d>"
tokenizer = RobertaTokenizerFast.from_pretrained(modelname)
tokenizer.add_tokens([Sentence_token, DOC_token])
# model
config = RobertaConfig.from_pretrained("roberta-base")
bestmodelpath = "save_models/TT-lr-5e-06-update_every-16-roberta-base-checkpoints/2021-11-02-13-33-54-979727/epoch-7_step-631295_D_em-0.932828.pt"
print(bestmodelpath)

qamodel = ARobertaForMulti(config)
bestmodel = torch.load(bestmodelpath).state_dict()
qamodel.resize_token_embeddings(len(tokenizer))
qamodel.load_state_dict(bestmodel)
# traindata
# databundle=HotpotQAPipe(tokenizer=tokenizer).process_from_file(paths=Hotpot_train_path)
# traindata=databundle.get_dataset("train")
# devdata
databundle = HotpotQAPipe(tokenizer=tokenizer).process_from_file(paths=Hotpot_dev_path)
devdata = databundle.get_dataset("train")
##################
# qamodel = ARobertaForMulti.from_pretrained("roberta-base")
# qamodel.resize_token_embeddings(len(tokenizer))
# read pretrained model


metrics = MultiMetric(tokenizer=tokenizer)
loss = commonLoss(loss="loss")

pptrainer = Tester(
    model=qamodel,
    device=device,
    data=devdata,
    loss=loss,
    metrics=metrics,
    batch_size=batchsize,
)
pptrainer.test()
