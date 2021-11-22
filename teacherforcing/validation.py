
from fastNLP.core.callback import Callback
from transformers import BertPreTrainedModel,BertModel,BertLayer
from transformers import RobertaPreTrainedModel,RobertaModel,BertTokenizerFast,RobertaTokenizerFast
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import json
from fastNLP.core.metrics import MetricBase,seq_len_to_mask
from fastNLP.core.losses import LossBase
from preprocess import HotpotQAPipe,HotpotQATestPipe
from torch.nn import CosineSimilarity
import fastNLP
from transformers import RobertaConfig
from transformers.models.albert import AlbertTokenizerFast,AlbertModel,AlbertForQuestionAnswering,AlbertPreTrainedModel
from model import (ARobertaForDocselection,)
# from model_nosatrans import ARobertaForQuestionAnswering,ABertForQuestionAnswering,AAlbertForQuestionAnswering


from selectionmetric import commonLoss,DocselectionMetric
####################Const
device="cuda:0"

seed=42
lr=1e-6
warmupsteps=5000
N_epochs=500
BATCH_SIZE=16
batchsize=1
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

################data

Hotpot_train_path = "/remote-home/share/zyyin/Hotpot/DATA_multi_task/hotpot_train_multi.json"
Hotpot_dev_path="/remote-home/share/zyyin/Hotpot/DATA_multi_task/hotpot_dev_multi.json"
modelname="roberta-base"

endfix="_sep</e>_docselection"

Sentence_token="</e>"
DOC_token="</d>"
tokenizer=RobertaTokenizerFast.from_pretrained(modelname)
tokenizer.add_tokens([Sentence_token,DOC_token])
###########model
config=RobertaConfig.from_pretrained("roberta-base")
bestmodelpath="/remote-home/yxwang/NLP/HotpotQA/Docselection/_roberta-base_sep</e>_docselection/2021-09-27-03-16-07-591433/epoch-7_step-631295_em-0.941901.pt"
qamodel=ARobertaForDocselection(config)
bestmodel=torch.load(bestmodelpath).state_dict()
qamodel.resize_token_embeddings(len(tokenizer))
qamodel.load_state_dict(bestmodel)
########traindata
# databundle=HotpotQAPipe(tokenizer=tokenizer).process_from_file(paths=Hotpot_train_path)
# traindata=databundle.get_dataset("train")
# #########devdata
databundle=HotpotQAPipe(tokenizer=tokenizer).process_from_file(paths=Hotpot_dev_path)
devdata=databundle.get_dataset("train")

qamodel.resize_token_embeddings(len(tokenizer))
###############################read pretrained model
# bestmodelpath="AllphosT/best__2deepchargeModelms2_all_mediancos_2021-06-02-11-38-03-717982"
# bestmodel=torch.load(bestmodelpath).state_dict()
# deepms2.load_state_dict(bestmodel)
###########Trainer
# qamodel.use_answertrans=True
# qamodel.use_sentencetrans=True

from fastNLP import Const
metrics=DocselectionMetric(tokenizer=tokenizer,selected_pair="selected_pair", gold_doc_pair="gold_doc_pair",doc_num="doc_num")
from fastNLP import MSELoss
loss=commonLoss(loss="loss")
import torch.optim as optim
optimizer=optim.Adam(qamodel.parameters(),lr=lr)
from fastNLP import WarmupCallback,SaveModelCallback
callback=[]
callback.append(WarmupCallback(warmupsteps))
callback.append(WandbCallback(project="HotpotQAdocselection",name="_"+modelname+endfix,config={"lr":lr,"seed":seed,
"Batch_size":BATCH_SIZE,"warmupsteps":warmupsteps}))
callback.append(SaveModelCallback(save_dir="_"+modelname+endfix,top=-1))

############trainer
from fastNLP import Tester

save_path="_"+modelname+endfix


pptrainer=Tester(model=qamodel,device=device,data=devdata,
                  loss=loss,metrics=metrics,
                   batch_size=batchsize)
pptrainer.test()