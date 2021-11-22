from fastNLP.core.callback import Callback
from fastNLP import Trainer
from fastNLP import WarmupCallback, SaveModelCallback


from transformers import RobertaTokenizerFast
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import numpy as np
import json

from preprocess import TeacherPipe
from model import ARobertaForTeacherforcingDocselection
from selectionmetric import commonLoss, DocselectionMetric_ids_Teacherforcing

#########Const###########
device = "cuda:0"
seed = 42
lr = 5e-6
warmupsteps = 0
N_epochs = 32
BATCH_SIZE = 4

####################
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)


# data
Hotpot_train_path = "./selected2/hotpot_train_T.json"
Hotpot_dev_path = "./selected2/hotpot_dev_T.json"

# debug
# Hotpot_train_path = "./selected2/debug.json"
# Hotpot_dev_path = "./selected2/debug.json"
modelname = "roberta-large"

endfix = "-lr{}-bz{}-ws{}".format(lr, BATCH_SIZE, warmupsteps)
print(modelname + endfix)

Sentence_token = "</e>"
DOC_token = "</d>"
tokenizer = RobertaTokenizerFast.from_pretrained(modelname)
tokenizer.add_tokens([Sentence_token, DOC_token])

qamodel = ARobertaForTeacherforcingDocselection.from_pretrained(modelname)
qamodel.resize_token_embeddings(len(tokenizer))

databundle = TeacherPipe(tokenizer=tokenizer).process_from_file(paths=Hotpot_train_path)
traindata = databundle.get_dataset("train")

databundle = TeacherPipe(tokenizer=tokenizer).process_from_file(paths=Hotpot_dev_path)
devdata = databundle.get_dataset("train")

metrics = DocselectionMetric_ids_Teacherforcing(tokenizer=tokenizer)
loss = commonLoss(loss="loss")
optimizer = optim.Adam(qamodel.parameters(), lr=lr)

callback = []
callback.append(WarmupCallback(warmupsteps))
callback.append(SaveModelCallback(save_dir=modelname + endfix, top=3))

save_path = modelname + " " + endfix


trainer = Trainer(
    model=qamodel,
    train_data=traindata,
    device=device,
    dev_data=devdata,
    save_path=save_path,
    loss=loss,
    metrics=metrics,
    callbacks=callback,
    optimizer=optimizer,
    n_epochs=N_epochs,
    batch_size=BATCH_SIZE,
    dev_batch_size=4,
    fp16=True,
    # validation_every=64,
)

trainer.train()
