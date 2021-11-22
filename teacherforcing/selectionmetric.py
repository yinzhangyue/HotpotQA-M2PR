from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

from fastNLP.core.metrics import MetricBase, seq_len_to_mask
from fastNLP.core.losses import LossBase
from torch.nn import CosineSimilarity
import fastNLP
import copy
import ipdb

from model import *


class commonLoss(LossBase):
    r"""
    MSE损失函数
    """

    def __init__(self, loss=None):
        super(commonLoss, self).__init__()
        self._init_param_map(loss=loss)

    def get_loss(self, loss):

        return loss


class DocselectionMetric_ids_Teacherforcing(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(
        self,
        tokenizer=None,
        doc_out="doc_out",
        labels_T="labels_T",
    ):

        r"""
        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()
        self.num = 0
        self._init_param_map(doc_out=doc_out, labels_T=labels_T)
        self.tokenizer = tokenizer
        self.metrics = {"acc": 0}
        self.outputdict = {}

    def evaluate(self, doc_out, labels_T):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        Batch_size = doc_out.size(0)
        self.num += Batch_size
        # ipdb.set_trace()
        self.metrics["acc"] += torch.sum(doc_out.squeeze().eq(labels_T)).item()

    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """

        evaluate_result = copy.deepcopy(self.metrics)

        for k in evaluate_result.keys():
            evaluate_result[k] /= self.num
        if reset:
            self.num = 0
            for key in self.metrics.keys():
                self.metrics[key] = 0
        return evaluate_result
