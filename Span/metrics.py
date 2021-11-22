import torch
import copy
from fastNLP.core.metrics import MetricBase
from fastNLP.core.losses import LossBase
from model import *
from collections import Counter
import string
import re
import ipdb


class commonLoss(LossBase):
    def __init__(self, loss=None):
        super(commonLoss, self).__init__()
        self._init_param_map(loss=loss)

    def get_loss(self, loss):
        return loss


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class DocselectionMetric(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(
        self,
        tokenizer=None,
        selected_pair="selected_pair",
        gold_doc_pair="gold_doc_pair",
        doc_num="doc_num",
        gold_answer_doc="gold_answer_doc",
        id="id",
        doc_select_3="doc_select_3",
        doc_select_3_save_filename="../save_file_tmp/doc_select_3.json",
        save_filename="../save_file_tmp/selectiondict.json",
    ):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()
        self.num = 0
        self._init_param_map(
            selected_pair=selected_pair,
            gold_doc_pair=gold_doc_pair,
            doc_num=doc_num,
            gold_answer_doc=gold_answer_doc,
            id=id,
            doc_select_3=doc_select_3,
        )
        self.tokenizer = tokenizer
        self.metrics = {"em": 0, "f1": 0, "prec": 0, "recall": 0, "Gold": 0}
        self.outputdict = {}
        self.doc_select_3_dict = {}
        self.doc_select_3_save_filename = doc_select_3_save_filename
        self.save_filename = save_filename

    def evaluate(
        self, selected_pair, gold_doc_pair, doc_num, gold_answer_doc, id, doc_select_3
    ):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        Batch_size = selected_pair.size(0)
        self.num += Batch_size

        for b in range(Batch_size):
            id_ = id[b]
            selected = selected_pair[b].long()
            gold = gold_doc_pair[b].long()
            goldanswer = gold_answer_doc[b].long()
            docnum = doc_num[b]

            self.outputdict[id_] = str(list(selected.detach().cpu().numpy()))
            self.doc_select_3_dict[id_] = str(list(doc_select_3.detach().cpu().numpy()))

            if Counter(selected.detach().cpu().numpy()) == Counter(
                gold.detach().cpu().numpy()
            ):
                self.metrics["em"] += 1
            for select in selected:
                if select in goldanswer:
                    self.metrics["Gold"] += 1
                    break
            tp, fp, fn = 0, 0, 0
            selectedlist = torch.zeros(docnum)
            goldlist = torch.zeros(docnum)
            for t in selected:
                selectedlist[t] = 1
            for t in gold:
                goldlist[t] = 1
            for s in range(docnum):
                if selectedlist[s] == 1:
                    if goldlist[s] == 1:
                        tp += 1
                    else:
                        fp += 1
                elif goldlist[s] == 1:
                    fn += 1
            prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
            self.metrics["f1"] += f1
            self.metrics["prec"] += prec
            self.metrics["recall"] += recall

    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        with open(self.save_filename, "w", encoding="utf-8") as f:
            json.dump(self.outputdict, f, ensure_ascii=False)
        with open(self.doc_select_3_save_filename, "w", encoding="utf-8") as f:
            json.dump(self.doc_select_3_dict, f, ensure_ascii=False)

        evaluate_result = copy.deepcopy(self.metrics)
        for k in evaluate_result.keys():
            evaluate_result[k] /= self.num
        if reset:
            self.num = 0
            for key in self.metrics.keys():
                self.metrics[key] = 0
        return evaluate_result


class SpanSentenceMetric(MetricBase):
    r"""
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(
        self,
        tokenizer=None,
        type_logits="type_logits",
        start_logits="start_logits",
        end_logits="end_logits",
        sentence_predictions="sentence_predictions",
        sentence_labels="sentence_labels",
        input_ids="input_ids",
        answer="answer",
        sentence_num="sentence_num",
    ):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()
        self.num = 0
        self._init_param_map(
            type_logits="type_logits",
            start_logits="start_logits",
            end_logits="end_logits",
            sentence_predictions="sentence_predictions",
            sentence_labels="sentence_labels",
            input_ids="input_ids",
            answer="answer",
            sentence_num="sentence_num",
        )
        self.tokenizer = tokenizer
        self.metrics = {
            "em": 0,
            "f1": 0,
            "prec": 0,
            "recall": 0,
            "sp_em": 0,
            "sp_f1": 0,
            "sp_prec": 0,
            "sp_recall": 0,
            "joint_em": 0,
            "joint_f1": 0,
            "joint_prec": 0,
            "joint_recall": 0,
        }

    def evaluate(
        self,
        type_logits,
        start_logits,
        end_logits,
        sentence_predictions,
        sentence_labels,
        input_ids,
        answer,
        sentence_num,
    ):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        Batch_size = type_logits.size(0)
        self.num += Batch_size
        for batch in range(Batch_size):
            type_logit, start, end, prediction, gold = (
                type_logits[batch],
                start_logits[batch],
                end_logits[batch],
                sentence_predictions[batch],
                sentence_labels[batch],
            )
            text = answer[batch]
            sentence = sentence_num[batch]
            ids = input_ids[batch]

            anstype = torch.argmax(type_logit).item()
            if anstype == 0:
                answers = "no"
            elif anstype == 1:
                answers = "yes"
            else:
                start = torch.argmax(start).item()
                end = torch.argmax(end).item()
                span_id = ids[start : end + 1]
                # ipdb.set_trace()
                if span_id is None:
                    answers = ""
                else:
                    answers = self.tokenizer.decode(span_id)
                    answers = normalize_answer(answers)
            # print("Target:{}\nPred:{}".format(text, answers))
            text = normalize_answer(text)
            em = answers == text
            normalized_prediction = answers
            normalized_ground_truth = text
            # print("gold:", answers, " predict:", text)
            # ipdb.set_trace()
            ZERO_METRIC = (0, 0, 0)
            if (
                normalized_prediction in ["yes", "no"]
                and normalized_prediction != normalized_ground_truth
            ):
                f1, precision, recall = ZERO_METRIC
            elif (
                normalized_ground_truth in ["yes", "no"]
                and normalized_prediction != normalized_ground_truth
            ):
                f1, precision, recall = ZERO_METRIC
            else:
                prediction_tokens = normalized_prediction.split()
                ground_truth_tokens = normalized_ground_truth.split()
                common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    f1, precision, recall = ZERO_METRIC
                else:
                    precision = 1.0 * num_same / len(prediction_tokens)
                    recall = 1.0 * num_same / len(ground_truth_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)
            self.metrics["em"] += float(em)
            self.metrics["f1"] += f1
            self.metrics["prec"] += precision
            self.metrics["recall"] += recall

            tp, fp, fn = ZERO_METRIC

            for s in range(sentence):
                if prediction[s] == 1:
                    if gold[s] == 1:
                        tp += 1
                    else:
                        fp += 1
                elif gold[s] == 1:
                    fn += 1
            sp_prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
            sp_recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
            sp_f1 = (
                2 * sp_prec * sp_recall / (sp_prec + sp_recall)
                if sp_prec + sp_recall > 0
                else 0.0
            )
            sp_em = 1.0 if fp + fn < 1 else 0.0
            self.metrics["sp_em"] += sp_em
            self.metrics["sp_f1"] += sp_f1
            self.metrics["sp_prec"] += sp_prec
            self.metrics["sp_recall"] += sp_recall

            joint_prec = precision * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.0
            joint_em = em * sp_em

            self.metrics["joint_em"] += joint_em
            self.metrics["joint_f1"] += joint_f1
            self.metrics["joint_prec"] += joint_prec
            self.metrics["joint_recall"] += joint_recall

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
