from transformers import AlbertTokenizerFast, RobertaTokenizerFast
from collections import OrderedDict
from fastNLP import DataSet
from fastNLP.io import Loader
from fastNLP.io import DataBundle
from fastNLP.io import Pipe
import pandas as pd
import jsonlines
import ipdb


class HotpotQALoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = [
            "id",
            "question",
            "docs",
            "doc_label",
            "answer",
            "answer_type",
            "context",
            "context_selected",
            "gold_doc_gold_sentence",
            "starchar",
            "doc_label_2",
        ]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with jsonlines.open(fpath) as f:
            for data in f:
                for field in fields:
                    datadict[field].append(data[field])

        return DataSet(datadict)

    def download(self, dev_ratio=0.1, re_download=False) -> str:
        pass


class HotpotQAPipe(Pipe):
    def __init__(self, tokenizer=None):
        self.fields = [
            "id",
            "question",
            "docs",
            "doc_label",
            "answer",
            "answer_type",
            "context",
            "gold_doc_gold_sentence",
            "starchar",
            "doc_label_2",
        ]
        self.input_fields = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "start_positions",
            "end_positions",
            "sentence_index_start",
            "sentence_index_end",
            "sentence_labels",
            "answer_type",
            "sentence_num",
            "question_ids",  # question tokenizer 去除结尾[SEP]
            "document_ids",  # document tokenizer 去除结尾[SEP]
            "document_labels",  # doc_label
            "gold_doc_pair",  # index [1,4]
            "question_attention_mask",  # tokenizer
            "document_attention_mask",  # tokenizer
            "doc_num",  # docs数量
            "doc_length",  # list 每个doc的长度
            "DOC_index",  # question tokenizer的长度 + 1
        ]
        # Tokenizer
        self.tokenizer = tokenizer
        # SEP
        self.SEP_token = "</e>"
        self.SEP_id = tokenizer.convert_tokens_to_ids(self.SEP_token)

        self.max_length = 1024
        self.target = [
            "id",
            "answer",
            "sentence_num",
            "answer_type",
            "sentence_labels",
            "document_labels",  # doc_label
            "gold_doc_pair",  # index [1,4]
            "doc_num",
            "gold_answer_doc",
        ]

    def process(self, data_bundle: DataBundle) -> DataBundle:
        def question_tokenize(question):
            tokenized_question = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
            )
            question_ids = tokenized_question["input_ids"][:-1]
            question_attention_mask = tokenized_question["attention_mask"][:-1]
            DOC_index = len(question_ids) + 1
            # ipdb.set_trace()
            return {
                "DOC_index": DOC_index,
                "question_ids": question_ids,
                "question_attention_mask": question_attention_mask,
            }

        data_bundle.apply_field_more(question_tokenize, field_name="question")

        def document_tokenize(document):
            doc_num = len(document)
            doc_length = [
                len(i)
                for i in self.tokenizer(
                    document, truncation=True, max_length=self.max_length
                )["input_ids"]
            ]
            tokenized_document = self.tokenizer(
                document,
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            document_ids = tokenized_document["input_ids"]
            document_attention_mask = tokenized_document["attention_mask"]
            # ipdb.set_trace()
            return {
                "doc_num": doc_num,
                "doc_length": doc_length,
                "document_ids": document_ids,
                "document_attention_mask": document_attention_mask,
            }

        data_bundle.apply_field_more(document_tokenize, field_name="docs")
        data_bundle.copy_field("doc_label", "document_labels")

        def find_gold_doc_pair(document_labels):
            gold_doc_pair = []
            length_document_label = len(document_labels)
            for i in range(length_document_label):
                if document_labels[i] >= 1:
                    gold_doc_pair.append(i)
            return {"gold_doc_pair": gold_doc_pair}

        data_bundle.apply_field_more(find_gold_doc_pair, field_name="document_labels")

        def find_answer_document(instance):
            gold_answer_doc = []
            document_labels = instance["doc_label_2"]
            length_document_label = len(document_labels)
            for i in range(length_document_label):
                if document_labels[i] == 2:
                    gold_answer_doc.append(i)
            if len(gold_answer_doc) == 0:
                gold_answer_doc = instance["gold_doc_pair"]

            return {"gold_answer_doc": gold_answer_doc}

        data_bundle.apply_more(find_answer_document)

        def _tokenize(instance):
            question = instance["question"]
            context = instance["context"]
            output = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            # ipdb.set_trace()

            sequence_ids = output.sequence_ids(0)
            d = dict(output)
            d.update({"sequence_ids": sequence_ids})
            return d

        input_fields = self.input_fields

        data_bundle.apply_more(_tokenize)

        # input_ids,attention_mask,token_type_ids,offsets_mapping
        def find_SEP(input_ids):
            SEP_id = self.SEP_id
            SEP_index = []
            length_input_ids = len(input_ids)
            for i in range(length_input_ids):
                if input_ids[i] == SEP_id:
                    SEP_index.append(i)
            # start&end
            sentence_index_start = SEP_index[:-1]
            sentence_index_end = SEP_index[1:]

            return {
                "sentence_index_start": sentence_index_start,
                "sentence_index_end": sentence_index_end,
            }

        data_bundle.apply_field_more(find_SEP, field_name="input_ids")

        def find_start_end_positions(instance):
            input_ids = instance["input_ids"]
            answer = instance["answer"]
            sequence_ids = instance["sequence_ids"]
            answer_start = instance["starchar"]
            offsets = instance["offset_mapping"]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            if answer_start == -1:
                return {"start_positions": cls_index, "end_positions": cls_index}
            else:
                # Start/end character index of the answer in the text.
                start_char = answer_start
                end_char = start_char + len(answer)

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    # print(sequence_ids[token_start_index])
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    return {"start_positions": cls_index, "end_positions": cls_index}
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    token_start_index = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    token_end_index = token_end_index + 1
                return {
                    "start_positions": token_start_index,
                    "end_positions": token_end_index,
                }

        data_bundle.apply_more(find_start_end_positions)

        def check_sep_cut(instance):
            gold = instance["gold_doc_gold_sentence"]
            cutofflen = len(instance["sentence_index_start"])
            if len(gold) > cutofflen:
                return {"sentence_labels": gold[:cutofflen]}
            else:
                return {"sentence_labels": gold}

        data_bundle.apply_more(check_sep_cut)
        data_bundle.apply_field(lambda x: len(x), "sentence_labels", "sentence_num")

        # set input and output
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*self.target)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = HotpotQALoader().load(paths)
        return self.process(data_bundle)


class HotpotQATestPipe(Pipe):
    def __init__(self, tokenizer=None):
        self.fields = [
            "id",
            "question",
            "docs",
            "doc_label",
            "answer",
            "answer_type",
            "context_selected",
            "gold_doc_gold_sentence",
            "starchar",
            "doc_label_2",
        ]
        self.input_fields = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "sentence_index_start",
            "sentence_index_end",
            "sentence_num",
            "question_ids",  # question tokenizer 去除结尾[SEP]
            "document_ids",  # document tokenizer 去除结尾[SEP]
            "question_attention_mask",  # tokenizer
            "document_attention_mask",  # tokenizer
            "doc_num",  # docs数量
            "doc_length",  # list 每个doc的长度
            "DOC_index",  # question tokenizer的长度 + 1
        ]
        # Tokenizer
        self.tokenizer = tokenizer
        # SEP
        self.SEP_token = "</e>"
        self.SEP_id = tokenizer.convert_tokens_to_ids(self.SEP_token)
        self.max_length = 512
        self.target = [
            "id",
            "answer",
            "sentence_num",
            "doc_num",
            "sentence_labels",
            "gold_doc_pair",  # index [1,4]
            "doc_num",
            "gold_answer_doc",
        ]

    def process(self, data_bundle: DataBundle) -> DataBundle:
        def question_tokenize(question):
            tokenized_question = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
            )
            question_ids = tokenized_question["input_ids"][:-1]
            question_attention_mask = tokenized_question["attention_mask"][:-1]
            DOC_index = len(question_ids) + 1
            # ipdb.set_trace()
            return {
                "DOC_index": DOC_index,
                "question_ids": question_ids,
                "question_attention_mask": question_attention_mask,
            }

        data_bundle.apply_field_more(question_tokenize, field_name="question")

        def document_tokenize(document):
            doc_num = len(document)
            doc_length = [
                len(i)
                for i in self.tokenizer(
                    document, truncation=True, max_length=self.max_length
                )["input_ids"]
            ]
            tokenized_document = self.tokenizer(
                document,
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            document_ids = tokenized_document["input_ids"]
            document_attention_mask = tokenized_document["attention_mask"]
            # ipdb.set_trace()
            return {
                "doc_num": doc_num,
                "doc_length": doc_length,
                "document_ids": document_ids,
                "document_attention_mask": document_attention_mask,
            }

        data_bundle.apply_field_more(document_tokenize, field_name="docs")
        data_bundle.copy_field("doc_label", "document_labels")

        def find_gold_doc_pair(document_labels):
            gold_doc_pair = []
            length_document_label = len(document_labels)
            for i in range(length_document_label):
                if document_labels[i] >= 1:
                    gold_doc_pair.append(i)
            return {"gold_doc_pair": gold_doc_pair}

        data_bundle.apply_field_more(find_gold_doc_pair, field_name="document_labels")

        def find_answer_document(instance):
            gold_answer_doc = []
            document_labels = instance["doc_label_2"]
            length_document_label = len(document_labels)
            for i in range(length_document_label):
                if document_labels[i] == 2:
                    gold_answer_doc.append(i)
            if len(gold_answer_doc) == 0:
                gold_answer_doc = instance["gold_doc_pair"]

            return {"gold_answer_doc": gold_answer_doc}

        data_bundle.apply_more(find_answer_document)

        def _tokenize(instance):
            question = instance["question"]
            context_selected = instance["context_selected"]
            output = self.tokenizer(
                question,
                context_selected,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )

            sequence_ids = output.sequence_ids(0)
            d = dict(output)
            d.update({"sequence_ids": sequence_ids})
            return d

        input_fields = self.input_fields

        data_bundle.apply_more(_tokenize)

        # input_ids,attention_mask,token_type_ids,offsets_mapping
        def find_SEP(input_ids):
            SEP_id = self.SEP_id
            SEP_index = []
            length_input_ids = len(input_ids)
            for i in range(length_input_ids):
                if input_ids[i] == SEP_id:
                    SEP_index.append(i)
            # start&end
            sentence_index_start = SEP_index[:-1]
            sentence_index_end = SEP_index[1:]

            return {
                "sentence_index_start": sentence_index_start,
                "sentence_index_end": sentence_index_end,
            }

        data_bundle.apply_field_more(find_SEP, field_name="input_ids")

        def check_sep_cut(instance):
            gold = instance["gold_doc_gold_sentence"]
            cutofflen = len(instance["sentence_index_start"])
            if len(gold) > cutofflen:
                return {"sentence_labels": gold[:cutofflen]}
            else:
                return {"sentence_labels": gold}

        data_bundle.apply_more(check_sep_cut)
        data_bundle.apply_field(lambda x: len(x), "sentence_labels", "sentence_num")

        # set input and output
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*self.target)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = HotpotQALoader().load(paths)
        return self.process(data_bundle)


# if __name__ == "__main__":
#     Hotpot_train_path = "../DATA/hotpot_train_multi.json"
#     Hotpot_dev_path = "../DATA/hotpot_dev_multi.json"
#     # modelname = "albert-xxlarge-v1"
#     # tokenizer = AlbertTokenizerFast.from_pretrained(modelname)
#     modelname = "roberta-large"
#     tokenizer = RobertaTokenizerFast.from_pretrained(modelname)

#     data = (
#         HotpotQAPipe(tokenizer=tokenizer)
#         .process_from_file(Hotpot_dev_path)
#         .get_dataset("train")
#     )
#     print(data[:5])
