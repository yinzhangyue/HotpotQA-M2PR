from transformers import AlbertTokenizerFast
from collections import OrderedDict
from fastNLP import DataSet
from fastNLP.io import Loader
from fastNLP.io import DataBundle
from fastNLP.io import Pipe
import pandas as pd
import jsonlines
import ipdb


class TeacherLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = ["id", "question", "answer", "context_T", "labels_T"]
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


class TeacherPipe(Pipe):
    def __init__(self, tokenizer=None):
        self.fields = ["id", "question", "answer", "context_T", "labels_T"]
        self.input_fields = [
            "teacher_ids",
            "teacher_attention_mask",
            "DOC_index",
            "labels_T",
        ]
        # Tokenizer
        self.tokenizer = tokenizer
        # SEP
        self.SEP_token = "</e>"
        self.SEP_id = tokenizer.convert_tokens_to_ids(self.SEP_token)
        self.max_length = 512
        self.target = ["labels_T"]

    def process(self, data_bundle: DataBundle) -> DataBundle:
        def _tokenize_Teacherforcing(instance):
            question = instance["question"]
            context = instance["context_T"]

            output = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            tokenized_question = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
            )
            question_ids = tokenized_question["input_ids"][:-1]

            d = dict()
            d["teacher_ids"] = output.input_ids
            d["teacher_attention_mask"] = output.attention_mask

            DOC_index = len(question_ids) + 1
            d["DOC_index"] = DOC_index
            return d

        data_bundle.apply_more(_tokenize_Teacherforcing)

        # set input and output
        data_bundle.set_input(*self.input_fields)
        data_bundle.set_target(*self.target)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = TeacherLoader().load(paths)
        return self.process(data_bundle)
