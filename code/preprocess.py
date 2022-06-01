from transformers import AlbertTokenizerFast, RobertaTokenizerFast
from collections import OrderedDict
from fastNLP import DataSet
from fastNLP.io import Loader
from fastNLP.io import DataBundle
from fastNLP.io import Pipe
import jsonlines
import ipdb


class HotpotRELoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = [
            "_id",
            "answer",
            "question",
            "supporting_facts",
            "context",
        ]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with jsonlines.open(fpath) as f:
            for data in f:
                for field in fields:
                    for i in range(len(data)):
                        datadict[field].append(data[i][field])

        return DataSet(datadict)

    def download(self) -> str:
        pass


class HotpotREPipe(Pipe):
    def __init__(self, tokenizer=None):
        self.fields = ["_id", "question", "answer", "supporting_facts", "context"]
        self.input_fields = ["question_ids", "document_ids", "doc_length", "question_length", "gold_doc_pair", "gold_answer_doc", "doc_num"]
        self.target_fields = ["id", "gold_doc_pair", "gold_answer_doc", "doc_num"]

        # Tokenizer
        self.tokenizer = tokenizer
        self.max_length = 512

    def process(self, data_bundle: DataBundle) -> DataBundle:
        data_bundle.copy_field("_id", "id")

        def question_tokenize(question):
            tokenized_question = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
            )
            question_ids = tokenized_question["input_ids"]

            return {
                "question_ids": question_ids,
                "question_length": len(question_ids),
            }

        data_bundle.apply_field_more(question_tokenize, field_name="question")

        def document_tokenize(context):
            doc_length = []
            document_ids = []
            doc_num = len(context)
            for i in range(doc_num):
                tokenized_document = self.tokenizer(
                    "".join([context[i][0]] + context[i][1]),
                    truncation=True,
                    max_length=self.max_length,
                )
                doc_length.append(len(tokenized_document["input_ids"][1:]))  # without [CLS]
                document_ids.append(tokenized_document["input_ids"][1:])  # without [CLS]

            return {
                "doc_length": doc_length,
                "document_ids": document_ids,
                "doc_num": doc_num,
            }

        data_bundle.apply_field_more(document_tokenize, field_name="context")

        def create_gold_doc_pair_and_answer_doc(instance):
            gold_doc_pair = []
            gold_answer_doc = -1
            titles_list = [con[0] for con in instance["context"]]
            context_list = ["".join(con[1]) for con in instance["context"]]
            titles_label_list = set([sup[0] for sup in instance["supporting_facts"]])
            for titles_label in titles_label_list:
                gold_doc_pair.append(titles_list.index(titles_label))
            for gold_doc in gold_doc_pair:
                if instance["answer"] in context_list[gold_doc]:
                    gold_answer_doc = gold_doc

            return {"gold_doc_pair": gold_doc_pair, "gold_answer_doc": gold_answer_doc}

        data_bundle.apply_more(create_gold_doc_pair_and_answer_doc)

        # set input and output
        data_bundle.set_input(*self.input_fields)
        data_bundle.set_target(*self.target_fields)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = HotpotRELoader().load(paths)
        return self.process(data_bundle)


class HotpotQALoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = ["id", "question", "answer", "context", "sentence_labels", "DOC1_SEP_num", "sentence_num"]
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
        self.fields = ["id", "question", "answer", "context", "sentence_labels", "DOC1_SEP_num", "sentence_num"]
        self.input_fields = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "start_positions",
            "end_positions",
            "sentence_index",
            "sentence_labels",
            "answer_type",
            "sentence_num",
        ]
        self.target = [
            "id",
            "sentence_labels",
            "input_ids",
            "answer",
            "sentence_num",
            "DOC1_SEP_num",  # DOC1 sep的数量
        ]

        # Tokenizer
        self.tokenizer = tokenizer
        # SEP
        self.SEP_id = tokenizer.convert_tokens_to_ids("</e>")
        # DOC
        self.DOC_id = tokenizer.convert_tokens_to_ids("</d>")

        self.max_length = 1024

    def process(self, data_bundle: DataBundle) -> DataBundle:
        def _tokenize(instance):
            question = instance["question"]
            context = " ".join(instance["context"])
            output = self.tokenizer(
                question,
                context,
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

        def find_start_end_positions(instance):
            input_ids = instance["input_ids"]
            answer = instance["answer"]
            sequence_ids = instance["sequence_ids"]
            start_char = (" ".join(instance["context"])).find(answer)
            offsets = instance["offset_mapping"]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            if start_char == -1:
                return {"start_positions": cls_index, "end_positions": cls_index}
            else:
                # Start/end character index of the answer in the text.
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
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    return {"start_positions": cls_index, "end_positions": cls_index}
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
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

        def find_SEP(input_ids):
            SEP_index = []
            length_input_ids = len(input_ids)
            for i in range(length_input_ids):
                if input_ids[i] == self.SEP_id:
                    SEP_index.append(i)
            return {"sentence_index": SEP_index}

        data_bundle.apply_field_more(find_SEP, field_name="input_ids")

        def parse_answer_type(answer):
            answer_type = 2
            if answer == "no":
                answer_type = 0
            elif answer == "yes":
                answer_type = 1

            return {"answer_type": answer_type}

        data_bundle.apply_field_more(parse_answer_type, field_name="answer")
        # set input and output
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*self.target)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = HotpotQALoader().load(paths)
        return self.process(data_bundle)


class HotpotTestLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, fpath: str):
        fields = [
            "_id",
            "question",
            "context",
        ]
        datadict = {}
        for field in fields:
            datadict[field] = []
        with jsonlines.open(fpath) as f:
            for data in f:
                for field in fields:
                    for i in range(len(data)):
                        datadict[field].append(data[i][field])

        return DataSet(datadict)

    def download(self, dev_ratio=0.1, re_download=False) -> str:
        pass
