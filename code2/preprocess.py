from fastNLP import DataSet
from fastNLP.io import Loader
from fastNLP.io import DataBundle
from fastNLP.io import Pipe
import jsonlines
import ipdb


class HotpotTrainLoader(Loader):
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


class HotpotREPipe(Pipe):
    def __init__(self, tokenizer=None):
        self.fields = ["_id", "question", "supporting_facts", "context"]
        self.input_fields = ["question_ids", "document_ids", "doc_num", "doc_length", "question_length"]
        self.target_fields = ["_id", "document_labels"]

        # Tokenizer
        self.tokenizer = tokenizer
        self.max_length = 1024

    def process(self, data_bundle: DataBundle) -> DataBundle:
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

        def create_document_labels(instance):
            document_labels = []
            titles_list = [con[0] for con in instance["context"]]
            titles_label_list = set([sup[0] for sup in instance["supporting_facts"]])
            for titles_label in titles_label_list:
                document_labels.append(titles_list.index(titles_label))
            ipdb.set_trace()

            return {"document_labels": document_labels}

        data_bundle.apply_more(create_document_labels)

        # set input and output
        data_bundle.set_input(*self.input_fields)
        data_bundle.set_target(*self.target_fields)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = HotpotTrainLoader().load(paths)
        return self.process(data_bundle)


class HotpotQAPipe(Pipe):
    def __init__(self, tokenizer=None):
        self.fields = ["_id", "answer", "question", "supporting_facts", "context"]
        self.input_fields = ["question_ids", "document_ids", "doc_num", "doc_length", "question_length"]
        self.target_fields = ["_id", "document_labels"]

        # Tokenizer
        self.tokenizer = tokenizer
        self.max_length = 1024

    def process(self, data_bundle: DataBundle) -> DataBundle:
        def question_tokenize(question):
            tokenized_question = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
            )
            question_ids = tokenized_question["input_ids"][:-1]  # without [SEP]
            return {
                "question_ids": question_ids,
                "question_length": len(question_ids),
            }

        data_bundle.apply_field_more(question_tokenize, field_name="question")

        def document_tokenize(question):
            tokenized_question = self.tokenizer(
                question,
                truncation=True,
                max_length=self.max_length,
            )
            question_ids = tokenized_question["input_ids"][:-1]  # without [SEP]
            return {
                "question_ids": question_ids,
                "question_length": len(question_ids),
            }

        data_bundle.apply_field_more(question_tokenize, field_name="question")
        # set input and output
        data_bundle.set_input(*self.input_fields)
        data_bundle.set_target(*self.target_fields)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        data_bundle = HotpotTrainLoader().load(paths)
        return self.process(data_bundle)


if __name__ == "__main__":
    # print(HotpotTrainLoader().load("../HotpotQA/hotpot_dev_distractor_v1.json").get_dataset("train"))
    # print(HotpotTestLoader().load("../HotpotQA/hotpot_test.json").get_dataset("train"))

    from transformers import DebertaV2TokenizerFast

    qa_tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v2-xlarge")
    print(HotpotREPipe(tokenizer=qa_tokenizer).process_from_file(paths="../HotpotQA/hotpot_dev_distractor_v1.json"))
