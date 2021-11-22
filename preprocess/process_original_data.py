import json
import jsonlines
import ipdb
from transformers import AlbertTokenizer, AlbertModel, BertTokenizer, RobertaTokenizer
from datasets import list_datasets, load_dataset
from pprint import pprint

SEP = "</e>"
DOC = "</d>"


def readFile(filename):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    # print(data[0].keys())

    return data


def saveFile(data, filename):
    with jsonlines.open(filename, "w") as out:
        for row in data:
            out.write(row)


def inSupportingFacts(supporting_facts, title, idx):
    tag = False
    for supporting_fact in supporting_facts:
        if title == supporting_fact[0] and idx == supporting_fact[1]:
            tag = True
    return tag


def find_index(tokenizer_list, tag="[SEP]"):
    result = []
    length_tokenizer_list = len(tokenizer_list)
    for i in range(length_tokenizer_list):
        if tokenizer_list[i] == tag:
            result.append(i)
    return result


def find_startchar(answer, gold_doc):
    return gold_doc.find(answer)


# def read_Squad_data():
#     # squad_dataset = load_dataset("squad")
#     squad_train = load_dataset("squad", split="train")
#     squad_valid = load_dataset("squad", split="validation")
#     return squad_train, squad_valid


# def process_Squad_data(data_list):
#     assert "question" in data_list.keys(), "question not in keys"
#     question = data_list["question"]
#     assert "answer" in data_list.keys(), "answer not in keys"
#     answer = data_list["answer"]
#     assert "context" in data_list.keys(), "context not in keys"
#     context_list = data_list["context"]
#     assert "title" in data_list.keys(), "title not in keys"


# 处理原始数据
def processOriginalData(data_list):
    outdata_list = []
    for data in data_list:
        outdata = {}
        ### 检查是否包含对应的keys ###
        assert "question" in data.keys(), "question not in keys"
        question = data["question"]
        assert "answer" in data.keys(), "answer not in keys"
        answer = data["answer"]
        assert "context" in data.keys(), "context not in keys"
        context_list = data["context"]
        assert "supporting_facts" in data.keys(), "supporting_facts not in keys"
        supporting_facts = data["supporting_facts"]
        doc_num = len(context_list)

        gold_doc_title = set(
            [supporting_fact[0] for supporting_fact in supporting_facts]
        )
        # 处理answer_type(No/Yes/Span分别用0/1/2表示)
        answer_type = 2
        if answer == "no":
            answer_type = 0
        elif answer == "yes":
            answer_type = 1

        # 处理context [SEP] title [SEP] sen1 [SEP] ... [SEP] sen_end
        context_processed_list = []
        doc_label_list = []
        for context in context_list:
            title = context[0]
            sentences = DOC
            num_temp = 0
            for sentence in context[1]:
                sentences += SEP + sentence.strip()
                # 取前14句话
                num_temp += 1
                if num_temp >= 14:
                    break
            context = sentences
            context_processed_list.append(context)

            if title in gold_doc_title:
                doc_label_list.append(1)
            else:
                doc_label_list.append(0)

        doc_label_list_2 = []
        for context in context_list:
            title = context[0]
            content = context[1]
            if title in gold_doc_title:
                if answer in title + " " + " ".join(content):
                    doc_label_list_2.append(2)
                else:
                    doc_label_list_2.append(1)
            else:
                doc_label_list_2.append(0)

        # if 2 not in doc_label_list:
        #     print(doc_label_list)
        #     print(data["_id"])
        #     print(answer)
        #     print(content)
        #     exit(0)

        # 处理gold_sentence
        gold_doc_gold_sentence = []
        gold_sentence_dict = {}
        gold_sentence_list = []

        for context in context_list:
            gold_sentence = []
            title = context[0]
            sentence_num = len(context[1])
            if sentence_num > 14:
                sentence_num = 14
            for idx in range(sentence_num):
                if inSupportingFacts(supporting_facts, title, idx):
                    gold_sentence.append(1)
                else:
                    gold_sentence.append(0)

            gold_sentence_dict[title] = gold_sentence
            gold_sentence_list.append(gold_sentence)

        # 处理gold_doc
        gold_doc = []
        for i in range(doc_num):
            if context_list[i][0] in gold_doc_title:
                gold_doc.append(context_processed_list[i])
                gold_doc_gold_sentence.append(gold_sentence_dict[context_list[i][0]])
        # print(gold_doc)
        gold_doc = gold_doc[0] + gold_doc[1]

        gold_doc_gold_sentence = gold_doc_gold_sentence[0] + gold_doc_gold_sentence[1]

        if doc_num == 2:
            continue
        # if doc_num > 2:
        #     continue

        outdata["id"] = data["_id"]
        outdata["question"] = question
        outdata["docs"] = context_processed_list
        outdata["doc_label"] = doc_label_list
        outdata["doc_label_2"] = doc_label_list_2
        outdata["answer"] = answer
        outdata["answer_type"] = answer_type

        # outdata["question_and_context"] = [question, gold_doc[0]]
        outdata["context"] = gold_doc + SEP
        outdata["gold_sentence"] = gold_sentence_list
        outdata["gold_doc_gold_sentence"] = gold_doc_gold_sentence
        outdata["starchar"] = find_startchar(answer, gold_doc)
        outdata["context_selected"] = ""

        outdata_list.append(outdata.copy())

    # 最后一个[SEP]不要 title的[SEP]不要
    # 14, 18, 39, 48, 102, 106, 139
    # sentence_index_start sentence_index_end

    return outdata_list


if __name__ == "__main__":
    print(SEP, DOC)
    train_file_path = "./DATA/hotpot_train_v1.1.json"
    dev_file_path = "./DATA/hotpot_dev_distractor_v1.json"

    train_output_path = "./DATA_processed/hotpot_train_multi_more.json"
    dev_output_path = "./DATA_processed/hotpot_dev_multi_more.json"

    data_list = readFile(train_file_path)
    data_list = processOriginalData(data_list)
    saveFile(data_list, train_output_path)
