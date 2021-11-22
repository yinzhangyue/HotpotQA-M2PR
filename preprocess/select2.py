import json
import jsonlines
import ipdb
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


# 处理原始数据
def processOriginalData(data_list, selected3_dict):
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
        id = data["_id"] + "-"

        doc_num = len(context_list)
        gold_doc_title = set(
            [supporting_fact[0] for supporting_fact in supporting_facts]
        )

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
            # 是gold_doc
            if title in gold_doc_title:
                doc_label_list.append(1)
            else:
                doc_label_list.append(0)

        outdata["question"] = question
        outdata["answer"] = answer

        id_ = 0
        gold_doc_idx = []

        for i in range(doc_num):
            outdata["id"] = id + str(id_)
            id_ += 1

            outdata["context_T"] = context_processed_list[i]
            outdata["labels_T"] = doc_label_list[i]
            outdata_list.append(outdata.copy())

            if doc_label_list[i] == 1:
                gold_doc_idx.append(i)

        if doc_num > 2:
            selected3 = eval(selected3_dict[id[:-1]])
            tag = False
            for i in range(3):
                for j in range(i, 3):

                    outdata["context_T"] = (
                        context_processed_list[selected3[i]]
                        + context_processed_list[selected3[j]]
                    )
                    if (
                        doc_label_list[selected3[i]] == 1
                        and doc_label_list[selected3[j]] == 1
                    ):
                        outdata["labels_T"] = 1
                        tag = True
                    else:
                        outdata["labels_T"] = 0

                    outdata["id"] = id + str(id_)
                    id_ += 1
                    outdata_list.append(outdata.copy())

        if not tag:
            outdata["context_T"] = (
                context_processed_list[gold_doc_idx[0]]
                + context_processed_list[gold_doc_idx[1]]
            )
            outdata["labels_T"] = 1
            outdata["id"] = id + str(id_)
            outdata_list.append(outdata.copy())

    # 最后一个[SEP]不要 title的[SEP]不要
    # 14, 18, 39, 48, 102, 106, 139
    # sentence_index_start sentence_index_end

    return outdata_list


if __name__ == "__main__":
    print(SEP, DOC)
    # File = "train"
    File = "dev"

    if File == "train":
        train_file_path = "./DATA/hotpot_train_v1.1.json"
        train_selected3_file = "./selected2/Roberta-large-doc_select_3_train.json"
        train_output_path = "./selected2/hotpot_train_T.json"

        data_list = readFile(train_file_path)
        selected3_dict = readFile(train_selected3_file)
        data_list = processOriginalData(data_list, selected3_dict)
        saveFile(data_list, train_output_path)

    elif File == "dev":
        dev_file_path = "./DATA/hotpot_dev_distractor_v1.json"
        dev_selected3_file = "./selected2/Roberta-large-doc_select_3_dev.json"
        dev_output_path = "./selected2/hotpot_dev_T.json"

        data_list = readFile(dev_file_path)
        selected3_dict = readFile(dev_selected3_file)
        data_list = processOriginalData(data_list, selected3_dict)
        saveFile(data_list, dev_output_path)
