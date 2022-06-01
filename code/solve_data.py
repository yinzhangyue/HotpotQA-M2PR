import json
import jsonlines
import ipdb
import re
from copy import deepcopy

SEP = "</e>"
DOC = "</d>"


def load_json(path: str):
    """
    Loads the JSON file of the Squad dataset.
    Returns the json object of the dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Length of data: ", len(data["data"]))
    print("Data Keys: ", data["data"][0].keys())
    print("Title: ", data["data"][0]["title"])

    return data


def parse_squad_data(data: dict) -> list:
    """
    Parses the JSON file of Squad dataset by looping through the
    keys and values and returns a list of dictionaries with
    id, context, question, sentence_labels and answer being the keys of each dict.
    """
    data = data["data"]  # data包含442个article有442个title
    qa_list = []
    answer_list = []
    for paragraphs in data:
        # title
        for para in paragraphs["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                id = qa["id"]
                question = qa["question"]
                # answers/plausible_answers
                for ans in qa["answers"]:
                    answer = ans["text"]
                    context_list = [SEP + " " + con for con in re.split(r"[!.?]", context) if con != ""]
                    sentence_labels = [1 if answer in con else 0 for con in context_list]
                    if answer in answer_list or sum(sentence_labels) != 1:
                        continue

                    answer_list.append(answer)
                    # ans_start = ans["answer_start"]
                    # ans_end = ans_start + len(answer)
                    # id, context, question, answer, sentence_labels
                    qa_dict = {}
                    qa_dict["id"] = id
                    qa_dict["context"] = context_list
                    qa_dict["question"] = question
                    qa_dict["sentence_labels"] = sentence_labels
                    qa_dict["answer"] = answer
                    qa_dict["DOC1_SEP_num"] = len(context_list)
                    qa_dict["sentence_num"] = len(context_list)

                    qa_list.append(qa_dict)
                    break
    return qa_list


def parse_nq_data(data: list) -> list:
    qa_list = []
    for info in data:
        id = info["example_id"]
        question = info["question_text"]
        document_text = info["document_text"].split()
        if info["annotations"][0]["long_answer"]["start_token"] < 0 or info["annotations"][0]["long_answer"]["end_token"] >= len(document_text) or info["annotations"][0]["short_answers"] == [] or info["annotations"][0]["yes_no_answer"] != "NONE":
            continue
        context = " ".join(document_text[info["annotations"][0]["long_answer"]["start_token"] : info["annotations"][0]["long_answer"]["end_token"]])
        context = re.sub("\<.*?\>", "", context).strip()
        answer = " ".join(document_text[info["annotations"][0]["short_answers"][0]["start_token"] : info["annotations"][0]["short_answers"][0]["end_token"]])
        context_list = [SEP + " " + con for con in re.split(r"[!.?]", context) if con != ""]

        qa_dict = {}
        qa_dict["id"] = id
        qa_dict["context"] = context_list
        qa_dict["question"] = question
        qa_dict["sentence_labels"] = [1 if answer in con else 0 for con in context_list]
        qa_dict["answer"] = answer
        qa_dict["DOC1_SEP_num"] = len(context_list)
        qa_dict["sentence_num"] = len(context_list)

        qa_list.append(qa_dict)

    return qa_list


def parse_hotpot_data(data: list) -> list:
    """
    Parses the JSONLINES file of HotpotQA dataset by looping through the
    keys and values and returns a list of dictionaries with
    id, context, question, sentence_labels and answer being the keys of each dict.
    """
    qa_list = []
    data_list = data[0]
    for info in data_list:

        qa_dict = {}
        con_dict = {}
        sup_titles = list(set([sup[0] for sup in info["supporting_facts"]]))
        sup_dict = {sup_titles[0]: [], sup_titles[1]: []}
        sup_len_dict = {}

        for con in info["context"]:
            con_dict[con[0]] = [DOC + " " + con[0]] + [SEP + " " + c for c in con[1]]
            sup_len_dict[con[0]] = len(con[1])
        # [['Allie Goertz', 0], ['Allie Goertz', 1], ['Allie Goertz', 2], ['Milhouse Van Houten', 0]]
        for sup in info["supporting_facts"]:
            if sup[1] < sup_len_dict[sup[0]]:
                sup_dict[sup[0]].append(sup[1])
            else:
                # ipdb.set_trace()
                print("Overflow Warning")

        qa_dict["id"] = info["_id"]
        qa_dict["question"] = info["question"]
        qa_dict["answer"] = info["answer"]

        qa_dict["context"] = con_dict[sup_titles[0]] + con_dict[sup_titles[1]]
        qa_dict["sentence_labels"] = [0 for i in range(sup_len_dict[sup_titles[0]] + sup_len_dict[sup_titles[1]])]
        for idx in sup_dict[sup_titles[0]]:
            qa_dict["sentence_labels"][idx] = 1
        for idx in sup_dict[sup_titles[1]]:
            qa_dict["sentence_labels"][sup_len_dict[sup_titles[0]] + idx] = 1

        qa_dict["DOC1_SEP_num"] = sup_len_dict[sup_titles[0]]
        qa_dict["sentence_num"] = sup_len_dict[sup_titles[0]] + sup_len_dict[sup_titles[1]]
        qa_list.append(deepcopy(qa_dict))
        # ipdb.set_trace()

        qa_dict["context"] = con_dict[sup_titles[1]] + con_dict[sup_titles[0]]
        qa_dict["sentence_labels"] = [0 for i in range(sup_len_dict[sup_titles[1]] + sup_len_dict[sup_titles[0]])]
        for idx in sup_dict[sup_titles[1]]:
            qa_dict["sentence_labels"][idx] = 1
        for idx in sup_dict[sup_titles[0]]:
            qa_dict["sentence_labels"][sup_len_dict[sup_titles[1]] + idx] = 1
        qa_dict["DOC1_SEP_num"] = sup_len_dict[sup_titles[1]]
        qa_dict["sentence_num"] = sup_len_dict[sup_titles[1]] + sup_len_dict[sup_titles[0]]
        qa_list.append(deepcopy(qa_dict))
        # if info["answer"] == "Barton Lee Hazlewood":
        #     ipdb.set_trace()
    return qa_list


def readJsonlinesFile(filename: str):
    item_list = []
    with open(filename, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            item_list.append(item)
    return item_list


def saveFile(data: list, filename: str):
    with jsonlines.open(filename, "w") as out:
        for row in data:
            out.write(row)


if __name__ == "__main__":
    # train_squad_data = load_json("../DATA/train-v2.0.json")
    # train_squad_list = parse_squad_data(train_squad_data)
    # saveFile(train_squad_list, "../DATA/squad_train.jsonl")

    # dev_squad_data = load_json("../DATA/dev-v2.0.json")
    # dev_squad_list = parse_squad_data(dev_squad_data)
    # saveFile(dev_squad_list, "../DATA/squad_dev.jsonl")

    # train_hotpot_data = readJsonlinesFile("../HotpotQA/hotpot_train_v1.1.json")
    # train_hotpot_data = parse_hotpot_data(train_hotpot_data)
    # saveFile(train_hotpot_data, "../HotpotQA/hotpot-train.json")
    # saveFile(train_squad_list+dev_squad_list+train_hotpot_data,"../DATA/train-sq-ht.json")

    dev_hotpot_data = readJsonlinesFile("../HotpotQA/hotpot_dev_distractor_v1.json")
    dev_hotpot_data = parse_hotpot_data(dev_hotpot_data)
    # saveFile(dev_hotpot_data, "../HotpotQA/hotpot-dev.json")
    saveFile(dev_hotpot_data[:16], "../HotpotQA/hotpot-debug.json")

    # train_nq_data = readJsonlinesFile("../DATA/simplified-nq-train.jsonl")
    # train_nq_list = parse_nq_data(train_nq_data)
    # saveFile(train_nq_list, "../DATA/nq-train.jsonl")
