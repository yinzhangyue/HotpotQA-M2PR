import json
import jsonlines
import ipdb

SEP = "</e>"


def readJsonFile(filename):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    # print(data[0].keys())

    return data


def readJsonlinesFile(filename):
    item_list = []
    with open(filename, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            item_list.append(item)
    return item_list


def saveFile(data, filename):
    with jsonlines.open(filename, "w") as out:
        for row in data:
            out.write(row)


def selectDoc(more_doc_list, select_doc_dict):
    return_list = []
    for more_doc in more_doc_list:
        more_doc_docs = more_doc["docs"]
        more_doc_id = more_doc["id"]
        more_doc_gold_sentence = more_doc["gold_sentence"]
        select_doc = eval(select_doc_dict[more_doc_id])
        select_doc.sort()

        context_selected = ""
        gold_doc_gold_sentence = []

        for i in select_doc:
            context_selected += more_doc_docs[i]
            # print(i, more_doc_gold_sentence)
            gold_doc_gold_sentence.extend(more_doc_gold_sentence[i])
        context_selected += "</e>"
        more_doc["context_selected"] = context_selected
        more_doc["gold_doc_gold_sentence"] = gold_doc_gold_sentence
        return_list.append(more_doc.copy())
    return return_list


def changeDocName(two_doc_list):
    return_list = []
    for two_doc in two_doc_list:
        two_doc["context_selected"] = two_doc["context"]
        return_list.append(two_doc.copy())
    return return_list


def generate_full_doc(
    select_doc_filename,
    savefilename,
    two_doc_filename="../DATA/hotpot_dev_multi_2.json",
    more_doc_filename="../DATA/hotpot_dev_multi_more.json",
):
    two_doc_list = readJsonlinesFile(two_doc_filename)
    more_doc_list = readJsonlinesFile(more_doc_filename)
    select_doc_dict = readJsonFile(select_doc_filename)
    # ipdb.set_trace()
    return_list1 = selectDoc(more_doc_list, select_doc_dict)
    return_list2 = changeDocName(two_doc_list)
    return_list = return_list1 + return_list2
    # for i in return_list:
    #     if "context_selected" not in i.keys():
    #         print(i)
    saveFile(return_list, savefilename)


if __name__ == "__main__":
    select_doc_filename = "../selection/Roberta_selection_result.json"
    savefilename = "../selection/Roberta_selection_save_file1.json"
    generate_full_doc(select_doc_filename, savefilename)
    # save_file = readJsonlinesFile("../selection/Roberta_selection_save_file1.json")
    # print(save_file[0])
