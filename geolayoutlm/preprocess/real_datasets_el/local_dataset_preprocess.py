import os
from glob import glob
from tqdm import tqdm
import json
import cv2
import argparse
import imagesize
from transformers import BertTokenizer
from PIL import Image, ImageDraw, ImageFont
from difflib import SequenceMatcher
import re

parser = argparse.ArgumentParser(
    description='process locally real datasets.'
)
parser.add_argument(
    '--inputpath', help='传入真实数据集路径'
)
parser.add_argument(
    '--outputpath', help='输出路径'
)
parser.add_argument(
    '--voca', help='预处理模型', default='../hlf/bert-base-uncased'
)
parser.add_argument(
    '--model_type', help='模型类型', default='bert'
)
args = parser.parse_args()
MAX_SEQ_LENGTH = 512
MODEL_TYPE = args.model_type
VOCA = args.voca

CLASSES = [
    "O", "SELL_QUESTION", "SELL_ANSWER", "BUY_QUESTION", "BUY_ANSWER",
    "DATE_QUESTION", "DATE_ANSWER", "NUMBER_QUESTION", "NUMBER_ANSWER",
    "AMOUNT_QUESTION", "AMOUNT_ANSWER"
]
CLASSES_VALID = [
    "SELL_QUESTION", "SELL_ANSWER", "BUY_QUESTION", "BUY_ANSWER",
    "DATE_QUESTION", "DATE_ANSWER", "NUMBER_QUESTION", "NUMBER_ANSWER",
    "AMOUNT_QUESTION", "AMOUNT_ANSWER"
]
INPUT_PATH = args.inputpath
anno_dir = 'annotations'
OUTPUT_PATH = args.outputpath
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "preprocessed"), exist_ok=True)


def main():
    tokenize = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)

    for dataset_split in ['train', 'val']:
        print(f"dataset_split: {dataset_split}")
        do_preprocess(tokenize, dataset_split)
    os.system(f"cp -r {os.path.join(INPUT_PATH, 'training_data')} {OUTPUT_PATH}")
    os.system(f"cp -r {os.path.join(INPUT_PATH, 'testing_data')} {OUTPUT_PATH}")
    save_class_names()


def do_preprocess(tokenizer, dataset_split):
    if dataset_split == 'train':
        dataset_root_path = os.path.join(INPUT_PATH, 'training_data')
    elif dataset_split == 'val':
        dataset_root_path = os.path.join(INPUT_PATH, 'testing_data')
    else:
        raise ValueError(f"Invalid dataset_split={dataset_split}")

    json_files = glob(os.path.join(dataset_root_path, anno_dir, '*.json'))
    preprocessed_fnames = []
    for idx_json, json_file in tqdm(enumerate(json_files)):
        json_file = json_file.replace('\\', '/')
        in_json_obj = json.load(open(json_file, "r", encoding="utf-8"))

        # local real datasets' json to FUNSD's json
        in_json_obj = real2funsd(in_json_obj)

        out_json_obj = {}
        out_json_obj['blocks'] = {'first_token_idx_list': [], 'boxes': []}
        out_json_obj["words"] = []
        out_json_obj["parse"] = {"class": {}}
        for c in CLASSES:
            out_json_obj["parse"]["class"][c] = []
        out_json_obj["parse"]["relations"] = []

        form_id_to_word_idx = {}  # record the word index of the first word of each block, starting from 0
        other_seq_list = {}
        num_tokens = 0

        # words
        for form_idx, form in enumerate(in_json_obj['form']):
            form_id = form['id']
            form_text = form['text'].strip()
            form_label = form['label']
            if form_label.startswith('O'):
                form_label = "O"
            form_linking = form['linking']
            form_box = form['box']

            if len(form_text) == 0:
                continue  # 过滤含有空文本的文本块

            word_cnt = 0
            class_seq = []
            real_word_idx = 0
            for word_idx, word in enumerate(form['words']):
                word_text = word['text']
                bb = word['box']
                bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
                tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))

                word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
                if len(word_text) != 0:  # 过滤空的words块
                    out_json_obj["words"].append(word_obj)
                    if real_word_idx == 0:
                        out_json_obj['blocks']['first_token_idx_list'].append(num_tokens + 1)
                    num_tokens += len(tokens)

                    word_cnt += 1
                    class_seq.append(len(out_json_obj['words']) - 1)  # word的索引值
                    real_word_idx += 1
            if real_word_idx > 0:
                out_json_obj['blocks']['boxes'].append(form_box)

            is_valid_class = False if form_label not in CLASSES else True
            if is_valid_class:
                out_json_obj["parse"]["class"][form_label].append(class_seq)
                form_id_to_word_idx[form_id] = len(out_json_obj["words"]) - word_cnt
            else:
                other_seq_list[form_id] = class_seq

        # parse
        for form_idx, form in enumerate(in_json_obj["form"]):
            form_id = form["id"]
            form_text = form["text"].strip()
            form_linking = form["linking"]

            if len(form_linking) == 0:
                continue

            for link_idx, link in enumerate(form_linking):
                if link[0] == form_id:
                    if (
                            link[1] in form_id_to_word_idx
                            and link[0] in form_id_to_word_idx
                    ):
                        relation_pair = [
                            form_id_to_word_idx[link[0]],
                            form_id_to_word_idx[link[1]],
                        ]
                        out_json_obj["parse"]["relations"].append(relation_pair)

        # meta
        out_json_obj['meta'] = {}
        image_file = (os.path.splitext(json_file.replace(f"/{anno_dir}/", "/images/"))[0] + ".jpg")
        if not os.path.exists(image_file):
            continue
        if dataset_split == "train":
            out_json_obj['meta']['image_path'] = image_file[image_file.find("training_data/"):]
        elif dataset_split == "val":
            out_json_obj['meta']['image_path'] = image_file[image_file.find("testing_data/"):]
        width, height = imagesize.get(image_file)
        out_json_obj['meta']['imageSize'] = {"width": width, "height": height}
        out_json_obj['meta']['voca'] = VOCA

        this_file_name = os.path.basename(json_file)

        # Save file name to list
        preprocessed_fnames.append(os.path.join("preprocessed", this_file_name).replace('\\', '/'))

        # Save to file
        data_obj_file = os.path.join(OUTPUT_PATH, "preprocessed", this_file_name)
        with open(data_obj_file, "w", encoding="UTF-8") as fw:
            json.dump(out_json_obj, fw, indent=4, ensure_ascii=False)

        # Save file name list file
    preprocessed_fnames_file = os.path.join(OUTPUT_PATH, f"preprocessed_files_{dataset_split}.txt")
    with open(preprocessed_fnames_file, "w", encoding="UTF-8") as fw:
        fw.write("\n".join(preprocessed_fnames))




def real2funsd(in_json_obj):
    """
    将本地真实数据集的json格式转换成FUNSD格式

    :param in_json_obj: 输入的真实数据集json格式
    :return: 已转换完成的json
    """
    in_json_obj = in_json_obj['items']
    data = {"form": []}
    for idx, itm in enumerate(in_json_obj):
        temp_json = {}
        temp_json.update({"box": itm['coords']})
        temp_json.update({"text": itm['text']})
        temp_json.update({"label": itm['item']})

        words = []
        temp_json_words = {}
        temp_json_words.update({"box": itm['coords']})
        temp_json_words.update({"text": itm['text']})
        words.append(temp_json_words)

        temp_json.update({"words": words})
        temp_json.update({"linking": []})
        temp_json.update({"id": idx})
        data['form'].append(temp_json)
    add_linking_group(data)
    other_To_O(data)
    return data


def other_To_O(data):
    for itm in data['form']:
        if itm['label'] == 'other':
            itm['label'] = 'O'
    return data


def add_linking_group(data):
    """
    添加关系链接，作用于linking

    :param data: 以funsd's json格式修改后的real dataset's json内容
    :return:
    """
    for i in range(len(data['form'])):
        for j in range(len(data['form'])):
            if data['form'][i]['label'] == 'SELL_QUESTION' and data['form'][j]['label'] == 'SELL_ANSWER':
                if [i, j] in data['form'][j]['linking']:
                    continue
                data['form'][i]['linking'].append([i, j])
                data['form'][j]['linking'].append([i, j])
                continue
            elif data['form'][i]['label'] == 'BUY_QUESTION' and data['form'][j]['label'] == 'BUY_ANSWER':
                if [i, j] in data['form'][j]['linking']:
                    continue
                data['form'][i]['linking'].append([i, j])
                data['form'][j]['linking'].append([i, j])
                continue
            elif data['form'][i]['label'] == 'AMOUNT_QUESTION' and data['form'][j]['label'] == 'AMOUNT_ANSWER':
                if [i, j] in data['form'][j]['linking']:
                    continue
                data['form'][i]['linking'].append([i, j])
                data['form'][j]['linking'].append([i, j])
                continue
            elif data['form'][i]['label'] == 'DATE_QUESTION' and data['form'][j]['label'] == 'DATE_ANSWER':
                if [i, j] in data['form'][j]['linking']:
                    continue
                data['form'][i]['linking'].append([i, j])
                data['form'][j]['linking'].append([i, j])
                continue
            elif data['form'][i]['label'] == 'NUMBER_QUESTION' and data['form'][j]['label'] == 'NUMBER_ANSWER':
                if [i, j] in data['form'][j]['linking']:
                    continue
                data['form'][i]['linking'].append([i, j])
                data['form'][j]['linking'].append([i, j])
                continue


def save_class_names():
    with open(
        os.path.join(OUTPUT_PATH, "class_names.txt"), "w", encoding="utf-8"
    ) as fp:
        fp.write("\n".join(CLASSES))


if __name__ == '__main__':
    main()
