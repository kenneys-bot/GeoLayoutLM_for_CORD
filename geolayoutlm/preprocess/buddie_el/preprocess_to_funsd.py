'''
Description: 将buddie数据集原始json格式转换为funsd数据集json格式
Author: kenneys-bot
Date: 2025-03-14 13:33:32
LastEditTime: 2025-03-14 16:12:35
LastEditors: kenneys-bot
'''
import os
import json
from typing import Optional
from tqdm import *
import shutil

def convert_buddie_to_funsd(buddie_doc, labels, split):
    """转换单个Buddie文档为FUNSD格式"""
    for itm in tqdm(buddie_doc):
        doc_id = itm['doc_id']
        buddie_tokens = itm['tokens']
        buddie_annotations = itm['annotations']
        funsd_doc = {
            "form": []
        }
        for index, item in tqdm(enumerate(buddie_annotations)):
            # 预定义funsd格式
            data_json = {
                "box": [],
                "text": "",
                "label": "",
                "words": [],
                "linking": [],
                "id": Optional[int]
            }

            # 获取坐标框
            xmin = int(item['bbox'][0])
            ymin = int(item['bbox'][1])
            xmax = int(item['bbox'][2] + item['bbox'][0])
            ymax = int(item['bbox'][3] + item['bbox'][1])

            # 获取标签类型
            class_id = item['class_id']
            if class_id == -1:
                print(doc_id)
            label_name = labels[class_id]['label'] if labels[class_id]['class_id'] == class_id else None
            if not label_name:
                print(f"label_name = {label_name}") 
                import ipdb
                ipdb.set_trace()
            
            data_json['box'] = [xmin, ymin, xmax, ymax]
            data_json['text'] = item['text']
            data_json['label'] = label_name

            tokens_ids = item['token_ids']
            for token_ids in tokens_ids:
                word_json = {
                    "box": [],
                    "text": ""
                }
                tokens = buddie_tokens[token_ids]
                word_json['box'] = [
                    int(tokens['x']),
                    int(tokens['y']),
                    int(tokens['x'] + tokens['width']),
                    int(tokens['y'] + tokens['height'])
                ]
                word_json['text'] = tokens['text']
                data_json['words'].append(word_json)

            data_json['id'] = index
        
            funsd_doc['form'].append(data_json)
        
        # 获取other标签
        for index, item in enumerate(buddie_tokens):
            class_id = item['class_id']
            if class_id != -1:
                continue
            
            
            data_json = {
                "box": [],
                "text": "",
                "label": "",
                "words": [],
                "linking": [],
                "id": Optional[int]
            }
            word_json = {
                "box": [],
                "text": ""
            }

            data_json['box'] = [
                int(item['x']),
                int(item['y']),
                int(item['x'] + item['width']),
                int(item['y'] + item['height'])
            ]
            data_json['text'] = item['text']
            data_json['label'] = 'other'
            word_json['box'] = [
                int(item['x']),
                int(item['y']),
                int(item['x'] + item['width']),
                int(item['y'] + item['height'])
            ]
            word_json['text'] = item['text']
            data_json['words'].append(word_json)
            data_json['id'] = len(funsd_doc['form'])

            funsd_doc['form'].append(data_json)

        this_file_output_path = os.path.join(
            output_path, 
            split if split == "dev" else f"{split}ing_data", 
            "annotations",
            f"{doc_id}.json"
        )
        ori_image_path = os.path.join(root_path, "images", f"{doc_id}.jpg")
        res_image_path = os.path.join(
            output_path, 
            split if split == "dev" else f"{split}ing_data",
            "images",
            f"{doc_id}.jpg"
        )
        with open(this_file_output_path, 'w', encoding='utf-8') as fw:
            json.dump(funsd_doc, fw, indent=4, ensure_ascii=False)
        
        shutil.copy(ori_image_path, res_image_path)


root_path = "/mnt/e/data/datasets/Buddie/Data/Dataset/"
output_path = "/mnt/d/geolayoutlm/V000_20250311/datasets/BUDDIE_v2"
os.makedirs(output_path, exist_ok=True)

train_json_path = os.path.join(root_path, "train.json")
test_json_path = os.path.join(root_path, "test.json")
dev_json_path = os.path.join(root_path, "val.json")
label_json_path = os.path.join(root_path, "labels.json")

labels_data = json.load(open(label_json_path, 'r', encoding='utf-8'))

key_entities_labels = labels_data['key_entities']
doc_class_labels = labels_data['doc_class']

for split in ['train', 'test', 'dev']:
    if split == 'train':
        data = json.load(open(train_json_path, 'r', encoding='utf-8'))
    elif split == 'test':
        data = json.load(open(test_json_path, 'r', encoding='utf-8'))
    elif split == 'dev':
        data = json.load(open(dev_json_path, 'r', encoding='utf-8'))

    os.makedirs(
        os.path.join(
            output_path, 
            split if split == "dev" else f"{split}ing_data", 
            "annotations"
        ),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(
            output_path, 
            split if split == "dev" else f"{split}ing_data", 
            "images"
        ),
        exist_ok=True
    )
    convert_buddie_to_funsd(data, key_entities_labels, split)