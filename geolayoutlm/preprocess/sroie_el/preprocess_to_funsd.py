'''
Description: 
Author: kenneys-bot
Date: 2025-03-15 03:08:59
LastEditTime: 2025-03-15 20:11:28
LastEditors: kenneys-bot
'''
import os
import json
from typing import Optional
from tqdm import *
import shutil
from glob import glob

import json
import re
from difflib import SequenceMatcher

def parse_txt(txt_content):
    entities = []
    lines = txt_content.strip().split('\n')
    for line_num, line in enumerate(lines):
        parts = line.strip().split(',')
        if len(parts) < 9:
            continue
        try:
            coords = list(map(int, parts[:8]))
        except ValueError:
            continue
        
        text = ','.join(parts[8:]).strip()
        # 将坐标转换为四个点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        bbox = [
            [coords[0], coords[1]],
            [coords[2], coords[3]],
            [coords[4], coords[5]],
            [coords[6], coords[7]]
        ]
        entities.append({
            "line_num": line_num,
            "text": text,
            "bbox": bbox
        })
    return entities

def find_matching_entity(entities, target, threshold=0.8):
    target_clean = re.sub(r'\W+', '', target).lower()
    best_match = None
    best_score = 0
    
    for entity in entities:
        text_clean = re.sub(r'\W+', '', entity['text']).lower()
        score = SequenceMatcher(None, target_clean, text_clean).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = entity
    
    return best_match

def find_address_entities(entities, full_address):
    address_parts = [part.strip() for part in full_address.split(',')]
    matched_entities = []
    
    current_part = 0
    for entity in entities:
        cleaned_text = entity['text'].strip().rstrip(',')
        if current_part < len(address_parts) and cleaned_text == address_parts[current_part]:
            matched_entities.append(entity)
            current_part += 1
    
    return matched_entities if current_part == len(address_parts) else []

def convert_to_funsd(txt_content, json_data):
    entities = parse_txt(txt_content)
    funsd_entities = []
    
    # 匹配company
    company_entity = find_matching_entity(entities, json_data["company"]) if "company" in json_data.keys() else None
    
    # 匹配date
    date_entity = next((e for e in entities if json_data["date"] in e["text"]), None) if "date" in json_data.keys() else None
    
    # 匹配address
    address_entities = find_address_entities(entities, json_data["address"]) if "address" in json_data.keys() else None
    
    # 匹配total
    total_entity = next((e for e in entities if e["text"] == json_data["total"]), None) if "total" in json_data.keys() else None
    
    # 构建标签映射
    label_map = {}
    if company_entity:
        label_map[company_entity['line_num']] = "company"
    if date_entity:
        label_map[date_entity['line_num']] = "date"
    if address_entities:
        for e in address_entities:
            label_map[e['line_num']] = "address"
    if total_entity:
        label_map[total_entity['line_num']] = "total"
    
    # 构建FUNSD格式
    funsd_form = []
    entity_id = 0
    for idx, entity in enumerate(entities):
        label = label_map.get(idx, "other")
        xmin, ymin, xmax, ymax = 9999, 9999, -1, -1
        for bbox in entity['bbox']:
            xmin = min(xmin, bbox[0])
            ymin = min(ymin, bbox[1])
            xmax = max(xmax, bbox[0])
            ymax = max(ymax, bbox[1])
        funsd_entity = {
            "id": entity_id,
            "text": entity["text"],
            "box": [xmin, ymin, xmax, ymax],
            "label": label,
            "linking": [],
            "words": [{
                "text": entity["text"],
                "box": [xmin, ymin, xmax, ymax]
            }]
        }
        funsd_form.append(funsd_entity)
        entity_id += 1
    
    return {"form": funsd_form}


def locationJudgment(box1, box2):
    mid_box1 = ((box1[2] - box1[0]) / 2 + box1[0], (box1[3] - box1[1]) / 2 + box1[1])
    mid_box2 = ((box2[2] - box2[0]) / 2 + box2[0], (box2[3] - box2[1]) / 2 + box2[1])
    
    # box1在box2的左边
    if abs(mid_box2[1] - mid_box1[1]) < 5 and mid_box1[0] < mid_box2[0]:
        return "left"
    # box1在box2的右边
    elif abs(mid_box2[1] - mid_box1[1]) < 5 and mid_box1[0] > mid_box2[0]:
        return "right"
    # box1在box2的上方
    if abs(mid_box2[0] - mid_box1[0]) < 5 and mid_box1[1] < mid_box2[1]:
        return "up"
    # box1在box2的下方
    elif abs(mid_box2[0] - mid_box1[0]) < 5 and mid_box1[1] > mid_box2[1]:
        return "down"


def Convert2Funsd(json_data, txt_content):
    lines = txt_content.strip().split('\n')
    funsd_form = []
    cnt = 0
    for key, value in json_data.items():
        funsd_entity = {
            "id": cnt,
            "text": "",
            "box": [],
            "label": "",
            "linking": [],
            "words": [{
                "text": "",
                "box": []
            }]
        }
        cnt += 1

        cur_xmin, cur_ymin, cur_xmax, cur_ymax = 9999, 9999, -1, -1
        cur_text = ""
        merged_line_num = []
        for line_num, line in enumerate(lines):
            if line_num in merged_line_num:
                continue
            parts = line.strip().split(',')
            xmin, ymin, xmax, ymax = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[5])
            text_content = parts[-1]
            
            # text_similar_ratio = similar_diff_ratio(text_content, value)
            # if text_similar_ratio < 0.4:
            #     continue
            if [cur_xmin, cur_ymin, cur_xmax, cur_ymax] == [9999, 9999, -1, -1]:
                [cur_xmin, cur_ymin, cur_xmax, cur_ymax] = [xmin, ymin, xmax, ymax]
            else:
                location = locationJudgment([cur_xmin, cur_ymin, cur_xmax, cur_ymax], [xmin, ymin, xmax, ymax])
            if len(cur_text) == 0:
                cur_text = text_content
            elif location == 'left':
                cur_text = cur_text + f" {text_content}"
            elif location == 'right':
                cur_text == f"{text_content} " + cur_text
            else:
                continue
            
            cur_xmin = min(xmin, cur_xmin)
            cur_ymin = min(ymin, cur_ymin)
            cur_xmax = max(xmax, cur_xmax)
            cur_ymax = max(ymax, cur_ymax)
            
            merged_line_num.append(line_num)
        text_similar_ratio = similar_diff_ratio(cur_text, value)
        # if text_similar_ratio < 0.4:
        
        funsd_entity['text'] = cur_text
        funsd_entity['box'] = [cur_xmin, cur_ymin, cur_xmax, cur_ymax]
        funsd_entity['label'] = key
        funsd_entity['words'][0]['text'] = cur_text
        funsd_entity['words'][0]['box'] = [cur_xmin, cur_ymin, cur_xmax, cur_ymax]

        funsd_form.append(funsd_entity)
    return {"form": funsd_form}
        



def similar_diff_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

root_path = "/mnt/e/data/datasets/SROIE/SROIE2019数据集/"
output_path = "/mnt/d/geolayoutlm/V000_20250311/datasets/SROIE"
os.makedirs(output_path, exist_ok=True)

train_path = os.path.join(root_path, "0325updated.task2train(626p)")
train_txt_path = "/mnt/e/data/datasets/SROIE/openxlab/SROIE/annotations/training"

test_image_path = os.path.join(root_path, "SROIE_test_images_task_3")
test_json_path = os.path.join(root_path, "SROIE_test_gt_task_3")
test_txt_path = os.path.join(root_path, "text.task1&2-test（361p)")

for split in ['train', 'val']:
    os.makedirs(os.path.join(output_path, f"{'testing_data/images' if split == 'val' else f'{split}ing_data/images'}"), exist_ok=True)
    os.makedirs(os.path.join(output_path, f"{'testing_data/annotations' if split == 'val' else f'{split}ing_data/annotations'}"), exist_ok=True)
    if split == 'train':
        image_files = glob(os.path.join(train_path, "*.jpg"))
        for image_file in tqdm(image_files):
            file_name = os.path.basename(image_file).split('.')[0]
            
            json_file = image_file.replace('.jpg', '.json')
            json_data = json.load(open(json_file, 'r', encoding='utf-8'))

            txt_file = os.path.join(train_txt_path, f"{file_name}.txt")
            with open(txt_file, 'r', encoding='utf-8') as f:
                txt_content = f.read()
            
            # funsd_data = convert_to_funsd(txt_content, json_data)

            funsd_data = Convert2Funsd(json_data, txt_content)



            json_output_path = os.path.join(output_path, f"{'testing_data/annotations' if split == 'val' else f'{split}ing_data/annotations'}", f"{file_name}.json")
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(funsd_data, f, indent=4, ensure_ascii=False)
            image_output_path = os.path.join(output_path, f"{'testing_data/images' if split == 'val' else f'{split}ing_data/images'}", f"{file_name}.jpg")
            shutil.copy(image_file, image_output_path)
    elif split == "val":
        image_files = glob(os.path.join(test_image_path, "*.jpg"))
        for image_file in tqdm(image_files):
            file_name = os.path.basename(image_file).split('.')[0]

            json_file = os.path.join(test_json_path, f"{file_name}.json")
            json_data = json.load(open(json_file, 'r', encoding='utf-8'))

            txt_file = os.path.join(test_txt_path, f"{file_name}.txt")
            with open(txt_file, 'r', encoding='utf-8') as f:
                txt_content = f.read()
            
            funsd_data = convert_to_funsd(txt_content, json_data)

            json_output_path = os.path.join(output_path, f"{'testing_data/annotations' if split == 'val' else f'{split}ing_data/annotations'}", f"{file_name}.json")
            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(funsd_data, f, indent=4, ensure_ascii=False)
            image_output_path = os.path.join(output_path, f"{'testing_data/images' if split == 'val' else f'{split}ing_data/images'}", f"{file_name}.jpg")
            shutil.copy(image_file, image_output_path) 

# json_files = glob(os.path.join(train_path, "*.json"))
# labels = []
# for index, json_file in enumerate(json_files):
#     json_data = json.load(open(json_file, 'r', encoding='utf-8'))
#     for key, value in json_data.items():
#         if key not in labels:
#             labels.append(key)
# print(labels)
### ['company', 'date', 'address', 'total']