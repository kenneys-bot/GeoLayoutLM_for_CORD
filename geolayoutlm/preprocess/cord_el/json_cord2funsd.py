import os
import json
from glob import glob
from tqdm import tqdm

def add_linking(data):
    """
    添加关系链接，作用于linking

    :param data: 以funsd's json格式修改后的cord json内容
    """
    for i in range(len(data['form'])):
        i_row_id = data['form'][i]['row_id']
        for j in range(len(data['form'])):
            if i >= j:
                continue
            j_row_id = data['form'][j]['row_id']
            if i_row_id == j_row_id:
                data['form'][i]['linking'].append([data['form'][i]['id'], data['form'][j]['id']])
                data['form'][j]['linking'].append([data['form'][i]['id'], data['form'][j]['id']])

def merge_text(text_group):
    """
    合并字符串

    :param text_group: ["str", "str"...]
    :return: "str"
    """
    text_merge = ""
    for text_idx, text in enumerate(text_group):
        text_merge = text_merge + " " + text

    return text_merge

def merge_bigbox(box_group):
    """
    输入多个矩形框坐标，计算出组成的最大区域矩形框范围

    :param box_group: [[xmin, ymin, xmax, ymax]...[xmin, ymin, xmax, ymax]]
    :return: [xmin, ymin, xmax, ymax]
    """
    xmin = 9999
    ymin = 9999
    xmax = -1
    ymax = -1
    for box_idx, box in enumerate(box_group):
        xmin = min(box[0], xmin)
        ymin = min(box[1], ymin)
        xmax = max(box[2], xmax)
        ymax = max(box[3], ymax)

    return [xmin, ymin, xmax, ymax]

def json_cord2funsd(in_json_obj):
    """
    将CORD datasets的json格式转换成FUNSD datasets的json格式

    :param in_json_obj: json data
    """
    cnt = 0
    data = {"form": []}
    img_width = in_json_obj['meta']['image_size']['width']
    img_height = in_json_obj['meta']['image_size']['height']
    in_json_obj = in_json_obj['valid_line']
    for form_inx, form in enumerate(in_json_obj):
        category = form['category']
        group_id = form['group_id']
        temp_json = {}
        temp_is_key = []

        if len(form['words']) == 1:
            itm = form['words'][0]

            temp_is_key.append(itm['is_key'])
            row_id = itm['row_id']
            text = itm['text']
            bbox = [itm['quad']['x1'], itm['quad']['y1'], itm['quad']['x3'], itm['quad']['y3']]

            temp_json = {}
            temp_json.update({"box": bbox})
            temp_json.update({"text": text})
            temp_json.update({"label": category})

            words = []
            temp_json_words = {}
            temp_json_words.update({"box": bbox})
            temp_json_words.update({"text": text})
            words.append(temp_json_words)

            temp_json.update({"words": words})
            temp_json.update({"linking": []})
            temp_json.update({"is_key": temp_is_key})
            temp_json.update({"row_id": row_id})
            temp_json.update({"group_id": group_id})
            temp_json.update({"id": cnt})
            cnt += 1

            data['form'].append(temp_json)

        elif len(form['words']) > 1:
            temp_text = []
            temp_is_key = []
            box_group = []
            for itm_inx, itm in enumerate(form['words']):
                temp_is_key.append(itm['is_key'])
                row_id = itm['row_id']

                temp_text.append(itm['text'])
                box = [itm['quad']['x1'], itm['quad']['y1'], itm['quad']['x3'], itm['quad']['y3']]
                box_group.append(box)

            text = merge_text(temp_text)
            bbox = merge_bigbox(box_group)

            temp_json = {}
            temp_json.update({"box": bbox})
            temp_json.update({"text": text})
            temp_json.update({"label": category})
            temp_json.update({"linking": []})

            words = {'words': []}
            for j in range(len(box_group)):
                temp_json_words = {}
                temp_json_words.update({"box": box_group[j]})
                temp_json_words.update({"text": temp_text[j]})
                words['words'].append(temp_json_words)
            temp_json.update(words)

            temp_json.update({"is_key": temp_is_key})
            temp_json.update({"row_id": row_id})
            temp_json.update({"group_id": group_id})
            temp_json.update({"id": cnt})
            cnt += 1

            data['form'].append(temp_json)

    add_linking(data)

    return data