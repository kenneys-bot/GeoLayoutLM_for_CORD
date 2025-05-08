from glob import glob
import os

def delete_not_match_img2json(path):
    os.system(f"mv {os.path.join(path, 'train')} {os.path.join(path, 'training_data')}")
    os.system(f"mv {os.path.join(path, 'test')} {os.path.join(path, 'testing_data')}")
    os.system(f"mv {os.path.join(path, 'training_data', 'image')} {os.path.join(path, 'training_data', 'images')}")
    os.system(f"mv {os.path.join(path, 'training_data', 'json')} {os.path.join(path, 'training_data', 'annotations')}")
    os.system(f"mv {os.path.join(path, 'testing_data', 'image')} {os.path.join(path, 'testing_data', 'images')}")
    os.system(f"mv {os.path.join(path, 'testing_data', 'json')} {os.path.join(path, 'testing_data', 'annotations')}")

    # img_cnt = 0
    # json_cnt = 0
    # for dataset_model in ['training_data']:
    #     images_files = glob(os.path.join(path, dataset_model, 'images', '*.png'))
    #     json_files = glob(os.path.join(path, dataset_model, 'annotations', '*.json'))
    #     # for image_file in images_files:
    #     #     file_name = os.path.basename(image_file)
    #     #     file = os.path.splitext(image_file.replace("/images/", '/annotations/'))[0] + ".json"
    #     #     if not os.path.exists(file):
    #     #         #os.remove(image_file)
    #     #         print(image_file)
    #     #         img_cnt += 1
    #     #         continue
    #     #     file = os.path.join(path, "preprocessed", file_name)
    #     #     if not os.path.exists(file):
    #     #         #os.remove(image_file)
    #     #         print(image_file)
    #     #         img_cnt += 1
    #     #         continue
    #     for json_file in json_files:
    #         file_name = os.path.basename(json_file)
    #         file = os.path.splitext(json_file.replace("/annotations/", "/images/"))[0] + ".png"
    #         if not os.path.exists(file):
    #             os.remove(json_file)
    #             # print(json_file)
    #             json_cnt += 1
    #             continue
    #         file = os.path.join(path, "preprocessed", file_name)
    #         if not os.path.exists(file):
    #             os.remove(json_file)
    #             # print(json_file)
    #             json_cnt += 1
    #             continue
    # print("img_cnt = {}".format(img_cnt))
    # print("json_cnt = {}".format(json_cnt))

if __name__ == "__main__":
    delete_not_match_img2json("/public/home/lab70432/LWW_workspace/env_sources/geolayoutlm/CORD/dataset/cord_geo")