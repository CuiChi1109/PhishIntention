'''
This file could detect AWL in an image and then store the results in "awl_log_coord.txt"
Also, it could draw the AWL on the image
'''
from phishintention.src.AWL_detector import *
from phishintention.phishintention_config import load_config
import os
import yaml
from tqdm import tqdm
import numpy as np
import torch
import json

# screenshot_dir = "/home/chi/PhishIntention/awl_data/val_imgs/"
# results_dir = "/home/chi/PhishIntention/awl_data/results/"
# results_dir = "/home/chi/PhishIntention/new_test_data/detect_results/"
# screenshot_dir = "/home/chi/PhishIntention/new_test_data/all_logo_in_dataset/"
screenshot_dir = "/home/chi/PhishIntention/new_test_data/test_recog_data/pn_gb/"

device = 'cpu'
with open("/home/chi/PhishIntention/phishintention/configs.yaml") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
AWL_CFG_PATH = os.path.join(os.path.dirname(__file__), configs['AWL_MODEL']['CFG_PATH'].replace('/', os.sep))
AWL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), configs['AWL_MODEL']['WEIGHTS_PATH'].replace('/', os.sep))
AWL_CONFIG, AWL_MODEL = element_config(rcnn_weights_path=AWL_WEIGHTS_PATH, rcnn_cfg_path=AWL_CFG_PATH, device=device)
category_dict = {0:"logo",
                 1:"input",
                 2:"button",
                 3:"label",
                 4:"block"}

# build img dict
# with open('/home/chi/PhishIntention/awl_data/val_coco.json', 'r') as f:
#      data = json.load(f)
# img_list = data['images']
# img_dict = {}
# for img in img_list:
#      img_dict[img['file_name']] = img['id']

pred = []
def test_awl(img_name):
    screenshot_path = screenshot_dir + img_name
    print(screenshot_path)
    pred_classes, pred_boxes, pred_scores = element_recognition(img=screenshot_path, model=AWL_MODEL)
    print(pred_classes)
    check = vis(screenshot_path, pred_boxes, pred_classes)
    # print(img_name, pred_boxes)

    # for i in range(len(pred_classes)):
    #     dic = {}
    #     dic['image_id'] = img_dict[img_name]
    #     dic['category_id'] = int(pred_classes[i]) + 1
    #     coor = pred_boxes[i].to(torch.int64).numpy().tolist()
    #     dic['bbox'] = [coor[0], coor[1], coor[2]-coor[0], coor[3]-coor[1]]
    #     dic['score'] = float(pred_scores[i])
    #     pred.append(dic)
        # result = img_name[:-4] + '   ' + str(tuple((pred_boxes[i].to(torch.int64).numpy().tolist()))) + '  ' + category_dict[int(pred_classes[i])] + '\n'

    # cv2.imwrite(results_dir + img_name, check)
    for i in range(len(pred_classes)):
        # if the pred_classes is logo
        if pred_classes[i] == 0:
            coord = [str(int(c)) for c in pred_boxes[i].tolist()]
            result = img_name[:-4]+","+','.join(coord)+"\n"
            print(result)
            with open('/home/chi/PhishIntention/new_test_data/test_recog_data/pn_gb/detect_coor.txt', 'a+') as f:
                f.write(result)
                f.close()


for img in tqdm(os.listdir(screenshot_dir)):
    if img[-3:] == 'png':
        test_awl(img)

# with open("/home/chi/PhishIntention/awl_data/awl_result.json", "w") as f:
#     json.dump(pred, f)