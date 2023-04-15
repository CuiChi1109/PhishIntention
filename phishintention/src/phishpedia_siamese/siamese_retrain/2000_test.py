import os
# os.chdir('../../../')
import sys
sys.path.append("/home/chi/PhishIntention/phishintention")
# print(sys.path)
from src.OCR_aided_siamese import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# here are 5 situation
# detect no logo                                ===> detect failed or there is no logo in the screenshot
# detect and predict are both NA                ===> recognize failed
# detect and predict are both brands and same   ===> succeess
# detect is NA but predict not                  ===> detect failed
# detect is brand but predict is NA             ===> something wrong when I prepare the dataset
ct_no_logo = 0
ct_pn_gn = 0
ct_pb_gb = 0
ct_pn_gb = 0
ct_pb_gn = 0

data_path = '/home/chi/PhishIntention/new_test_data/'
dataset_path = data_path + 'phishing_test_dataset/'

results_path = data_path + 'final_results/'
# failed_data_path = data_path + "failed_data_with_detect/"
# succeed_data_path = data_path + "succeed_data/"
no_logo_path = results_path + 'no_logo/'
pn_gn_path = results_path + 'pn_gn/'
pb_gb_path = results_path + 'pb_gb/'
pn_gb_path = results_path + 'pn_gb/'
pb_gn_path = results_path + 'pb_gn/'
pn_gb_gt_path = results_path + 'pb_gn_gt/'
check_path = '/home/chi/PhishIntention/new_test_data/all_logo_results/pn_gb/strange'

pred_coor_path = data_path + "awl_log_coord.txt"
gt_coor_path = data_path + 'coor.txt'
csv_path = data_path + "results_with_detection.csv"

# os.system("rm "+failed_data_path+"*")
# os.system("rm "+succeed_data_path+"*")

pred_annot = [x.strip().split(',') for x in open(pred_coor_path).readlines()]
gt_annot = [x.strip().split(',') for x in open(gt_coor_path).readlines()]
pred_annot_dict = {}
gt_annot_dict = {}

# inital pred_annot_dict and gt_annot_dict
# it is possible that there are several detected logo rectangles in one screenshot
for c in pred_annot:
    assert len(c) == 5
    x1, y1, x2, y2 = map(float, c[1:])
    if c[0] not in list(pred_annot_dict.keys()):
        pred_annot_dict[c[0]] = np.asarray([[x1, y1, x2, y2]])
    else:
        pred_annot_dict[c[0]] = np.append(pred_annot_dict[c[0]], [[x1, y1, x2, y2]], axis=0)

for c in gt_annot:
    assert len(c) == 5
    x1, y1, x2, y2 = map(float, c[1:])
    gt_annot_dict[c[0]] = np.asarray([[x1, y1, x2, y2]])


# pedia_model, ocr_model, logo_feat_list, file_name_list = phishpedia_config_OCR(num_classes=277,
#                                                 weights_path='./src/OCR_siamese_utils/output/targetlist_lr0.01/bit.pth.tar',
#                                                 ocr_weights_path='./src/OCR_siamese_utils/demo_downgrade.pth.tar',
#                                                 targetlist_path='./src/phishpedia_siamese/expand_targetlist')

# domain_map_path = '/home/chi/PhishIntention/phishintention/src/phishpedia_siamese/domain_map.pkl'

pedia_model, ocr_model, logo_feat_list, file_name_list = phishpedia_config_OCR(num_classes=277,
                                                weights_path='/home/chi/PhishIntention/phishintention/src/OCR_siamese_utils/output/targetlist_lr0.01/bit.pth.tar',
                                                ocr_weights_path='/home/chi/PhishIntention/phishintention/src/OCR_siamese_utils/demo_downgrade.pth.tar',
                                                targetlist_path='/home/chi/PhishIntention/phishintention/src/phishpedia_siamese/expand_targetlist')

domain_map_path = '/home/chi/PhishIntention/phishintention/src/phishpedia_siamese/domain_map.pkl'

names,results, types = [], [], []

idx = np.random.randint(1000, size = 100)
i = 0
ct_wrong_reg = 0


def load_results(pred_boxes_list, img_path, result_path):
    if pred_boxes_list is not None:
        img = cv2.imread(img_path)
        for pred_boxes in pred_boxes_list:
            cv2.rectangle(img, (int(pred_boxes[0]), int(pred_boxes[1])), (int(pred_boxes[2]), int(pred_boxes[3])), (0, 0, 255), 2)
        cv2.imwrite(result_path, img)
names,results, types = [], [], []
# here are 5 situation
# detect no logo                                ===> detect failed or there is no logo in the screenshot
# detect and predict are both NA                ===> recognize failed
# detect and predict are both brands and same   ===> succeess
# detect is NA but predict not                  ===> detect failed
# detect is brand but predict is NA             ===> something wrong when I prepare the dataset
ct_no_logo = 0
ct_pn_gn = 0
ct_pb_gb = 0
ct_pn_gb = 0
ct_pb_gn = 0
# os.system("rm "+no_logo_path+"*")
# os.system("rm "+pb_gb_path+"*")
# os.system("rm "+pb_gn_path+"*")
# os.system("rm "+pn_gb_path+"*")
# os.system("rm "+pn_gn_path+"*")
# os.system("rm "+pn_gb_gt_path+"*")
# For every screenshot of webpage
for path in tqdm(os.listdir(dataset_path)):

    # only detect images in strange
    if path not in os.listdir(check_path):
        continue

# for path in tqdm(os.listdir('/home/chi/PhishIntention/logo_reg_dataset/Sample_benign1000')):
    # print("image ", path)
    i += 1
    url = '' # dummy value, not important
    img_path = dataset_path + path
    # img_path = dataset_path + path + '/shot.png'
    print("img:", img_path)

    # img_path = '/home/chi/PhishIntention/logo_reg_dataset/Sample_benign1000/' + path + '/shot.png'
    # annot = [x.strip().split(',') for x in open('/home/chi/PhishIntention/logo_reg_dataset/benign1000_coord.txt').readlines()]

    try:
        pred_boxes = pred_annot_dict[path[:-4]]
    except:
        ct_no_logo += 1
        load_results(None,path, no_logo_path+path)
        # load_results(None,path, no_logo_path+path+".png")
        # print("pred no logo ", path)
        results.append("pred no logo")
        types.append("no logo")
        names.append(path)
        continue


    gt_boxes = gt_annot_dict[path[:-4]]

    # get pred target list

    pred_classes = np.zeros(pred_boxes.shape[0])
    logo_boxes = pred_boxes[pred_classes==0]
    pred_target, _, _ = phishpedia_classifier_OCR(pred_classes=pred_classes, pred_boxes=pred_boxes,
                                                domain_map_path=domain_map_path,
                                                model=pedia_model,
                                                ocr_model=ocr_model,
                                                logo_feat_list=logo_feat_list,
                                                file_name_list=file_name_list,
                                                url=url,
                                                shot_path=img_path,
                                                ts=0.83)

    # get gt target
    gt_target, _, _ = phishpedia_classifier_OCR(pred_classes=np.asarray([0.]), pred_boxes=gt_boxes,
                                                    domain_map_path=domain_map_path,
                                                    model=pedia_model,
                                                    ocr_model=ocr_model,
                                                    logo_feat_list=logo_feat_list,
                                                    file_name_list=file_name_list,
                                                    url=url,
                                                    shot_path=img_path,
                                                    ts=0.83)

    # pb_gb
    if pred_target is not None and gt_target is not None:
        if gt_target == pred_target:
            ct_pb_gb += 1
            # print(pred_target, path.split('+')[0])
            load_results(pred_boxes, path, pb_gb_path+path)
            results.append(gt_target)
            types.append("pb_gb")
        else:
            print("I'm crazy")
            exit()

    elif pred_target is None and gt_target is None:
        ct_pn_gn += 1
        load_results(pred_boxes,path, pn_gn_path+path)
        results.append('NA')
        types.append("pn_gn")

    elif pred_target is not None and gt_target is None:
        ct_pb_gn += 1
        load_results(pred_boxes,path, pb_gn_path+path)
        results.append(pred_target)
        types.append("pb_gn")

    elif pred_target is None and gt_target is not None:
        ct_pn_gb += 1
        load_results(pred_boxes,path, pn_gb_path+path)
        load_results(gt_boxes,path, pn_gb_gt_path+path)
        results.append(gt_target)
        types.append("pn_gb")

    names.append(path)
    assert len(names) == len(types)



# df = pd.DataFrame({"names":names, "results":results, "type":types})
# df.to_csv(csv_path)

# print("total :", len(os.listdir(dataset_path)))
# print("pb_gb :", ct_pb_gb)
# print("pn_gn :", ct_pn_gn)
# print("pn_gb :", ct_pn_gb)
# print("pb_gn :", ct_pb_gn)
# print("no_logo :", ct_no_logo)
# print(ct_benign, '/', len(os.listdir('/home/chi/PhishIntention/logo_reg_dataset/Sample_benign1000')))

# i = 0
# for path in tqdm(os.listdir(pn_gb_path)):
#     img_path = pn_gb_path + path
#     i+=1
#     if i > 5:
#         break
#     pred_boxes = pred_annot_dict[path[:-4]]
#     pred_classes = np.zeros(pred_boxes.shape[0])
#     pred_target, _, _ = phishpedia_classifier_OCR(pred_classes=pred_classes, pred_boxes=pred_boxes,
#                                                 domain_map_path=domain_map_path,
#                                                 model=pedia_model,
#                                                 ocr_model=ocr_model,
#                                                 logo_feat_list=logo_feat_list,
#                                                 file_name_list=file_name_list,
#                                                 url='',
#                                                 shot_path=img_path,
#                                                 ts=0.83)
#     print("pred_target", pred_target)