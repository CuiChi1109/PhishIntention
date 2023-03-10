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

annot = [x.strip().split(',') for x in open('/home/chi/PhishIntention/logo_reg_dataset/phish1000_coord.txt').readlines()]
annot_dict = {}
for c in annot:
    assert len(c) == 5
    x1, y1, x2, y2 = map(float, c[1:])
    annot_dict[c[0]] = np.asarray([[x1, y1, x2, y2]])
# print(annot_dict)

pedia_model, ocr_model, logo_feat_list, file_name_list = phishpedia_config_OCR(num_classes=277,
                                                weights_path='./src/OCR_siamese_utils/output/targetlist_lr0.01/bit.pth.tar',
                                                ocr_weights_path='./src/OCR_siamese_utils/demo_downgrade.pth.tar',
                                                targetlist_path='./src/phishpedia_siamese/expand_targetlist')

domain_map_path = '/home/chi/PhishIntention/phishintention/src/phishpedia_siamese/domain_map.pkl'
ct_benign = 0
ct_phish = 0

wrong_brand, wrong_img_name, wrong_type = [], [], []

idx = np.random.randint(1000, size = 100)
i = 0
ct_wrong_reg = 0

for path in tqdm(os.listdir('/home/chi/PhishIntention/logo_reg_dataset/Sample_phish1000')):
# for path in tqdm(os.listdir('/home/chi/PhishIntention/logo_reg_dataset/Sample_benign1000')):
    i += 1
    url = '' # dummy value, not important
    img_path = '/home/chi/PhishIntention/logo_reg_dataset/Sample_phish1000/' + path + '/shot.png'

    # img_path = '/home/chi/PhishIntention/logo_reg_dataset/Sample_benign1000/' + path + '/shot.png'
    # annot = [x.strip().split(',') for x in open('/home/chi/PhishIntention/logo_reg_dataset/benign1000_coord.txt').readlines()]


    # read labelled
    # for c in annot:
    #     assert len(c) != 5
    #     if c[0] == path:
    #         x1, y1, x2, y2 = map(float, c[1:])
    #         break
    # pred_boxes = np.asarray([[x1, y1, x2, y2]])
    pred_boxes = annot_dict[path]
    pred_classes = np.asarray([0.])

    # get predicted targeted brand
    pred_target, _, _ = phishpedia_classifier_OCR(pred_classes=pred_classes, pred_boxes=pred_boxes,
                                                domain_map_path=domain_map_path,
                                                model=pedia_model,
                                                ocr_model=ocr_model,
                                                logo_feat_list=logo_feat_list,
                                                file_name_list=file_name_list,
                                                url=url,
                                                shot_path=img_path,
                                                ts=0.83)

    if pred_target is not None:
        # ct_benign += 1  # if test on benign, look at this
        if brand_converter(pred_target) == brand_converter(path.split('+')[0]):
            ct_phish += 1 # if test on phish, look at this
        else:
            ct_wrong_reg += 1
            print(pred_target, path.split('+')[0])

    else:
        img = cv2.imread(img_path)
        cv2.rectangle(img, (int(pred_boxes[0][0]), int(pred_boxes[0][1])), (int(pred_boxes[0][2]), int(pred_boxes[0][3])), (0, 0, 255), 2)
        cv2.imwrite("/home/chi/PhishIntention/phishintention/src/phishpedia_siamese/siamese_retrain/results_failed_phish/"+path+".png", img)

print(ct_phish, ct_wrong_reg)
# print(ct_benign, '/', len(os.listdir('/home/chi/PhishIntention/logo_reg_dataset/Sample_benign1000')))