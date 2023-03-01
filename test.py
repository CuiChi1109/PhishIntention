from phishintention.phishintention_main import test
import matplotlib.pyplot as plt
from phishintention.phishintention_config import load_config
from phishintention.phishintention_main import element_recognition, phishpedia_classifier_OCR, credential_classifier_mixed_al, driver_loader, dynamic_analysis

# use full model
url = open("phishintention/datasets/test_sites/accounts.g.cdcde.com/info.txt").read().strip()
screenshot_path = "phishintention/datasets/test_sites/accounts.g.cdcde.com/shot.png"
device = 'cuda' # or device = 'cpu'
cfg_path = None # None means use default config.yaml
AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(cfg_path, device)

phish_category, pred_target, plotvis, siamese_conf, dynamic, _, pred_boxes, pred_classes = test(url, screenshot_path,
                                                                      AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)

print('Phishing (1) or Benign (0) ?', phish_category)
print('What is its targeted brand if it is a phishing ?', pred_target)
print('What is the siamese matching confidence ?', siamese_conf)
print('Where are the predicted bounding boxes (in [x_min, y_min, x_max, y_max])?', pred_boxes)
plt.imshow(plotvis[:, :, ::-1])
plt.title("Predicted screenshot with annotations")
plt.show()