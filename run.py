
from phishintention.phishintention_main import *
import time
import datetime
import sys
from datetime import datetime, timedelta, time
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':

    AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(None)

    while True: # comment if you want to process it once
        # date = '2021-12-22'
        # date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        date = datetime.today().strftime('%Y-%m-%d')
        print('Today is:', date)
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', "--folder",
                            default='E:\\screenshots_rf\\{}'.format(date),
                            help='Input folder path to parse')
        parser.add_argument('-r', "--results", default=date + '.txt',
                            help='Input results file name')
        parser.add_argument('--repeat', action='store_true')
        parser.add_argument('--no_repeat', action='store_true')
        args = parser.parse_args()
        print(args)
        runit(args.folder, args.results, AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL,
              SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH) # if running phishintention
        print('Process finish')

        if args.no_repeat:
            break
