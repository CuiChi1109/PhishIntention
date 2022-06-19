# PhishIntention

## PhishIntention
- This is the official implementation of "Inferring Phishing Intention via Webpage Appearance and Dynamics: A Deep Vision Based Approach"USENIX'22 [link to paper](), [link to our website](https://sites.google.com/view/phishintention/home)
- The contributions of our paper:
   - [x] We propose a referenced-based phishing detection system that captures both brand intention and credential-taking intention. To the best of our knowledge, is the first work which analyzes both brand intention and credential-taking intentions in a systematic way for phishing detection.
   - [x] We address various technical challenges in detecting the intentions by orchestrating multiple deep learning models. By design, our system is robust against misleading legitimacies and HTML obfuscation attack.
   - [x] We conduct extensive experiments to evaluate our system. The experiments evaluate the overall and step-wise effectiveness, robustness against various adversarial attacks, and usefulness in practice.
   - [x] We implement our system with a phishing monitoring system. It reports phishing webpages per day with the highest precision in comparison to the state-of-the-art phishing detection solutions.
    
## Framework
    
<img src="big_pic/Screenshot 2021-08-13 at 9.15.56 PM.png" style="width:2000px;height:350px"/>

```Input```: a screenshot, ```Output```: Phish/Benign, Phishing target
- Step 1: Enter <b>Abstract Layout detector</b>, get predicted elements

- Step 2: Enter <b>Siamese Logo Comparison</b>
    - If Siamese report no target, ```Return  Benign, None```
    - Else Siamese report a target, Enter step 3 <b>CRP classifier</b>
       
- Step 3: <b>CRP classifier</b>
   - If <b>CRP classifier</b> reports its a CRP page, go to step 5 <b>Return</b>
   - ElIf not a CRP page and havent execute <b>CRP Locator</b> before, go to step 4: <b>CRP Locator</b>
   - Else not a CRP page but have done <b>CRP Locator</b> before, ```Return Benign, None``` 

- Step 4: <b>CRP Locator</b>
   - Find login/signup links and click, if reach a CRP page at the end, go back to step 1 <b>Abstract Layout detector</b> with updated URL and screenshot
   - Else cannot reach a CRP page, ```Return Benign, None``` 
   
- Step 5: 
    - If reach a CRP + Siamese report target: ```Return Phish, Phishing target``` 
    - Else ```Return Benign, None``` 
    
## Project structure
```
src
    |___ AWL_detector_utils/: scripts for abstract layout detector 
        |__ output/
            |__ website_lr0.001/
                |__ model_final.pth
    |___ crp_classifier_utils/: scripts for CRP classifier
            |__ output/
                |__ Increase_resolution_lr0.005/
                    |__ BiT-M-R50x1V2_0.005.pth.tar
    |___ crp_locator_utils/: scripts for CRP locator 
        |__ login_finder/
            |__ output/
                |__ lr0.001_finetune/
                    |__ model_final.pth
    |___ OCR_siamese_utils/: scripts for OCR-aided Siamese
        |__ demo_downgrade.pth.tar
        |__ output/
            |__ targetlist_lr0.01/
                |__ bit.pth.tar
    |___ util/: other scripts (chromedriver utilities)
    
    |___ phishpedia_logo_detector/: training script for logo detector (for Phishpedia not PhishIntention)
    |___ phishpedia_siamese/: inference script for siamese (for Phishpedia not PhishIntention)
        |__ domain_map.pkl
        |__ expand_targetlist/
        
    |___ adv_attack/: adversarial attacking scripts
    |___ layout_matcher/: deprecated scripts
    
    |___ AWL_detector.py: inference script for AWL detector
    |___ crp_classifier.py: inference script for CRP classifier
    |___ OCR_aided_siamese.py: inference script for OCR-aided siamese
    |___ crp_locator.py: inference script for CRP-Transition locator
    |___ pipeline_eval.py: evaluation script 

phishintention_config.py: phish-discovery experiment config file for PhishIntention
phishintention_main.py: phish-discovery experiment evaluation script for PhishIntention
```

<!-- ## Requirements
Tested with Windows/Linux

python=3.7 

torch>=1.5.1 

torchvision>=0.6.0

Install Detectron2 manually, see the [official installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). Windows please follow this [guide](https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c) instead.

Then Run
```
pip install -r requirements.txt
```

<!--
## Instructions
### 1. Unzip targetlist
```bash
cd src/phishpedia_siamese/
unzip expand_targetlist.zip -d expand_targetlist
```
- If unzip has some problem, try downloading the folder manually [here](https://drive.google.com/file/d/1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I/view?usp=sharing).

### 2. Download all the model files, if you are Windows user you can skip, Linux user please do this step:
- Download all model files [here](https://drive.google.com/drive/folders/1XGiLfIeSHwoeoXEpMXhMR4M2tkj3pErJ?usp=sharing) and put them in the locations shown as project directory tree.
- Make sure your directory tree looks like above


### 3. Download all data files in the paper(Skip if you want to test your own data)
- Download [Phish 30k](https://drive.google.com/file/d/12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g/view?usp=sharing), out of which 4093 are non-credential-requiring phishing, see this [list](https://drive.google.com/file/d/1UVoK-Af3j4ixYy2_jEzG9ZBbYpRkuKFK/view?usp=sharing), shall filter them out when running experiment
- Download [Benign 25k](https://drive.google.com/file/d/1ymkGrDT8LpTmohOOOnA2yjhEny1XYenj/view?usp=sharing) dataset,
unzip and move them to **datasets/**
- Download [Mislead 3k](https://drive.google.com/file/d/1xmB_P6I9BwnNYGJb7yeN-o2A1fMlX3Oh/view?usp=sharing), unzip and move them to **datasets/**

### 4. Run experiment on dataset in paper (Skip if you want to test on your own dataset)
- For general experiment on Phish25K nonCRP, Benign25K, Mislead3K:
please run evaluation scripts
```bash
python -m src.pipeline_eval --data-dir [data folder] \
                            --mode [phish|benign] \ # set to phish if you are testing on Phish25K, set to benign if you ware testing on Benign25K or Mislead3K
                            --write-txt output.txt \
                            --exp intention \ # evaluate Phishpedia or PhishIntention
                            --ts 0.83
```

### 5. Run experiment on customized dataset
- The data folder should be organized in [this format](https://github.com/lindsey98/PhishIntention/tree/main/datasets/test_sites/www.paypal.com) (i.e. there should be an info.txt storing the url, html.txt storing the HTML code, and shot.png storing screenshot):

```bash
python phishintention_main.py --folder [data_folder_you_want_to_test] --results [name_you_want_to_give.txt]
```
-->

<!-- ## Telegram service to label found phishing (Optional)
### Introduction
- When phishing are reported by the model, users may also want to manually verify the intention of the websites, thus we also developed a telegram-bot to help labeling the screenshot. An example is like this <img src="big_pic/tele.png"/>
- In this application, we support the following command:
```
/start # this will return all the unlabelled data
/get all/date # this will return the statistics for all the data namely how many positive and negatives there are
/classify disagree # this will bring up phishing pages with any disagreement, ie one voted not phishing and one voted phishing for a revote
```
### Setup tele-bot
- 1. Create an empty google sheet for saving the results (foldername, voting results etc.)
- 2. Follow the [guide](https://www.analyticsvidhya.com/blog/2020/07/read-and-update-google-spreadsheets-with-python/) to download JSON file which stores the credential for that particular google sheet, save as **tele/cred.json**
- 3. Go to **tele/tele.py**, Change 
```
token = '[token for telebot]' 
folder = "[the folder you want to label]"
```
[How do I find token for telebot?](https://core.telegram.org/bots#botfather)
- 4. Go to **tele/**, Run **tele.py**
 -->
 
## Miscellaneous
- In our paper, we also implement several phishing detection and identification baselines, see [here](https://github.com/lindsey98/PhishingBaseline)
- We obtain feed from [Certstream phish_catcher](https://github.com/x0rz/phishing_catcher), we lower the score threshold to be 40 to process more suspicious websites, readers can refer to their repo for details
- We use Scrapy for website crawling [Repo here](https://github.com/lindsey98/MyScrapy.git) 

 -->
