# SemEval2010-to-TACRED
A simple tool for converting data format from SemEval2010 to TACRED.

## Environment Requirements
* python 3.6
* tqdm
* [StanfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/) \[[download](https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip)\]

## Data
* [SemEval2010 Task8](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50) \[[paper](https://www.aclweb.org/anthology/S10-1006.pdf)\]

## Usage
1. [Download CoreNlp](https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip), put it in `./tools/` folder and then unzip it.
2. Use the following command to convert data:
```
python convert.py
```
3. The result file will be store in `./result/` folder.

## Reference Link:
* https://github.com/ChristophAlt/StAn
* https://github.com/waxin/ToTACRED
