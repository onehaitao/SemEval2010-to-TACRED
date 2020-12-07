#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import json
import re
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'./resource/stanford-corenlp-full-2018-10-05')
props = dict(
    annotators="tokenize,pos,ner,depparse",
    pipelineLanguage="en",
    outputFormat="json",
)


def relation_process(raw_relation):
    relation = raw_relation.strip()
    subj_type = 'Other'
    obj_type = 'Other'
    pattern = r'(.*?)-(.*?)\((e[12]),(e[12])\)'
    if relation != 'Other':
        value = re.findall(pattern, relation)[0]
        if value[2] == 'e1':
            subj_type == value[0]
            obj_type = value[1]
        else:
            subj_type = value[1]
            obj_type = value[0]
    res = dict(
        relation=relation,
        subj_type=subj_type,
        obj_type=obj_type
    )
    return res


def sentence_process(raw_sentence):
    # remove quotes
    sentence = raw_sentence[1:-1]
    # remove postion indicators
    head_entity = re.findall(r'<e1>(.*?)</e1>', sentence)[0]
    tail_entity = re.findall(r'<e2>(.*?)</e2>', sentence)[0]
    sentence = re.sub(r'</?e[12]>', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)

    res = json.loads(nlp.annotate(sentence, properties=props))
    sents = res['sentences']
    token = [x['word'] for sent in sents for x in sent['tokens']]
    pos = [x['pos'] for sent in sents for x in sent['tokens']]
    ner = [x['ner'] for sent in sents for x in sent['tokens']]
    dep = []
    for sent in sents:
        dep_res = [(x['dep'], x['governor'], x['dependent'])
                   for x in sent['basicDependencies']]
        dep += sorted(dep_res, key=lambda x: x[2], reverse=False)
    deprel = [x[0] for x in dep]
    head = [x[1] for x in dep]
    lengths = map(len, [token, pos, ner, deprel, head])
    assert len(set(lengths)) == 1
    subj_start, subj_end = find_entity_pos(token, head_entity)
    obj_start, obj_end = find_entity_pos(token, tail_entity)

    ret = dict(
        token=token,
        subj_start=subj_start,
        subj_end=subj_end,
        obj_start=obj_start,
        obj_end=obj_end,
        stanford_pos=pos,
        stanford_ner=ner,
        stanford_deprel=deprel,
        stanford_head=head
    )
    return ret


def find_entity_pos(token, entity):
    entity_list = nlp.word_tokenize(entity)
    size = len(entity_list)
    start = end = None
    for i in range(0, len(token)-size+1):
        window = token[i:i+size]
        if window == entity_list:
            start = i
            end = i + size - 1
    if start is None or end is None:
        raise ValueError('Entity {} is not found!'.format(entity))
        import pdb
        pdb.set_trace()
    return start, end


def convert(src_file, des_file):
    with open(src_file, 'r', encoding='utf-8') as fr:
        file_data = fr.readlines()

    data = []
    for i in tqdm(range(0, len(file_data), 4)):
        meta = {}
        s = file_data[i].strip().split('\t')
        assert len(s) == 2
        meta['id'] = s[0]
        meta['docid'] = s[0]
        meta['comment'] = file_data[i+2].strip()
        sen_res = sentence_process(s[1])
        rel_res = relation_process(file_data[i+1])
        data.append({**meta, **sen_res, **rel_res})

    with open(des_file, 'w', encoding='utf-8') as fw:
        json.dump(data, fw, ensure_ascii=False)


if __name__ == '__main__':
    train_src = './SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    test_src = './SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    train_des = './result/train.json'
    test_des = './result/test.json'
    convert(train_src, train_des)
    convert(test_src, test_des)
    nlp.close()
