import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def cleanText(text):
    text = text.replace('xxxx','')
    text = text.replace('<end>','')
    text = text.strip().split(' ')

    if text[0] == 'and':
        text.remove(text[0])
    return ' '.join(text)

def checkPassive(text):
    if text[1] == 'JJ' and text[0][-2:] == 'ed':
        return True
    return False

def solveEndingVerb(subj,obj):
    sent = obj[-1]
    pos_tag = preprocess(sent)
    for i in range(len(pos_tag) - 1, -1 , - 1):
        # print(i, pos_tag[i])
        if pos_tag[i][1] == 'EX' or pos_tag[i][1] == 'DT' or ( pos_tag[i][1] == 'JJ' and pos_tag[i-1][1] != 'NN' and pos_tag[i-1][1] != 'NNS'):
            mark = i
            obj[-1] = ' '.join(sent.split(' ')[:mark])
            subj.append(' '.join(sent.split(' ')[mark:]))
            break

    return subj,obj

def checkSubjInLast(text):
    pos_tag = preprocess(text)
    return False, False

def isNounPhrase(sent):
    tags = preprocess(sent)
    if tags == []:
        return False
    if tags[-1][1] == 'VBP' or tags[-1][1] == 'VBZ':
        return True
    return False

def extract_keyvalue(sentences):
    noun = ''
    subj = []
    obj = []
    for item in sentences.split('.'):
        noun = ''
        objs = ''
        getObj = False
        item = cleanText(item)
        pos_tag = preprocess(item)
        # print(pos_tag)
        cnt = 0
        for i,tags in enumerate(pos_tag):
            if checkPassive(tags):
                getObj = True

            if getObj == False:
                if tags[1] != 'VBP' and tags[1] != 'VBZ':
                    noun += tags[0] + ' '
            elif getObj == True:
                if tags[0] != 'and':
                    objs += tags[0] + ' '
            
            if tags[1] == 'VBP' or tags[1] == 'VBZ' or i == len(pos_tag) - 1 or tags[0] == 'and' :
                if getObj:
                    obj.append(objs.strip())
                    if not checkSubjInLast(objs)[0]:
                        objs = ''

                    if noun != '':
                        subj.append(noun.strip())
                        noun = ''
                else:
                    subj.append(noun.strip())
                    noun = ''

                # if getObj and len(obj[-1].split(' ')) >= 3 and (preprocess(obj[-1])[-2][1] == 'EX' or preprocess(obj[-1])[-2][1] == 'EX'):
                if getObj and len(obj[-1].split(' ')) >= 3 and (preprocess(obj[-1])[-1][1] == 'VBZ' or preprocess(obj[-1])[-1][1] == 'VBP'):
                    subj , obj = solveEndingVerb(subj,obj)
                    getObj = not getObj

                if getObj and len(subj) >= 1 and subj[-1].split(' ')[-1] == 'and':
                    if isNounPhrase(obj[-1]) and obj[-1] != '':
                        subj[-1] = subj[-1] + ' ' + obj[-1]
                        obj.pop()
                        getObj = not getObj
                getObj = not getObj
        
        # subj, obj = refineSubjObj(subj,obj)
        if len(subj) > len(obj):
            obj.extend([0] * (len(subj) - len(obj)))
        elif len(subj) < len(obj):
            subj.extend([0] * (len(obj) - len(subj)))
    return subj, obj

def get_keyvalue(sentences):
    subj , obj = extract_keyvalue2(sentences)
    res = convert_list_to_dict(subj,obj)
    return res

def extract_keyvalue2(sentences):
    subj = []
    obj = []
    sentences = sentences.split('.')

    for sent in sentences:
        pos_tag = preprocess(sent)
        for i,item in enumerate(pos_tag):
            if item[1] == 'VBZ' or item[1] == 'VBP':
                subj.append(' '.join(sent.strip().split(' ')[:i]))
                obj.append(' '.join(sent.strip().split(' ')[i+1:]))
                break
    return subj,obj

def convert_list_to_dict(li1,li2):
    res = {}
    for i,item in enumerate(li1):
        res[item] = li2[i]
    return res
if __name__ == "__main__":
    text = 'trachea is midline . cardio cardiomediastinal silhouette is within normal limits . there is no focal consolidation , pleural effusion or airspace opacity . limited evaluation reveals mild multilevel degenerative changes of the spine . interval resolution of the high svc . negative for focal pulmonary nodules are clear . are normal , and pulmonary vascularity appear within normal limits , and there is no focal consolidations , pleural effusion , or airspace consolidation . no focal area of consolidation . no pneumothorax or pleural effusion . lungs are hyperaerated suggestive of focal airspace consolidation are normal contour normal . the thoracic aorta is noted .",the cardiac silhouette and mediastinum size are within normal limits . there is no pulmonary edema . there is no focal consolidation . there are no of a pleural effusion . there is no evidence of pneumothorax .'
    res = get_keyvalue(text)
    print(res)