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
    text = text.strip().split(' ')

    if text[0] == 'and':
        text.remove(text[0])
    return ' '.join(text)

def checkPassive(text):
    if text[1] == 'JJ' and text[0][-2:] == 'ed':
        return True
    return False

def solveEndingVerb(subj,obj,idx,ix):
    sent = obj[-1]
    pos_tag = preprocess(sent)
    for i in range(len(pos_tag) - 1, -1 , - 1):
        # print(i, pos_tag[i])
        if pos_tag[i][1] == 'EX' or pos_tag[i][1] == 'DT' or ( pos_tag[i][1] == 'JJ' and pos_tag[i-1][1] != 'NN' and pos_tag[i-1][1] != 'NNS'):
            mark = i
            obj[-1] = ' '.join(sent.split(' ')[:mark])
            subj.append(' '.join(sent.split(' ')[mark:]))
            idx.append(ix)
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
                # if tags[1] != 'VBP' and tags[1] != 'VBZ':
                noun += tags[0] + ' '
            elif getObj == True:
                # if tags[0] != 'and':
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
                    subj , obj = solveEndingVerb(subj,obj,idx,ix)
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
    subj , obj = extract_keyvalue(sentences)
    res = convert_list_to_dict(subj,obj)
    return res

def convert_list_to_dict(li1,li2):
    res = {}
    for i,item in enumerate(li1):
        res[item] = li2[i]
    return res
if __name__ == "__main__":
    data = pd.read_csv('out.csv')
    src = data['predict'].tolist()
    noun = ''
    subj = []
    obj = []
    idx = []
    for ix,sentences in enumerate(src):
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
                    # if tags[1] != 'VBP' and tags[1] != 'VBZ':
                    noun += tags[0] + ' '
                elif getObj == True:
                    # if tags[0] != 'and':
                    objs += tags[0] + ' '
                
                if tags[1] == 'VBP' or tags[1] == 'VBZ' or i == len(pos_tag) - 1 or tags[0] == 'and' :
                    if getObj:
                        obj.append(objs.strip())
                        if not checkSubjInLast(objs)[0]:
                            objs = ''

                        if noun != '':
                            subj.append(noun.strip())
                            noun = ''
                            idx.append(ix)
                    else:
                        subj.append(noun.strip())
                        noun = ''
                        idx.append(ix)

                    # if getObj and len(obj[-1].split(' ')) >= 3 and (preprocess(obj[-1])[-2][1] == 'EX' or preprocess(obj[-1])[-2][1] == 'EX'):
                    if getObj and len(obj[-1].split(' ')) >= 3 and (preprocess(obj[-1])[-1][1] == 'VBZ' or preprocess(obj[-1])[-1][1] == 'VBP'):
                        subj , obj = solveEndingVerb(subj,obj,idx,ix)
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
                idx.extend([ix] * (len(subj) - len(idx)))
            elif len(subj) < len(obj):
                subj.extend([0] * (len(obj) - len(subj)))
                idx.extend([ix] * (len(obj) - len(idx)))

    print(len(idx))
    print(len(subj))
    print(len(obj))
    # print(subj)
    # print(obj)
    # padding
    # obj.extend([0] * (len(subj) - len(obj)))
    out_df = pd.DataFrame({
                            'idx': idx,
                            'subj':subj,
                            'obj':obj
                        })
    out = out_df.to_csv('split_sentence.csv',index = False)