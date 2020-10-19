import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu 
from rouge import Rouge 
from statistics import mean

def compute_bleu(hypo,ref):
    hypo = hypo.split()
    ref = ref.split()
    cc = SmoothingFunction()
    BLEUscore = sentence_bleu([ref], hypo , smoothing_function=cc.method4)
    return BLEUscore

def compute_rouge(hypo,ref):
    rouge = Rouge()
    scores = rouge.get_scores(hypo, ref)
    r1 = scores[0]['rouge-1']['f']
    r2 = scores[0]['rouge-2']['f']
    rl = scores[0]['rouge-l']['f']

    return r1,r2,rl

def get_metrics(preds,labels):
    bleu = []
    r1 = []
    r2 = []
    rl = []
    for hypo,ref in zip(preds,labels):
        bleu_score = compute_bleu(hypo,ref)
        r1_,r2_,rl_ = compute_rouge(hypo,ref)
        bleu.append(bleu_score)
        r1.append(r1_)
        r2.append(r2_)
        rl.append(rl_)
    return mean(bleu) , mean(r1) , mean(r2) , mean(rl)

if __name__ == "__main__":
    result = pd.read_csv("out.csv")
    preds = result['predict'].tolist()
    labels = result['label'].tolist()
    bleu, r1, r2 ,rl =  get_metrics(preds,labels)
    print("Bleu score :" , bleu)
    print("R1 = {} , R2 = {} , RL = {}".format(r1,r2,rl))