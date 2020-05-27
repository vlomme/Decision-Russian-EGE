import re
import random
from fuzzywuzzy import fuzz
import pickle
from string import punctuation
from operator import itemgetter
from solvers.utils import BertEmbedder
from solvers.utils import Bert
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy2
import re


class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.representatives = {}
        self.morph = pymorphy2.MorphAnalyzer()
        self.dic_known_word = []
        with open('./data/dic_known_word.csv', encoding='utf-8') as f:
            for line in f:
                self.dic_known_word.append(line.replace("\n",""))   
    def init_seed(self):
        return random.seed(self.seed)

    def get_target(self, task):
        solution = task["solution"]["correct_variants"] if "correct_variants" in task["solution"] \
            else [task["solution"]["correct"]]
        return solution

    def process_task(self, task):
        text = re.sub(r"([а-яё]+)\.([а-яё]+)", r"\1. \2", task["text"].lower().replace(".(", ". (").replace("?", ". "))
        words = [w.replace("(", "").replace(")", "") for w in
                 [w.strip(punctuation) for w in text.split() if any([s in w for s in ("(", ")")])]]
        return words

    def get_representatives(self, word, threshold=75):
        representatives = [rep for rep in self.representatives if fuzz.ratio(word, rep) >= threshold]
        return representatives

    def get_similarity(self, word, representatives):
        x, y = self.token_embedding([word]).reshape(1, -1), [self.representatives[rep] for rep in representatives]
        similarity = max([cosine_similarity(x, y_.reshape(1, -1))[0][0] for y_ in y])
        return similarity

    def fit(self, train):
        for task in train:
            words, solution = self.process_task(task), self.get_target(task)
            if len(words) == 10:
                for i in range(0, len(words), 2):
                    word1, word2 = words[i], words[i+1]
                    candidate = word1 + word2
                    if candidate in solution:
                        if word1 not in self.representatives:
                            self.representatives[word1] = self.token_embedding([word1])
                        if word2 not in self.representatives:
                            self.representatives[word2] = self.token_embedding([word2])

    def save(self, path="data/models/solver14.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.representatives, f)
    
    def load(self, path="data/models/solver14.pkl"):
        with open(path, "rb") as f:
            self.representatives = pickle.load(f)

    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        predictions = {}
        task1 = task['text'].replace("!", ". ").replace(":", " :").replace(")(", "").replace("«", "« ").replace("»", " »")
        
        sen = [word for word in task1.split("\n") if (1 < len(word)and not 'пишутся' in word and not 'выпишите' in word)]
        if (len(sen)<3):
            sen = re.findall('[^\)]*\([а-яёА-яЁ]*\)[^(]*[\(][а-яёА-яЁ]*\)[^\.\!\?\(]*',task1)
        del_word = []
        #words = self.process_task(task)
        #print(words)
        #print(sen).replace("«", "").replace("»", "")
        for j in range(len(sen)-1,-1,-1):
            s = ' '+sen[j].replace(","," , ").replace("."," . ").replace("?"," ? ").replace(":"," : ").replace(" ","").replace("  "," ")+' '
            word = re.findall('[^А-ЯЁа-яё]\([А-ЯЁ ]*\)[А-ЯЁ ][А-ЯЁ]*|[А-ЯЁ]*[А-ЯЁ ]\([А-ЯЁ ]*\)[^А-ЯЁа-яё]|\([А-ЯЁ]*\)[А-ЯЁ]*|[А-ЯЁ]*\([А-ЯЁ]*\)',s)
            word[0] = word[0].replace(",","").replace(".","").replace(":","").replace("«", "").replace("»", "")
            word[1] = word[1].replace(",","").replace(".","").replace(":","").replace("«", "").replace("»", "")
            w0 = word[0].replace(" ","").replace('(','').replace(')','').lower()
            w1 = word[1].replace(" ","").replace('(','').replace(')','').lower()
            #print(w0,self.morph.word_is_known(w0),w1,self.morph.word_is_known(w1))
            if len(word)>1 and (not self.morph.word_is_known(w0) and not w0 in self.dic_known_word or not self.morph.word_is_known(w1) and not w1 in self.dic_known_word):
                sen.pop(j) 
            else:    
                for i in range(len(word)-1,-1,-1):
                    word[i] = word[i].strip()
                    s = s.replace(word[i],'[MASK]')
                    word[i] = word[i].replace(" ","").lower()
                    del_word.append(word[i])
               
                sen[j] = s
 
        #print(sen)
        #print(del_word)            
        #if predictions:
        #    for j in range(len(sen)-1,-1,-1):
        #        if not j in id_w:
        #            sen.pop(j) 
        #            del_word.pop(j)                    
       
        w_x = del_word[::-1]
        sentence = sen
        pred,word_no_pred,word_pred,word_all,sentence1 = [],[],[],[],[]
        #print(sentence)    
        #print(w_x)
        for i in range(len(w_x)):
            pred.append(re.search('\(.*?\)',w_x[i]).group(0))
            pred[i] = pred[i].replace('(','').replace(')','').lower()
            word_no_pred.append(re.sub('\(.*?\)',"",w_x[i]).lower())
            if word_no_pred[i] =="":
                word_no_pred[i] = 'в'
            word_pred.append(w_x[i].replace('(','').replace(')','').lower())
            
        for i in range(len(sentence)):
            if (w_x[2*i][0]=='('):
                sentence1.append(sentence[i].replace('[MASK]',pred[2*i]+' MASK]', 1))
            else:
                sentence1.append(sentence[i].replace('[MASK]','MASK] '+pred[2*i], 1))
            if (w_x[2*i+1][0]=='('):
                sentence1[i] = sentence1[i].replace('[MASK]',pred[2*i+1]+' [MASK]', 1)
            else:
                sentence1[i] = sentence1[i].replace('[MASK]','[MASK] '+pred[2*i+1], 1)
            #print(sentence1)
            sentence1[i] = sentence1[i].replace('MASK]','[MASK]', 1)

        #print(sentence1)
        #print(pred)
        #print(word_no_pred)
        #print(word_pred)    
        sentence = ' '.join(sentence) + ' '.join(sentence1) + ' .'
        #print(sentence)
        for i in range(len(word_pred)):
            word_all.append(word_pred[i])    
        for i in range(len(word_no_pred)):
            word_all.append(word_no_pred[i])
        #print(sentence)
        #print(word_all)        
        result = Bert().what_mask14(sentence,word_all)
        #result ='1'
        return result