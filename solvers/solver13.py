import random
import pickle
import numpy as np
from fuzzywuzzy import fuzz
from operator import itemgetter
from string import punctuation
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
        sentences = [t.strip() for t in task["text"].lower().replace("…", ".").replace("!", ".").split(".")
                     if "(не)" in t]
        words = [[w.strip(",.") for w in s.split() if "(не)" in w][0] for s in sentences]
        return words

    def get_representatives(self, word, threshold=75):
        representatives = [rep for rep in self.representatives if fuzz.ratio(word, rep) >= threshold]
        return representatives

    def get_similarity(self, word, representatives):
        x = self.token_embedding([word]).reshape(1, -1)
        y = [self.representatives[rep] for rep in representatives]
        similarity = max([cosine_similarity(x, y_.reshape(1, -1))[0][0] for y_ in y])
        return similarity

    def fit(self, train):
        for task in train:
            solution = self.get_target(task)
            for word in solution:
                word = word.strip(punctuation).replace("не", "(не)")
                if word not in self.representatives:
                    self.representatives[word] = self.token_embedding([word])

    def save(self, path="data/models/solver13.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.representatives, f)

    def load(self, path="data/models/solver13.pkl"):
        with open(path, "rb") as f:
            self.representatives = pickle.load(f)

    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        predictions,predictions2, words = {},{}, self.process_task(task)
        if 'какую букву' in task['text']:
            task1 = task['text'].replace("..", "и").replace("!", ". ").replace(":", " :").replace(") ", ")")
        else:
            task1 = task['text'].replace("!", ". ").replace(":", " :").replace(") ", ")")
        sen = [word for word in task1.split("\n") if (1 < len(word)and not 'пишется' in word and not 'выпишите' in word and not 'какую букву' in word and not '(НЕ)КТО иной,' in word)]
        if (len(sen)<3):
            sen = [word for word in task1.split(".") if (1 < len(word) and not 'пишется' in word and not 'выпишите' in word and not 'какую букву' in word and not '(НЕ)КТО иной,' in word)]
        #print(sen)
        #print(words)

        for j, s in enumerate(sen):
            if (re.search('\(не\)смотря (на меня|[^н])',s.lower())):
                sen.pop(j)          
        
        for j in range(len(words)-1,-1,-1):
            s = words[j].replace("(не)", "не").replace("(ни)", "ни").replace(":", "")
            if not self.morph.word_is_known(s) and not s in self.dic_known_word:
                #sen.pop(j) 
                words.pop(j)
                continue
            if 'некто' == s or 'несмотря' == s:
                words.pop(j)
        #print(sen)        
        #print(words)
        id_w = []
        for j, word in enumerate(words):
            representatives = self.get_representatives(word)
            #print(representatives)
            if representatives:
                similarity = self.get_similarity(word, representatives)
                #print(similarity)
                word = word.replace("(не)", "не").strip(punctuation)
                #print(similarity, word)
                predictions2[word] = similarity
                if (similarity>0.99):
                    id_w.append(j)
                    predictions[word] = similarity
                    
        #print(predictions)
        del_word = [] 
        if len(predictions)==1:
            #result = predictions.keys()[0] 
            result = max(predictions.items(), key=itemgetter(1))[0] if predictions else words[0].replace("(не)", "не").strip(punctuation)

        else:
            for j in range(len(sen)-1,-1,-1):
                s = sen[j].replace(","," , ").replace("."," . ").replace(":"," : ")
                word = s.split()
                #print(word)
                for i, x in enumerate(word):
                    if (')' in x and'(' in x and len(x)>6):
                        x = x.replace('(','').replace(')','').lower()
                        if self.morph.word_is_known(x) or x in self.dic_known_word:
                            del_word.append(word[i])
                            word[i] = '[MASK]'                
                sen[j] = ' '.join(word)
                if not '[MASK]' in sen[j]:
                    sen.pop(j) 
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
            for i in range(len(w_x)):
                pred.append(re.search('\(.*?\)',w_x[i]).group(0))
                pred[i] = pred[i].replace('(','').replace(')','').lower()

                word_no_pred.append(re.sub('\(.*?\)',"",w_x[i]).lower())

                word_pred.append(w_x[i].replace('(','').replace(')','').lower())
            for i in range(len(sentence)):
                sentence1.append(sentence[i].replace('[MASK]',pred[i]+' [MASK]', 1))
            sentence = ' . '.join(sentence) +' . '+ ' . '.join(sentence1)
            for i in range(len(word_pred)):
                word_all.append(word_pred[i])    
            for i in range(len(word_no_pred)):
                word_all.append(word_no_pred[i])
            #print(sentence)
            #print(word_all)        
            result = Bert().what_mask13(sentence,word_all,predictions2)
        #print(result)

        
        
        return result