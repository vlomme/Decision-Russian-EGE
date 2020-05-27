import random
import pickle
import numpy as np
from string import punctuation
from fuzzywuzzy import fuzz, process
from solvers.utils import BertEmbedder
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy2
import re

class Solver(BertEmbedder):

    def __init__(self, seed=42, rnc_path="data/1grams-3.txt"):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.representatives = {}
        self.rnc_path = rnc_path
        self.rnc_unigrams = self.lazy_unigrams(self.rnc_path)
        self.bad_form = {}
        with open('./data/bad_form.csv', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n","").lower().split('|')
                self.bad_form[line[0]] = line[1]
    def init_seed(self):
        return random.seed(self.seed)

    @staticmethod
    def lazy_unigrams(rnc_path):
        unigram_freq_dict = {}
        for line in open(rnc_path, "r", encoding="utf-8").read().split("\n"):
            pair = line.lower().split("\t")
            try:
                freq, unigram = int(pair[0]), " ".join([el for el in pair[1:]
                                                        if el is not "" and el not in punctuation])
                if unigram not in unigram_freq_dict:
                    unigram_freq_dict[unigram] = freq
                else:
                    unigram_freq_dict[unigram] += freq
            except ValueError:
                pass
        return unigram_freq_dict

    def get_target(self, task):
        solution = task["solution"]["correct_variants"] if "correct_variants" in task["solution"] else [
            task["solution"]["correct"]]
        return solution

    def process_task(self, task):
        text = " ".join([t for t in task["text"].split(".") if "\n" in t])
        words = [w.strip(punctuation).split() for w in text.split("\n") if "Исправьте" not in w and len(w) > 1]
        words = [" ".join([w.lower() for w in words_ if w.isupper()]) for words_ in words]
        return words
        
    def process_task2(self, task):
        text = " ".join([t for t in task["text"].split(".") if "\n" in t])
        words = [w.strip(punctuation).split() for w in text.split("\n") if "Исправьте" not in w and len(w) > 1]
        words = [" ".join([w for w in words_]) for words_ in words]
        return words
        
    def get_representatives(self, word, threshold=65):
        representatives = [rep for rep in self.representatives if fuzz.ratio(word, rep) >= threshold]
        return representatives

    def get_error_word(self, words):
        frequencies = [self.rnc_unigrams[w] if w in self.rnc_unigrams else 10 for w in words]
        error_word = words[np.argmin(frequencies)]
        return error_word

    def get_similarity(self, word, representatives):
        x, y = self.token_embedding([word]).reshape(1, -1), [self.representatives[rep] for rep in representatives]
        similarities = [cosine_similarity(x, y_.reshape(1, -1))[0][0] for y_ in y]
        prediction = representatives[np.argmax(similarities)]
        return prediction

    def fit(self, train):
        for task in train:
            words, solution = self.process_task(task), self.get_target(task)
            error_word = process.extractOne(solution[0], words)[0]
            for word in words:
                if word != error_word and word not in self.representatives:
                    self.representatives[word] = self.token_embedding([word])
            for correct in solution:
                if correct not in self.representatives:
                    self.representatives[correct] = self.token_embedding([correct])

    def save(self, path="data/models/solver7.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.representatives, f)
    
    def load(self, path="data/models/solver7.pkl"):
        with open(path, "rb") as f:
            self.representatives = pickle.load(f)

    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        morph = pymorphy2.MorphAnalyzer()
        bad_form = self.bad_form
        w = [] 
        w_id  = [] 
        may_be = []      
        words = self.process_task(task)
        words2 = self.process_task2(task)
        #print(words)
        #print(words2)
        cases = {}
        
        for i, x in enumerate(words):
            x = x.lower().replace('ьi','ы').replace('более ','').replace(' ','')
            if not morph.word_is_known(x):
                y = bad_form.get(x)
                if y:
                    w_id.append(i)
                    w.append(y)
                else:
                    print('-'*10,x)
        #print('Нет слов',w)

        if not w:
            if re.search('БОЛЕЕ [А-ЯЁ]*(Е|ИЙ)([^А-ЯЁ]|$)',task["text"]):
                y = re.search('БОЛЕЕ [А-ЯЁ]*(Е|ИЙ)([^А-ЯЁ]|$)',task["text"]).group(0)
                w.append(y.lower().replace('более ','').strip())
                w_id.append(-1)
                #print('БОЛЕЕ ','-'*10,w)
              
        if not w:
            if re.search('более [А-ЯЁ]*(Е|ИЙ)([^А-ЯЁ]|$)',task["text"]):
                y = re.search('более [А-ЯЁ]*(Е|ИЙ)([^А-ЯЁ]|$)',task["text"]).group(0).replace('более ','').strip()
                word2 = morph.parse(y)[0]
                if word2.tag.POS == 'NUMR' and word2.inflect({'gent'}) is not None:
                    y = word2.inflect({'gent'}).word
                else:
                    y = re.sub('ЕЕ$','о',y)
                    y = re.sub('Е$','о',y)
                    y = re.sub('ИЙ$','о',y)
                if not morph.word_is_known(y) and morph.word_is_known(y+'го'):
                    y = y+'го'
                #print(y,morph.word_is_known(y.replace('Жо','го')))
                if not morph.word_is_known(y) and morph.word_is_known(y.replace('Жо','го')):
                    y = y.replace('Жо','го')
                if morph.word_is_known(y):
                    w.append(y.lower())
                    w_id.append(-1)
                    #print('более ','-'*10,w)
         
        #if not w:
        for i, x in enumerate(words2):
            n_case, p_case = False, False
            bad,bad1 = False, False
            x = x.split()
            all_cases = ([morph.parse(ok) for ok in x])     
            pos = ([str(morph.parse(ok)[0].tag.POS) for ok in x])
            #print(x)
            #print(pos)
            #print(case)
            for j, p in enumerate(pos):
                if p != 'PREP' and x[j].lower() != 'более' and not x[j].isupper():
                    n_case = all_cases[j]
                if x[j].isupper():
                    p_case = all_cases[j]
                if p == 'PREP' and (n_case and not p_case or not n_case and p_case):
                    n_case, p_case = False, False

            if n_case and p_case:
                cases[i] = n_case[0]
                for cas_j in n_case:
                    for cas_z in p_case:
                        if (cas_z.tag.POS == 'NOUN' and cas_j.tag.POS == 'NOUN'):
                            bad1 = True
                            break  
                        elif (cas_z.tag.POS == 'NOUN' or cas_z.tag.POS == 'NUMR') and (cas_j.tag.POS == 'NOUN' or cas_j.tag.POS == 'NUMR'):
                            bad = True
                            bad1 = True
                            break
                     
                        if cas_j.score>0.02 and cas_z.score>0.02 and (cas_j.tag.case == cas_z.tag.case or str(cas_j.tag.case) == 'None' or str(cas_z.tag.case) == 'None')  and (cas_j.tag.number == cas_z.tag.number or str(cas_j.tag.number) == 'None' or str(cas_z.tag.number) == 'None'): 
                            bad = True
                            bad1 = True
                            break
            else:
                bad = True
                bad1 = True
            if not w and (not bad and not bad1):
                w.append(p_case[0].word)
                w_id.append(i)
                #print('Не соответствует',words2[i],'-----|-----',p_case[0].word)
            if not w and (not bad):
                word2 = p_case[0]
                p = n_case[0]
                if p.tag.number and word2.inflect({p.tag.number}) is not None:
                    word2 = word2.inflect({p.tag.number})
                if word2.inflect({'gent'}) is not None:
                    word2 = word2.inflect({'gent'})                
                #print('Мей би',words2[i],'-----|-----',word2.word)
                may_be.append(word2.word)     

    
        if not w:
            #print('!!!!!')
            for i, x in enumerate(words):
                y = bad_form.get(x.lower().replace('ьi','ы').replace(' ',''))
                if y:
                    w.append(y)   
                    w_id.append(i)        
        #print('Плохая форма',w)
        
        if w:
            #print('!!!!',w,w_id,cases)
            for i,iw in enumerate(w):
                if w_id[i] in cases and (morph.parse(iw)[0].tag.POS == 'NOUN' or morph.parse(iw)[0].tag.POS == 'NUMR'):
                    p = cases[w_id[i]]
                    #print('=0',p.word,p.tag.POS,iw,morph.parse(iw)[0].tag.POS)
                    if p.tag.POS == 'NOUN' and morph.parse(iw)[0].tag.POS == 'NOUN':
                        continue
                    if p.tag.POS == 'NUMR' and morph.parse(iw)[0].tag.POS == 'NOUN':
                        continue                    
                    if p.word == 'году' or p.word == 'года':
                        continue                    
                    word2 = morph.parse(iw)[0]
                    #print('=1',p.word,p.tag.case,word2.word,word2.tag.case)
                    if p.tag.case and word2.inflect({p.tag.case}) is not None:
                        word2 = word2.inflect({p.tag.case}) 
                        #print('=2',p.word,p.tag.case,word2.word,word2.tag.case)                        
                    if p.tag.POS == 'VERB' and p.tag.transitivity == 'tran' and word2.inflect({'accs'}) is not None:
                        word2 = word2.inflect({'accs'})
                        #print('=3',p.word,p.tag.case,word2.word,word2.tag.case)
                    if p.tag.number and word2.tag.number and word2.inflect({p.tag.number}) is not None:
                       word2 = word2.inflect({p.tag.number})
                       #print('=4',p.word,p.tag.case,word2.word,word2.tag.case)
                    #print('Склоняю',iw,word2.word)
                    
                    w[i] = word2.word
                    if w[i] == 'договора':
                        w[i] = 'договоры'
        
        if not w and may_be:
            #print('123+++++',may_be)
            w = may_be
     
        if w:
            result = random.choice(w)
            if result == 'ихние':
                result = 'их'
        else:
            #print("!!!")
            error_word = self.get_error_word(words)
            representatives = self.get_representatives(error_word)
            if representatives:
                prediction = self.get_similarity(error_word, representatives)
                return prediction
            result = words[0].strip(punctuation)
        return result