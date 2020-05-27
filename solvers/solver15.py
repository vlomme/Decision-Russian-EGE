import re
import random
import pickle
from fuzzywuzzy import fuzz
from string import punctuation
from operator import itemgetter
from solvers.utils import BertEmbedder
from sklearn.metrics.pairwise import cosine_similarity
from solvers.utils import Bert
import pymorphy2

class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.representatives = {"n": {}, "nn": {}}
        self.morph = pymorphy2.MorphAnalyzer()
        self.dic_known_word = []
        with open('./data/dic_known_word.csv', encoding='utf-8') as f:
            for line in f:
                self.dic_known_word.append(line.replace("\n",""))          
    def init_seed(self):
        return random.seed(self.seed)

    def get_target(self, task):
        solution = task["solution"]["correct_variants"][0] if "correct_variants" in task["solution"] else \
            task["solution"]["correct"]
        return solution

    def process_task(self, task):
        text = task["text"].replace("?", ".").replace("\xa0", "")
        placeholders = [choice["placeholder"] for choice in task["question"]["choices"]]
        words = [re.sub("[^а-яё]+", "0", word.strip(punctuation)) for word in text.split() if ")" in word
                 and any([n in word for n in placeholders])]
        n_number = text.split(".")[0].split()[-1].lower()
        return text, words, n_number

    def get_representatives(self, word, representatives, threshold=70):
        representatives = [rep for rep in representatives if fuzz.ratio(word, rep) >= threshold]
        return representatives

    def get_similarity(self, x, representatives):
        y = [self.representatives["n"][rep] if rep in self.representatives["n"]
             else self.representatives["nn"][rep] for rep in representatives]
        similarity = max([cosine_similarity(x, y_.reshape(1, -1))[0][0] for y_ in y])
        return similarity
    
    def parse_representatives(self, task):
        text, words, n_number = self.process_task(task)
        solution = self.get_target(task)
        if len(n_number) == 1:
            n_words = [re.sub("[^а-яё]+", "н", word.strip(punctuation)) for word in text.split()
                       if any([d in word for d in solution]) and ")" in word]
            for word in n_words:
                if word not in self.representatives["n"]:
                    self.representatives["n"][word] = self.token_embedding([word])
            for word in words:
                n_replacement = word.replace("0", "н")
                nn_replacement = word.replace("0", "нн")
                if n_replacement not in n_words and nn_replacement not in self.representatives["nn"]:
                    self.representatives["nn"][nn_replacement] = self.token_embedding([nn_replacement])
        elif len(n_number) == 2:
            nn_words = [re.sub("[^а-яё]+", "нн", word.strip(punctuation)) for word in text.split()
                        if any([d in word for d in solution]) and ")" in word]
            for word in nn_words:
                if word not in self.representatives["nn"]:
                    self.representatives["nn"][word] = self.token_embedding([word]) 
            for word in words:
                n_replacement = word.replace("0", "н")
                nn_replacement = word.replace("0", "нн")
                if nn_replacement not in nn_words and n_replacement not in self.representatives["n"]:
                    self.representatives["n"][n_replacement] = self.token_embedding([n_replacement])

    def fit(self, tasks):
        for task in tasks:
            self.parse_representatives(task)
            
    def save(self, path="data/models/solver15.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.representatives, f)
    
    def load(self, path="data/models/solver15.pkl"):
        with open(path, "rb") as f:
            self.representatives = pickle.load(f)

    def predict_from_model(self, task):
        """prediction = []
        text, words, n_number = self.process_task(task)
        print(text)
        print(words)
        print(n_number)
        for i, word in enumerate([word.replace("0", "н") for word in words]):
            representatives = {}
            x = self.token_embedding([word]).reshape(1, -1)
            c1 = self.get_representatives(word, self.representatives["n"])
            c2 = self.get_representatives(word, self.representatives["nn"])
            #print(c1,c2)
            if c1:
                representatives["н"] = self.get_similarity(x, c1)
            if c2:
                representatives["нн"] = self.get_similarity(x, c2)
            print(representatives)
            if representatives:
                answer = max(representatives.items(), key=itemgetter(1))[0]
                if answer == n_number:
                    prediction.append(str(i + 1))
        if prediction: 
            answer = sorted(prediction) 
        else:
            answer = ["1"]
        print('Чужое',prediction)"""
        #print('-'*40,task['id'],'-'*40)
        words1,words_z ,del_word1,del_word2= [],[],[],[]
                
        text = task['text'].replace(","," ,").replace("."," .").replace(";"," ;").replace(":"," :").replace("»"," »").replace("«","« ").replace("(З)","(3)")
        if 'пишется НН.'.lower() in task['text'].lower():
            wol = 'нн'
            wol2 = 'н'
        elif 'пишется Н.'.lower() in task['text'].lower() or 'одна буква Н'.lower() in task['text'].lower() or 'одно Н'.lower() in task['text'].lower() or 'одна Н.'.lower() in task['text'].lower() :
            wol = 'н'
            wol2 = 'нн'
        else:
            print('нн=(')
            wol = 'нн'
            wol2 = 'н'
        words = text.split()
        for j, word in enumerate(words):
            #print(re.search('\([0-9]\)',word))
            if (re.search('\([0-9]\)',word) is not None):
                
                id_t = re.search('[0-9]',word).group(0)
                
                word1 = re.sub('\([0-9]\)',wol,word)
                word2 = re.sub('\([0-9]\)',wol2,word)
                
                est_word1 = self.morph.word_is_known(word1) or word1 in self.dic_known_word
                est_word2 = self.morph.word_is_known(word2) or word2 in self.dic_known_word
                #print(id_t,word1,est_word1,word2,est_word2)
                if (est_word1 and not est_word2):
                    words1.append(id_t)
                    words[j] = word1
                elif (est_word1 and est_word2):
                    words[j] = '[MASK]'
                    words_z.append(id_t)
                    del_word1.append(word1)
                    del_word2.append(word2)
                else:
                    words[j] = word2
        #for word in words_z:
        #    text = re.sub('\(['+word+']\)','[MASK]',text)
        text =' '.join(words)
        text = re.sub('\([0-9]\)','',text)
        text = re.sub('  ',' ',text) 
        
        del_word1.extend(del_word2)
        #print(text)
        #print(del_word1)
        if wol =='нн':
            delta = 0.047
        else:
            delta = -0.06
            #-0.047
        if del_word1:
            results = Bert().what_mask15(text+text,del_word1,delta)
            for result in results:
                #print(result,del_word1.index(result))
                if (del_word1.index(result)<len(words_z)):
                    words1.append(words_z[del_word1.index(result)])
        #print(del_word1)
        words1.sort()
        answer = words1        
        #print('Моё',answer)
        
        
        
        
        
        
        
        return answer