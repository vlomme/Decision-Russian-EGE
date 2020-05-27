import re
import random
import operator
import pymorphy2
from nltk.tokenize import ToktokTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from solvers.utils import BertEmbedder
import numpy as np



class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.is_train_task = False
        self.morph = pymorphy2.MorphAnalyzer()
        self.toktok = ToktokTokenizer()
        self.seed = seed
        self.init_seed()
      
    
    def init_seed(self):
        random.seed(self.seed)

    def predict(self, task):
        return self.predict_from_model(task)

    def get_num(self, text):
        lemmas = [self.morph.parse(word)[0].normal_form for word in self.toktok.tokenize(text)]
        if 'указывать' in lemmas and 'предложение' in lemmas:
            w = lemmas[lemmas.index('указывать') + 1]  # first
            d = {'один': 1,
                 'два': 2,
                 'три': 3,
                 'четыре': 4,
                 'предложение': 1}
            if w in d:
                return d[w]
        elif 'указывать' in lemmas and 'вариант' in lemmas:
            return 'unknown'
        return 1

    def compare_text_with_variants(self, text, variants, num=1):
        text_vector = self.sentence_embedding([text])
        variant_vectors = self.sentence_embedding(variants)
        i, predictions = 0, {}
        for j in variant_vectors:
            sim = cosine_similarity(text_vector[0].reshape(1, -1), j.reshape(1, -1)).flatten()[0]
            predictions[i] = sim*(len(variants[i])**(1/5))
            i += 1
        #print(1,predictions)
        #indexes = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)[:num]
        #print(2,indexes)
        #return [str(i[0] + 1) for i in indexes]
        return predictions
    

    
    def sent_split(self, text):
        reg = r'\(*\d+\)'
        return re.split(reg, text)

    def process_task(self, task):
        first_phrase, task_text = re.split(r'\(*1\)', task['text'])[:2]
        variants = [t['text'].replace("—","").replace("<...>","").replace("<…>","").replace(",","").replace(".","").replace(":","").replace("»","").replace("«","").replace("-"," ") for t in task['question']['choices']]
        text, task = "", ""
        if 'Укажите' in task_text:
            text, task = re.split('Укажите ', task_text)
            task = 'Укажите ' + task
        elif 'Укажите' in first_phrase:
            text, task = task_text.replace("—","").replace("<...>","").replace("<…>","").replace(",","").replace(".","").replace(":","").replace("»","").replace("«","").replace("-"," "), first_phrase
        return text, task, variants

    def fit(self, tasks):
        pass

    def load(self, path=""):
        pass
    
    def save(self, path=''):
        pass

    def predict_from_model(self, task, num=2):
        #print(task["id"])
        text, task, variants = self.process_task(task)
        text = re.sub('[0-9]*\)','',text).replace('   ',' ').replace('  ',' ')  
        for i, _ in enumerate(variants):
            variants[i] = re.sub('[0-9]*\)','',variants[i])
            variants[i] = variants[i].replace('   ',' ').replace('  ',' ')
        #print(text)
        #print(variants)
        result = self.compare_text_with_variants(text, variants, num=num)
        text = [text]
        text.extend(variants)
        result2 = self.compare_text_with_variants2(text)
        indexes1 = sorted(result.items(), key=operator.itemgetter(1), reverse=True)[:num]
        
        #print(1,[str(i[0] + 1) for i in indexes1])      
        indexes2 = sorted(result2.items(), key=operator.itemgetter(1), reverse=True)[-num:]
        #print(2,[str(i[0] + 1) for i in indexes2])   
        symm1,symm2 = 0,0
        for i in range(len(result)):
            symm1 +=result[i]
            symm2 +=result2[i]
        dif =  symm2/symm1   
        for i in range(len(result)):
            #print(i+1,result[i],result2[i])
            result[i] -=result2[i]/(dif*4)
        #print(result)
        indexes = sorted(result.items(), key=operator.itemgetter(1), reverse=True)[:num]
        ans = [str(i[0] + 1) for i in indexes]
        
        return sorted(ans)
