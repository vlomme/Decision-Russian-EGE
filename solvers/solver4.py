import re
import os
import random
from string import punctuation
import pymorphy2

class Solver(object):

    def __init__(self, seed=42, data_path='data/'):
        self.morph = pymorphy2.MorphAnalyzer()
        
        self.dect_acs = []
        self.dect_acs2 = []
        #self.is_train_task = False
        self.seed = seed
        self.init_seed()
        self.stress = open(os.path.join(data_path, 'agi_stress.txt'), 'r', encoding='utf8').read().split('\n')[:-1]
        with open('./data/acs.csv', encoding='utf-8') as f:
            for line in f:
                self.dect_acs.append(line.replace("\n",""))
        with open('./data/ac2.csv', encoding='utf-8') as f:
            for line in f:
                self.dect_acs2.append(line.replace("\n",""))
    def init_seed(self):
        random.seed(self.seed)

    def predict(self, task):
        return self.predict_from_model(task)


    def process_task(self, task):
        task_text = re.split(r'\n', task['text'])
        variants = task_text[1:-1]
        if 'Выпишите' in task_text[-1]:
            task = task_text[0] + task_text[-1]
        else:
            task = task_text[0]
        if 'неверно' in task.lower():
            task_type = 'incorrect'
        else:
            task_type = 'correct'
        return task_type, task, variants

    def fit(self, tasks):
        pass

    """def load(self, path="data/models/solver4.pkl"):
        pass

    def save(self, path="data/models/solver4.pkl"):
        pass"""

    def predict_from_model(self, task):

        #task_type, task, variants = self.process_task(task)
        #print(task_type)
        #print(task)
        #print(task['text'])
        #print(task['text'].split('\\n'))
        words = [word for word in task['text'].split("\n") if (1 < len(word))]
        if len(words)<2:
            words = [word for word in task['text'].split("\\n") if (1 < len(word))]
        words1 = words[1:]
        #print(words1)
        words = []
        dect_acs2 = self.dect_acs2
        dect_acs = self.dect_acs        
        #print(dect_acs[:100])
        words1 = (" ".join(words1))
        words1 = re.findall('[а-яё]*[А-ЯЁ][а-яё]*',words1)
        #print(words1)
        for word in words1:
            if  (not 'Выпишите' in word):
                #print(word, word in dect_acs)
                #if not (word in dect_acs or self.morph.parse(word)[0].normal_form in dect_acs or 'по'+word in dect_acs):
                if not (word in dect_acs):
                    words.append(word.lower())
        #print(words)
        if not words:
            for word in words1:
                if (word.lower() in dect_acs2) and not ('Выпишите' in word):
                    words.append(word.lower())
                    break
        #print(words)
        result = random.choice(words)
        #result = self.compare_text_with_variants(variants, task_type)
        return result
