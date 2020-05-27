import re
import random
from solvers.utils import standardize_task, AbstractSolver
import pymorphy2

class Solver(AbstractSolver):
    def __init__(self, **kwargs):
        self.morph = pymorphy2.MorphAnalyzer()
        super().__init__()
        self.prov_chered = []
        self.prov_yes = []
        self.prov_no = []
        self.dect_acs = []
        with open('./data/prov_chered.csv', encoding='utf-8') as f:
            for line in f:
                self.prov_chered.append(line.replace("\n",""))
        with open('./data/prov_yes.csv', encoding='utf-8') as f:
            for line in f:
                self.prov_yes.append(line.replace("\n",""))
        with open('./data/prov_no.csv', encoding='utf-8') as f:
            for line in f:
                self.prov_no.append(line.replace("\n",""))
        with open('./data/acs.csv', encoding='utf-8') as f:
            for line in f:
                self.dect_acs.append(line.replace("\n",""))                
    def predict_from_model(self, task):                                   
        #print('-'*40,task['id'],'-'*40)
        result = []
        if "чередующаяся" in task['text']:
            prov = 3
        elif "непроверяемая" in task['text']:
            prov = 1
        elif "проверяемая" in task['text']:
            prov = 2    
        else: prov = 0  

        for i, choice in enumerate(task["question"]["choices"]):
            
            choice1 = re.sub(" \(.*?\)","",choice['text'])
            choice1 = re.sub("\(.*?\) ","",choice1).replace(")","").replace("(","")
            choice1 = re.sub("[0-9]","",choice1).strip()
            parts = [x for x in choice1.split(', ')]
            parts = [x for x in parts]
            #print(parts,i,prov)
            x,vowel = self.get_answer_by_vowel_9(parts,i,prov)
            #print(x)
            if x:
                if task["question"]["type"] =="text":
                    result.append(choice['text'].replace('..',vowel).lower())
                else:
                    result.append(choice['id'])
        result.sort()
        if (not result):
            result = ['2']
        if task["question"]["type"] =="text":
            result = ''.join(result)
        return result
    def get_answer_by_vowel_9(self,parts,i,prov = 0):
        #print(parts)
        n_word1 = 0
        n_word2 = 0
        result = ''
        w  = ''
        end_t = False
        word2,word3 = [],[]    
        for word in parts:
            chered_true = False
            if end_t:
                break
            for vowel in "ЭОУАЫЕЁЮЯИ":
                word1 = word.replace("..", vowel)
                all_p = []
                if (self.morph.word_is_known(word1)):
                    w  = ''
                    acs = word1 in self.dect_acs
                    if (prov == 2):
                        if ((word1.lower() in self.prov_yes) and not acs):
                            break
                        elif (word1.lower() in self.prov_no or word1.lower() in self.prov_chered):
                            word2 = []
                            end_t = True
                            break                   
                        else:
                            w = word1
                    elif (prov == 1):
                        if ((word1.lower() in self.prov_no) and not acs):
                            break
                        elif (word1.lower() in self.prov_yes or word1.lower() in self.prov_chered):
                            word2 = []
                            end_t = True
                            break                    
                        else:
                            w = word1
                    elif (prov == 3):
                        if (word1.lower() in self.prov_chered and not acs):
                            break
                        elif (word1.lower() in self.prov_yes or word1.lower() in self.prov_no):
                            #print(word1)
                            word2 = []
                            end_t = True
                            break
                        else:
                            w = word1
            if w:
                word2.append(w)
        #print(end_t, word2)
        if not(word2) and not end_t:
            result = str(i+1)
        elif (len(word2)+1<len(parts) and not end_t):
            result = str(i+1)
        elif prov == 2 and not end_t:
            result = str(i+1)
        else: 
            result = ''
            
        return result, vowel