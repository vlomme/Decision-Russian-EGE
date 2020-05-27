import re
import random
import pymorphy2
from solvers.utils import standardize_task, AbstractSolver


class Solver(object):
    """
    Solver for tasks 10, 11, 12
    """
    def get_answer_by_vowel_10(self, choices,vowel,num = 0):
        morph = self.morph
        result = []
        
        for i, choice in enumerate(choices):
            choice1 = re.sub(" \(.*?\)","",choice)
            choice1 = re.sub("\(.*?\) ","",choice1).replace(")","").replace("(","").replace("...","..").replace(".. ","..").replace(". ",", ").replace(";",",").replace(","," , ")
            choice1 = re.sub("[0-9]","",choice1).strip().lower()
            parts = re.findall('[а-яёА-Я]*[\.][\.][а-яё]*',choice1)
            #parts = [re.sub(r"^\d\) ?\(.*?\) ?","",x) for x in choice1.split(', ')]
            parts = [x.replace("..", vowel).strip() for x in parts]
            if all(morph.word_is_known(word) or word in self.dic_known_word for word in parts): 
                #print(parts)
                if num == 1:
                    result.append(str(i+1))
                else:
                    result.append("".join(parts))
        #print(1,num,result)
        if num == 0:
            if result:
                result = result[0]
            else:
                result = ''
        else:
            result = sorted(result)
        return result    
    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.morph = pymorphy2.MorphAnalyzer()
        self.dic_known_word = []
        with open('./data/dic_known_word.csv', encoding='utf-8') as f:
            for line in f:
                self.dic_known_word.append(line.replace("\n",""))


    def init_seed(self):
        return random.seed(self.seed)

    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        words = [word for word in task['text'].split("\n") if (1 < len(word))]
        if task['question']['type'] == 'multiple_choice':
            words = []
            num = 1
            for ans in task['question']['choices']:
                words.append(ans['text'])
        else:
            num = 0
            words = words[1:]
        #print(words)
        result = []
        match = re.search(r'буква ([ЭОУАЫЕЁЮЯИ])',task['text'])
        
        if match:
            letter = match.group(1)
            return self.get_answer_by_vowel_10(words,letter.lower(),num)
        elif "одна и та же" in task['text']:
            for vowel in "эоуаыеёюяидтсзьъ":
                result_with_this_vowel = self.get_answer_by_vowel_10(words,vowel,num)
                if num == 1:
                    result.extend(result_with_this_vowel)
                elif result_with_this_vowel:
                    result = result_with_this_vowel
                    break
        #print('я',result)
        if not result:
            result, task = [], standardize_task(task)
            #print(task)
            match = re.search(r'буква ([ЭОУАЫЕЁЮЯИ])', task["text"])
            if match:
                letter = match.group(1)
                return self.get_answer_by_vowel(task["question"]["choices"], letter.lower(),num)
            elif "одна и та же буква" in task["text"]:
                for vowel in "эоуаыеёюяидтсз":
                    result_with_this_vowel = self.get_answer_by_vowel(task["question"]["choices"], vowel,num)
                    result.extend(result_with_this_vowel)
            #print('не я',result)
        #print(num,result)
        if num == 1:
            answer = sorted(list(set(result)))
        else:
            answer = result
        return answer

    def get_answer_by_vowel(self, choices, vowel,num = 1):
        result = list()
        
        for choice in choices:
            parts = [re.sub(r"^\d\) ?| ?\(.*?\) ?", "", x) for x in choice["parts"]]
            parts = [x.replace("..", vowel) for x in parts]
            #print(parts)
            if all(self.morph.word_is_known(word) for word in parts):
                result.append(choice["id"])
        #print(2,num,result)
        if num == 0:
            if result:
                result = result[0]
            else:
                result = ''
        else:
            result = sorted(result)
        return result

    def load(self, path=""):
        pass

    def save(self, path=""):
        pass

    def fit(self, path=""):
        pass
