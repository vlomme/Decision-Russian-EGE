from ufal.udpipe import Model, Pipeline
from difflib import get_close_matches
from string import punctuation
import pickle
import pymorphy2
import re
import random
from sys import getdefaultencoding
from solvers.utils import Bert

class Solver(Bert):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.morph = pymorphy2.MorphAnalyzer()
        self.dect_paronim = {}
        with open('./data/paronim.csv', encoding='utf-8') as f:
            for i, lines in enumerate(f):
                lines = lines.replace("\n","").lower().split('|')
                #morph.parse('думающему')[0].normal_form
                self.dect_paronim[lines[0]] = lines[1]
                self.dect_paronim[lines[1]] = lines[0]
        self.most_popular = []
        with open('./data/most_popular_paronim.csv', encoding='utf-8') as f:
            for line in f:
                self.most_popular.append(line.replace("\n",""))

    def load(self, path="data/models/solver5.pkl"):
        pass

    def predict_from_model(self, task):
        #print('-'*20,task['id'],'-'*20)
        if 'Запишите подобранное слово.'in task['text']:
            words = re.sub('.*Запишите подобранное слово.','',task['text']).replace("\n"," ").replace(","," , ").replace("."," . ").replace(":"," : ").replace("!"," ! ").replace("  "," ")
        elif 'пароним'in task['text']:
            words = re.sub('.*пароним','',task['text']).replace("\n"," ").replace(","," , ").replace("."," . ").replace(":"," : ").replace("!"," ! ").replace("  "," ")
        else:
            words = task['text'].replace("\n"," ").replace(","," , ").replace("."," . ").replace(":"," : ").replace("!"," ! ").replace("  "," ")
                    
        del_word = []
        new_word = []
        word = words.split()
        for i, x in enumerate(word):
            #print(x)
            if (x.isupper() and len(x)>1 and x!='НЕВЕРНО'):
                del_word.append(word[i].lower())
                word[i] = '[MASK]'                
        words = ' '.join(word+word)
        #print(del_word)
        for word in del_word:
            p = self.morph.parse(word)[0]
            word = p.normal_form
            #print(word)
            word2 = self.dect_paronim.get(word)
            if not word2:
                word2 = self.dect_paronim.get(word+'_')
            if word2:
                word2 = self.morph.parse(word2)[0]
                if word2:
                    if p.tag.POS and word2.inflect({p.tag.POS}) is not None:
                        word2 = word2.inflect({p.tag.POS})
                    if p.tag.animacy and word2.inflect({p.tag.animacy}) is not None:
                        word2 = word2.inflect({p.tag.animacy})
                    if p.tag.aspect and word2.inflect({p.tag.aspect}) is not None:
                        word2 = word2.inflect({p.tag.aspect})
                    if p.tag.case and word2.inflect({p.tag.case}) is not None:
                        word2 = word2.inflect({p.tag.case})
                    if p.tag.gender and word2.inflect({p.tag.gender}) is not None:
                        word2 = word2.inflect({p.tag.gender})
                    if p.tag.involvement and word2.inflect({p.tag.involvement}) is not None:
                        word2 = word2.inflect({p.tag.involvement})
                    if p.tag.mood and word2.inflect({p.tag.mood}) is not None:
                        word2 = word2.inflect({p.tag.mood})
                    if p.tag.number and word2.inflect({p.tag.number}) is not None:
                        word2 = word2.inflect({p.tag.number})
                    if p.tag.person and word2.inflect({p.tag.person}) is not None:
                        word2 = word2.inflect({p.tag.person})
                    if p.tag.tense and word2.inflect({p.tag.tense}) is not None:
                        word2 = word2.inflect({p.tag.tense})
                    if p.tag.transitivity and word2.inflect({p.tag.transitivity}) is not None:
                        word2 = word2.inflect({p.tag.transitivity})
                    if p.tag.voice and word2.inflect({p.tag.voice}) is not None:
                        word2 = word2.inflect({p.tag.voice})
                    word2 = word2.word
            if word2:
                new_word.append(word2)
            else:
                new_word.append('11111')    
        new_word.extend(del_word)
        #words = ' '.join(words+words)
        #print(words)
        #print(new_word)
        result = ""
        outputs = self.what_mask5(words+words,new_word)
        #print(outputs)
        if len(outputs)>1:
            for outpu in outputs:
                if outpu and outpu in self.most_popular:
                    result = outpu
                    break
            if not result and outputs:
                result = outputs[0]
        elif outputs:
            result = outputs[0]
        return result
        