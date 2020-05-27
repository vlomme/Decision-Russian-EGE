import random
import numpy as np
from nltk.tokenize import sent_tokenize
from solvers.utils import Bert
from string import punctuation
import joblib
import re

class Solver(Bert):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.dect_word = {}
        with open('./data/word.csv', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n","").split('|')
                self.dect_word[line[0]] = line[1]      
        

    @staticmethod
    def get_close_sentence(text):
        sentences = sent_tokenize(text)
        #print(sentences)
        if any("<...>" in sent or "<…>" in sent for sent in sentences):
            num = next(num for num, sent in enumerate(sentences) if "<...>" in sent or "<…>" in sent)
            if num == 0:
                return ' '.join(sentences[0:num + 1])
            else:
                return ' '.join(sentences[num-1:num+1])
        else:
            try:
                num = next(num for num, sent in enumerate(sentences) if ("..." in sent or "…" in sent)
                           and not sent.endswith("...") and not sent.endswith("…"))
                if num ==0:
                    return ' '.join(sentences[0:num + 1])
                else:    
                    return ' '.join(sentences[num - 1:num + 1])
            except StopIteration:
                return None


    def predict_from_model(self, task):
        #print('-'*20,task["id"],'-'*20)
        if task.get("text") is None:
            #print('2')
            return "но"
        close = self.get_close_sentence(task["text"])
        #print(close)
        if close is None:
            return self.default_word
        words = close.replace('<....>',' [MASK] ')
        words = words.replace('<...>',' [MASK] ')
        words = words.replace('<…>',' [MASK] ')
        words = words.replace('( ..... )',' [MASK] ')
        #print(words)
        if not '[MASK]' in words:
            words = words.replace('…',' [MASK] ')
            #print(words)
        if not '[MASK]' in words:
            words = words.replace('...',' [MASK] ')        
        
        if not '[MASK]' in words:
            words = re.search('.{0,100}\(\.\.\.\).{0,100}',words).group(0).replace('(...)','[MASK]')
        words = words.replace('  ',' ')
        #print(task["text"])
        #print('-'*20)
        z = []
        my_dect_word = []
        type_a = 0
        
        
        if 'подберите сочетание' in task["text"]:
            #words  = words.replace('[MASK]','[MASK] [MASK]')
            if 'производного предлога с указательным местоимением' in task["text"]:
                if 'но [' in words.lower():
                    result = 'несмотрянаэто' 
                else:
                    result = 'вследствиеэтого'
            elif 'частицы с наречием' in task["text"]:
                result = 'именнопоэтому'    
            elif 'подчинительного союза и определительного местоимения' in task["text"]:
                result = 'каклюбое'  
            elif 'числительного с предлогом' in task["text"]:
                result = 'одиниз' 
            elif 'предлога с относительным местоимением' in task["text"]:
                result = 'изкоторых'
            elif 'частицы с указательным местоимением' in task["text"]:
                result = 'именноэта'    
           
            elif 'предлога со словом' in task["text"]:
                result = 'вслучае'             
            elif '] ,' in words:
                result = 'деловтом' 
            elif '] полагают' in words:
                result = 'другиеже' 
            elif 'сочетание частицы с местоименным наречием' in task["text"]:
                result = 'именнопоэтому'             
            else:
                result = 'деловтомчто'
            type_a = 2
        if 'наречие' in task["text"]:
            z.append(1)

        if 'ограничительно-выделительную частицу' in task["text"]:
            z.append(2)      
        elif 'частиц' in task["text"]:
            z.extend([2,3,4,5,6,7,8,9,10,11])
        if 'союзное слово' in task["text"]:
            z.extend([12,28,13,14])
        if 'сочинительный союз' in task["text"]:
            z.extend([15,16,17])
        if 'подчинительный составной союз' in task["text"]:
            result = 'потомучто'
            type_a = 2
        if 'фразеологическое словосочетание' in task["text"]:
            result = 'вконцеконцов'
            type_a = 2
        if 'подчинительный составной союз' in task["text"]:
            result = 'потомучто'
            type_a = 2            
        elif 'составной союз' in task["text"]:
            result = 'вместестем'
            type_a = 2             
        if 'подберите глагол' in task["text"]:
            result = 'оказалось'
            type_a = 2        
        if 'пояснительный союз' in task["text"]:
            result = 'тоесть'
            type_a = 2           
        if 'сочетание частицы со сложным предлогом' in task["text"]:
            result = 'именноизза'
            type_a = 2              
        if 'производный составной предлог' in task["text"]:
            result = 'вотличиеот'
            type_a = 2         
        elif 'подчинительный союз' in task["text"]:
            z.append(14)
        elif 'противительный союз' in task["text"]:
            z.append(16)
        elif 'союз' in task["text"]:
            z.extend([14,15,16,17])
        if 'предлог' in task["text"]:
            z.append(19)
        if 'указательное местоимение' in task["text"] or 'указательным местоимением' in task["text"]:
            z.append(18)
        elif 'относительное местоимение' in task["text"]:
            z.append(12)
        elif 'определительное местоимение' in task["text"]:
            z.append(20)
        elif 'личное местоимение' in task["text"]:
            z.append(24)            
        elif 'местоимение' in task["text"]:
            z.extend([12,18,20])
        if 'из приведённых ниже слов' in task["text"] or 'слово или сочетание слов' in task["text"]:
            type_a = 1
            words2 = re.search('\?.*?1',task["text"]).group(0)
            words2 = re.sub('.*\.','',words2)
            words2 = words2.replace('?','').replace('1','').replace('(','')
            words2 = re.findall('[А-Я][^A-Я]*',words2)
            for i in range(len(words2)):
                words2[i] = words2[i].replace(',','').replace('.','').strip().lower()
            if not words2:
                words2 = re.search('1[^)]*$',task["text"]).group(0)
                words2 = re.findall('[0-9][^0-9]*',words2)
                for i in range(len(words2)):
                    words2[i] = re.sub('[0-9]','', words2[i]).replace(',','').replace('.','').strip().lower()              
            my_dect_word = words2
            #print ('из приведённых ниже слов',words2)             
        if 'вводное словосочетание' in task["text"]:
            z.append(26)

        elif 'вводное слово' in task["text"]:
            z.append(21)            
        if 'вводную конструкцию' in task["text"]:        
            z.append(26)
            my_dect_word.append('кроме этого')
        if not z and not my_dect_word:
            z = [i for i in range(21)]
        for key in self.dect_word.keys():
            #print(self.dect_word[key])
            if int(self.dect_word[key]) in z:
                if not 'вводную конструкцию' in task["text"] or key.strip() != 'таким образом':
                    my_dect_word.append(key.strip())
            

        words = words.replace('(','')
        words = words.replace(')','')
        words = re.sub("[0-9]","",words).replace(","," ,").replace("."," .").replace(":"," :").replace("  "," ")
        
        if type_a!=2:
            #print (words)
            #print (my_dect_word)
            result,search1 = self.what_mask2(words,my_dect_word,type_a)
            
            if not search1:
                words = words.replace('[MASK]','[MASK] [MASK]')
                result2,search2 = self.what_mask2(words,my_dect_word,4)
            if not search1 and search2:
                result = result2
            #print(result)
        return result.strip(punctuation)
