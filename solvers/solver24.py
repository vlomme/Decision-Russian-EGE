import os
import random
import re
import pymorphy2
from nltk.tokenize import ToktokTokenizer


class Solver(object):

    def __init__(self, seed=42, data_path = 'data/'):
        self.is_train_task = False
        self.morph = pymorphy2.MorphAnalyzer()
        self.toktok = ToktokTokenizer()
        self.seed = seed
        self.init_seed()
        #self.synonyms = open(os.path.join(data_path, r'synonyms.txt'), 'r', encoding='utf8').readlines()
        #self.synonyms = [re.sub('\.','', t.lower().strip('\n')).split(' ') for t in self.synonyms]
        #self.synonyms = [[t for t in l if t]  for l in self.synonyms]
        self.synonyms = open('./data/synmaster.txt', 'r', encoding='utf8').readlines()
        self.synonyms = [re.sub('\.','', t.lower().strip('\n')).split('|') for t in self.synonyms if len(t)>5]
        self.antonyms = open('./data/antonyms.txt', 'r', encoding='utf8').readlines()
        self.antonyms = [re.sub('\.','', t.lower().strip('\n')).split('|') for t in self.antonyms if len(t)>5]        
        #self.antonyms = open(os.path.join(data_path, r'antonyms.txt'), 'r', encoding='utf8').readlines()
        #self.antonyms = [t.strip(' \n').split(' - ') for t in self.antonyms]
        self.phraseology = []
        self.razgov = []
        self.musor = []
        with open('./data/word.csv', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n","").split('|')
                if line[1] == '23':
                    self.phraseology.append(line[0])
                    #self.musor.append(line[0])                      
                if line[1] == '25':              
                    self.razgov.append(line[0])
                    #self.musor.append(line[0])  
                if line[1] == '99':              
                    self.musor.append(line[0])                    
        #self.phraseology = open(os.path.join(data_path, r'phraseologs.txt'), 'r', encoding='utf8').readlines()
        #self.phraseology = [[l for l in self.lemmatize(l) if l not in ['\n', ' ','...', '' ,',', '-', '.', '?',r' (', r'/']] for l in self.phraseology]

    def init_seed(self):
        random.seed(self.seed)

    def lemmatize(self, text):
        return [self.morph.parse(word)[0].normal_form for word in
                self.toktok.tokenize(text.strip())]

    def predict(self, task):
        return self.predict_from_model(task)

    def get_word(self, text):
        try:
            return re.split('»', re.split('«', text)[1])[0]
        except:
            return ''

    def get_pos(self, text):
        pos = []
        text = text.replace('.',' .')
        lemmas = self.lemmatize(text)
        lemmas = [l for l in lemmas if l!=' ']
        #print('!'*10,lemmas)
        if 'фразеологизм' in lemmas:
            pos = "PHR"
        elif 'синоним к слову' in text:
            pos = "SYN2"
            self.mem_word = text.split('синоним к слову')[1]
            self.mem_word = re.sub('».*$|\..*$','',self.mem_word)[:60]
        elif 'синоним (один) к слову' in text:
            pos = "SYN2"
            self.mem_word = text.split('синоним (один) к слову')[1]
            self.mem_word = re.sub('».*$|\..*$','',self.mem_word)[:60]        
        elif 'со значением' in text:
            pos = "SYN2"
            self.mem_word = text.split('со значением')[1]            
            self.mem_word = re.sub('».*$|\..*$','',self.mem_word)[:60]
        elif 'имеющее значение' in text:
            pos = "SYN2"
            self.mem_word = text.split('имеющее значение')[1]            
            self.mem_word = re.sub('».*$|\..*$','',self.mem_word)[:60]          
           
        elif 'синоним' in lemmas:
            pos = "SYN"
        elif 'антоним' in lemmas:
            pos = "ANT"
        elif 'антонимический' in lemmas:
            pos = "ANT"

        elif 'синонимический' in lemmas:
            pos = "SYN"
        elif 'разговорный' in lemmas:
            pos = "RAZ"        
        elif 'переносный' in lemmas:
            pos = "DEF" 
        elif 'Выпишите слова' in text:
            pos = "DEF"
            #print('Выпишите слова')        
        else:
            pos = "DEF"
        return pos


    def sent_split(self, text):
        reg = r'\(*\n*\d+\n*\)'
        return re.split(reg, text)
        
    def search3(self, text_lemmas, lst):
        text_lemmas = text_lemmas.split()
        text_lemmas2 = ([self.morph.parse(word)[0].normal_form for word in text_lemmas])
        #text_lemmas2 = ''.join(text_lemmas2)
        #print(text_lemmas2)
        for l in lst:
            #for syn1 in l:
            mem = self.morph.parse(l[0])[0].normal_form
            if mem in text_lemmas2:
                norm_mem = text_lemmas[text_lemmas2.index(mem)]
                mem_id = text_lemmas2.index(mem)
                #print(norm_mem,l)
                for syn in l:
                    syn = self.morph.parse(syn)[0].normal_form
                    if syn and syn.replace('ё','е') != mem.replace('ё','е') and (syn in text_lemmas2 or syn.replace('е','ё') in text_lemmas2):
                        norm_syn = text_lemmas[text_lemmas2.index(syn)]
                        syn_id = text_lemmas2.index(syn)
                        if syn_id < mem_id:
                            return norm_syn+norm_mem
                        else:
                            return norm_mem+norm_syn                           
                        #return norm_mem+norm_syn
                    """if syn and syn != mem and (' '+syn+' ' in text_lemmas or ' '+syn+' ' in text_lemmas.replace('ё','е')):
                        print(syn, l)
                        return mem+syn"""
        return ''
    def search4(self, text_lemmas, lst):
        self.mem_word = self.mem_word.replace("«","").replace("»","").replace(".","").strip().lower()
        self.mem_word2 = re.split('[\,\ ]',self.mem_word)
        #print(self.mem_word2)
        mem_words = []
        for l in lst:
            for mem_w in self.mem_word2:
                if mem_w and mem_w in l:
                    for syn in l:
                        if not syn in mem_words:
                            mem_words.append(syn)
        #print(mem_words)
        text_lemmas = text_lemmas.replace(","," ,").replace("."," .").replace(":"," :").replace("!"," !").replace("?"," ?").replace("\n"," ").replace("  "," ").replace("\t"," ").replace("«","").replace("»","").lower()
        text_lemmas = text_lemmas.split()
        text_lemmas2 = ([self.morph.parse(word)[0].normal_form for word in text_lemmas])        
        text_lemmas2.extend(text_lemmas)
        #print(text_lemmas2)
        for syn in mem_words:
            if syn and (syn in text_lemmas2 or syn.replace('е','ё') in text_lemmas2):
                return syn
        return ''

    def search1(self, text_lemmas, lst):
        text_lemmas = text_lemmas.split()
        text_lemmas2 = ([self.morph.parse(word)[0].normal_form for word in text_lemmas])        
        text_lemmas.extend(text_lemmas2)
        #print(text_lemmas)
        #print(lst)
        for l in lst:
            k=0
            mem =''
            for syn in l:
                #syn = self.morph.parse(syn)[0].normal_form
                if syn and syn.replace('ё','е') != mem.replace('ё','е') and (syn in text_lemmas or syn.replace('е','ё') in text_lemmas):
                    #norm_syn = text_lemmas[text_lemmas2.index(syn)]
                    k+=1
                    if syn in text_lemmas:
                        syn_id = text_lemmas.index(syn)
                    elif syn.replace('е','ё') in text_lemmas:
                        syn_id = text_lemmas.index(syn.replace('е','ё'))
                    else:
                        syn_id = -1
                    #print(k, syn, l)
                    """if k>1 and syn_id != mem_id+len(text_lemmas2) and syn_id+len(text_lemmas2)  != mem_id:
                        
                        print(syn_id, mem_id, len(text_lemmas2))
                        if syn_id> len(text_lemmas2):
                            syn_id = syn_id - len(text_lemmas2)
                        if mem_id> len(text_lemmas2):
                            mem_id = mem_id - len(text_lemmas2) """
                    if (k>1 and (syn_id<len(text_lemmas)/2 and mem_id<len(text_lemmas)/2 or syn_id>=len(text_lemmas)/2 and mem_id>=len(text_lemmas)/2)):
                                               
                        if syn_id < mem_id:
                            return syn+mem
                        else:
                            return mem+syn
                    mem = syn
                    #norm_mem = norm_syn
                    mem_id = syn_id
                    
        return ''
    def search2(self, text_lemmas, lst):
        for l in lst:
            if ' '+l+' ' in text_lemmas or ' '+l+' ' in text_lemmas.replace('ё','е'):
                l = l.replace(' мне ','').replace(' свои ','').replace(' нам ','').replace(' себе ','')
                return l.replace(' ','')
        """text_lemmas = text_lemmas.split()
        text_lemmas2 = ([self.morph.parse(word)[0].normal_form for word in text_lemmas])        
        text_lemmas.extend(text_lemmas2)        
        #print(text_lemmas)
        #print(lst)
        for l0 in lst:
            bad = False
            l1 = l0.split()
            l2 = ([self.morph.parse(word)[0].normal_form for word in l1])        
            #print(l1)
            #print(l,' '+l+' ' in text_lemmas, l+' ' in text_lemmas,l in text_lemmas)
            for i,l in enumerate(l1):
                #print(l1[i],l1[i] in text_lemmas)
                if not l1[i] in text_lemmas and not l2[i] in text_lemmas:
                    bad = True
            if not bad:
                l = l0.replace(' мне ','').replace(' свои ','').replace(' нам ','')
                return l.replace(' ','')"""
        return ''

    def get_num(self, text):
        nums = 0
        text = text[3:]
        text = re.sub('\(.*[\–\-\—\−].*\)','',text)
        res = re.search('\d+[\–\-\—\−]\d+', text)
        #print(text)
        if res:
            res = res[0]
            if '–' in res:
                nums = res.split('–')
                nums = list(range(int(nums[0]), int(nums[1])+1))
            elif '-' in res:
                nums = res.split('-')
                nums = list(range(int(nums[0]), int(nums[1])+1))
            elif '—' in res:
                nums = res.split('—')
                nums = list(range(int(nums[0]), int(nums[1])+1))                
            elif '−' in res:
                nums = res.split('−')
                nums = list(range(int(nums[0]), int(nums[1])+1))            
            else:
                nums = [int(res)]
        else:
            res = re.search('\d+', text)
            nums = [int(res[0])]
        return nums

    def compare_text_with_variants(self,pos, text, nums=[], word=''):
        """indexes = []
        sents = self.sent_split(text)
        print(sents[s-1])
        lemmas_all = []
        for s in nums:
            
            lemmas = self.lemmatize(sents[s-1])
            lemmas_all += [l for l in lemmas if l!=' ']
            conditions=0
        lemmas_all = [l for l in lemmas_all if re.match('\w+', l) and re.match('\w+', l)[0]==l]
        
        for s in nums:
            lemmas_all = lemmas_all+sents[s-1]"""        
        lemmas_all =' '
        words = re.sub('\([^0-9]*?\)',"", text).replace('\n',' ')
        #print(words)
        sents = {} 
        words = re.findall('[0-9 ]{1,3}\).*?\(|[0-9 ]{1,3}\).*?$',words)
        #print(words)
        for word in words:
            if re.search('[0-9 ]{1,3}\)',word):
                z = re.search('[0-9 ]{1,3}\)',word).group(0).replace(')','').replace(' ','')
                sents[z]=re.sub('[0-9]*?',"",word).replace('(','').replace(')','').replace('–','').replace('…',' ')
        #print(sents)
        if not sents.get(str(nums[0])):
            nums.append(nums[0]-1)
        for j in sents:
            if int(j) in nums:        
                lemmas_all = lemmas_all+sents[j]
        lemmas_all = lemmas_all +' '
        #print(lemmas_all)
        lemmas_all = lemmas_all.replace(","," ,").replace("."," . ").replace(":"," :").replace("!"," !").replace("?"," ?").replace("\n"," ").replace("  "," ").replace("\t"," ").replace("«","").replace("»","").replace(";","").lower()
        
        
        
        
        #print(lemmas_all)
        if pos=='SYN':
            variant = self.search1(lemmas_all, self.synonyms)
        elif pos=='SYN2':
            variant = self.search4(lemmas_all, self.synonyms)
        elif pos=='ANT':
            variant = self.search3(lemmas_all, self.antonyms)
        elif pos=='RAZ':
            variant = self.search2(lemmas_all, self.razgov)        
        elif pos=='PHR':
            variant = self.search2(lemmas_all, self.phraseology)          
        else:
            #print(self.musor)
            variant = self.search2(lemmas_all, self.musor)
        if variant:
            return variant
        else:
            return str(random.choice(lemmas_all))

    def eat_json(self, task):
        
        
        try:
            firstphrase, tasktext = re.split(r'\(\n*1\n*\)', task['text'])
        except ValueError:
            firstphrase, tasktext = ' '.join(re.split(r'\(\n*1\n*\)', task['text'])[:-1]),re.split(r'\(\n*1\n*\)', task['text'])[-1]
        if 'Из предложени' in tasktext:
            text, task = re.split('Из предложени', tasktext)
            task = 'Из предложени '+task
        else:
            text, task = tasktext, firstphrase
        nums = self.get_num(task)
        pos = self.get_pos(task)
        word = ''
        if pos=='DEF':
            word = self.get_word(task)
        return text, task, pos, nums, word


    def fit(self, tasks):
        pass

    def load(self, path='data/models/solver24.pkl'):
        pass

    def save(self, path='data/models/solver24.pkl'):
        pass

    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        text, task1, pos, nums, word = self.eat_json(task)
        text = re.sub('\*.*$','',task['text']).replace('(З','(3').replace('(3З)','(33)')
        #print(text)
        #print(task1)
        #print(pos)
        #print(nums)
        #print(word)
        result = self.compare_text_with_variants(pos, text, nums=nums, word=word)
        return result
