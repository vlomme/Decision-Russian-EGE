import random
import re
import time
import joblib
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from solvers.utils import BertEmbedder
from utils import read_config
import pymorphy2

class Solver(BertEmbedder):

    def __init__(self, seed=42, model_config="data/model_26.json"):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.model_config = model_config
        self.config = read_config(self.model_config)
        self.unified_substrings = self.config["unified_substrings"]
        self.replacements = self.config["replacements"]
        self.duplicates = self.config["duplicates"]
        self.classifier = LogisticRegression(verbose=10)
        self.label_encoder = LabelEncoder()
        self.synonyms = open('./data/synmaster.txt', 'r', encoding='utf8').readlines()
        self.synonyms = [re.sub('\.','', t.lower().strip('\n')).split('|') for t in self.synonyms if len(t)>5]
        self.antonyms = open('./data/antonyms.txt', 'r', encoding='utf8').readlines()
        self.antonyms = [re.sub('\.','', t.lower().strip('\n')).split('|') for t in self.antonyms if len(t)>5] 
        self.morph = pymorphy2.MorphAnalyzer()
        self.phraseology = []
        self.razgov = []
        self.razgov2 = []
        self.musor = []
        self.vvodnoe = []
        with open('./data/word.csv', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n","").split('|')
                line[0] = line[0].strip().replace('!','')
                if line[1] == '23':
                    self.phraseology.append(line[0])
                    #self.musor.append(line[0])                      
                if line[1] == '25':              
                    self.razgov.append(line[0])
                if line[1] == '27':              
                    self.razgov2.append(line[0])                    
                    self.razgov.append(line[0]) 
                if line[1] == '99':              
                    self.musor.append(line[0])
                if line[1] == '26' or line[1] == '21':
                    self.vvodnoe.append(line[0])                    
    def init_seed(self):
        return random.seed(self.seed)

    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        used_answers, self.choices = set(), [self.unify_type(choice["text"]) for choice in task["question"]["choices"]]
        for i in range(len(self.choices)):
            self.choices[i] = re.sub('[0-9]\)','',self.choices[i])
            self.choices[i] = re.sub('\(.*?\)','',self.choices[i])
            self.choices[i] = re.sub('\(.*$','',self.choices[i])
        #print(self.choices)
        decisions = dict()
        phrases, phrases_id = self.extract_phrases(task)
        #print(phrases_id)        
        #print(phrases)
        
        for letter in "ABCD":
            if len(phrases[letter]) == 0:
                decisions[letter] = "1"
            elif phrases_id[letter]:
                answer = phrases_id[letter][0]
                answer_id = str(self.choices.index(answer) + 1)
                decisions[letter] = answer_id       
                used_answers.add(answer)        
            else:
                embedding = np.mean(np.vstack(self.sentence_embedding(phrases[letter])), 0)
                proba = self.classifier.predict_proba(embedding.reshape((1, -1)))[0]
                options = list(self.label_encoder.inverse_transform(np.argsort(proba)))[::-1]
                #print(options) 
                try:
                    answer = next(option for option in options if option in self.choices and option not in used_answers)
                    #print(answer) 
                except StopIteration:
                    decisions[letter] = "1"
                    continue
                #used_answers.add(answer)
                answer_id = str(self.choices.index(answer) + 1)
                decisions[letter] = answer_id
        return decisions

    def unify_type(self, type_):
        type_ = re.split(r"\s", type_, 1)[-1]
        #print(type_)
        type_ = type_.strip(" \t\n\v\f\r-–—−()").replace("и ", "")
        #print(type_)
        for key, value in self.unified_substrings.items():
            #print(key,value)
            if key in type_:
                return value
        for key, value in self.replacements.items():
            #print(key,value)
            type_ = re.sub(key + r"\b", value, type_)
        for duplicate_list in self.duplicates:
            #print(duplicate_list)
            if type_ in duplicate_list:
                return duplicate_list[0]
        return type_

    @staticmethod
    def get_sent_num(sent: str):
        match = re.search(r"\(([\dЗбOО]{1,2})\)", sent)
        if match:
            num = match.group(1)
            table = str.maketrans("ЗбОO", "3600")
            num = num.translate(table)
            num = int(num)
            return num
        match = re.search(r"([\dЗбOО]{1,2})\)", sent)
        if match:
            num = match.group(1)
            table = str.maketrans("ЗбОO", "3600")
            num = num.translate(table)
            num = int(num)
            return num

    def extract_phrases(self, task):
        
        choices = task["question"]["choices"]
        result, text = {key: list() for key in "ABCD"}, task["text"]
        result_i = {key: list() for key in "ABCD"}
        text = text.replace("\xa0", " ")
        
        
        """words = re.sub('\([^0-9]*?\)',"", text).replace('\n',' ')
        #print(words)
        sents1 = {} 
        words = re.findall('[0-9 ]{1,3}\).*?\(|[0-9 ]{1,3}\).*?$',words)
        #print(words)
        for word in words:
            if re.search('[0-9 ]{1,3}\)',word):
                z = re.search('[0-9 ]{1,3}\)',word).group(0).replace(')','').replace(' ','')
                sents1[z]=re.sub('[0-9]*?',"",word).replace('(','').replace(')','').replace('–','').replace('…',' ')
        print(sents1)   """     
                 
        
        citations = [sent for sent in sent_tokenize(text.split("Список терминов")[0])
                     if re.search(r"\([А-Г]\)|\(?[А-Г]\)?_{2,}", sent)]
        text = [x for x in re.split(r"[АA]БВГ\.?\s*", text) if x != ""][-1]
        text = re.sub(r"(\([\dЗбOО]{1,2}\))", r" \1 ", text)
        sents = sent_tokenize(text)
        sents = [x.strip() for sent in sents for x in re.split(r"…|\.\.\.", sent)]
        sents = [x.strip() for sent in sents for x in re.split(" (?=\([\dЗбОO])", sent)]
        sents = [sent for sent in sents if re.match(r"\s*\(?[\dЗбОO]{1,2}\)", sent)]
        assert all(re.search(r"\({}\)|\(?{}\)?_{2,}".replace("{}", letter), ' '.join(citations))
                   for letter in "АБВГ"), "Not all letters found in {}".format(citations)
        citations = " ".join(citations)
        citations = re.split("\([А-Г]\)|\(?[А-Г]\)?_{2,}", citations)[1:]
        assert len(citations) == 4, "Expected 4 (not {}) citations: {}".format(len(citations), citations)
        
        #print(sents)
        #solution = task["solution"]["correct"]
        for citation, letter in zip(citations, "ABCD"):
            sent_nums = list()
            matches = re.finditer(r"предложени\w{,3}\s*(\d[\d\-–—− ,]*)", citation)
            for match in matches:
                sent_nums_str = match.group(1)
                for part in re.split(r",\s*", sent_nums_str):
                    part = part.strip(" \t\n\v\f\r-–—−")
                    if len(part) > 0:
                        if part.isdigit():
                            sent_nums.append(int(part))
                        else:
                            from_, to = re.split(r"[-–—−]", part)
                            extension = range(int(from_), int(to) + 1)
                            sent_nums.extend(extension)
            
            
            #print('-'*20,choices[int(solution[letter])-1]["text"],'-'*20)
            #print(sent_nums)
            sents_ = [sent for sent in sents if self.get_sent_num(sent) in sent_nums]
            sents_ = [re.sub(r"(\([\dЗбOО]{1,2}\))\s*", "", sent) for sent in sents_ if not re.sub(r"(\([\dЗбOО]{1,2}\))\s*", "", sent).isspace()]
            
            
            
            
            
            result[letter].extend(sents_)
                        
    
            #print(citation)
            matches = re.finditer("([«\"](.*?)[»\"])([ ]*[-–—−]*[ ]*)([«\"](.*?)[»\"])*", citation)
            if not sents_ and not re.search("([«\"](.*?)[»\"])([ ]*[-–—−]*[ ]*)([«\"](.*?)[»\"])*",citation):
                matches = re.finditer("\(.*?\)", citation)
            mybe_met = False
            mybe_pril = False
            mem_syn = ''
            for match in matches:
                x = match.group(0).replace('−','-').replace('—','-').replace('–','-').replace('«','').replace('»','').replace('"','').replace('-',' - ').replace('  ',' ').strip()
                x = x.replace('(','').replace(')','')
                #print(x)
                if mem_syn and 'синоним' in self.choices:
                    for syn in self.synonyms:
                        if (mem_syn in syn) and (x in syn):
                            result_i[letter].append('синоним')
                            #print('!!2синоним',x)
                            break
                if mem_syn and 'антоним' in self.choices:
                    for syn in self.antonyms:
                        if (mem_syn in syn) and (x in syn):
                            result_i[letter].append('антоним')
                            #print('!!2антоним',x)
                            break                             
                mem_syn = x
                
                result[letter].append(x)

                if x in self.phraseology and 'фразеологизм' in self.choices:
                    #print('!!фразеологизм',x)
                    result_i[letter].append('фразеологизм') 
                if x in self.vvodnoe and 'вводные слова и конструкции' in self.choices:
                    #print('!!вводные слова и конструкции',x)
                    result_i[letter].append('вводные слова и конструкции')
                if x in self.musor and 'жаргонная лексика' in self.choices:
                    #print('!!жаргонная лексика',x)
                    result_i[letter].append('жаргонная лексика')
                if x in self.musor and 'устаревшее слово' in self.choices:
                    #print('!!устаревшее слово',x)
                    result_i[letter].append('устаревшее слово')                    
                if x in self.musor and 'книжная лексика' in self.choices:
                    #print('!!книжная лексика',x)
                    result_i[letter].append('книжная лексика')  
                if x in self.musor and 'профессиональная лексика' in self.choices:
                    #print('!!профессиональная лексика',x)
                    result_i[letter].append('профессиональная лексика')                    
                if x in self.musor and 'заимствованная лексика' in self.choices:
                    #print('!!заимствованная лексика',x)
                    result_i[letter].append('заимствованная лексика') 
                if x in self.musor and 'термин' in self.choices:
                    #print('!!термин',x)
                    result_i[letter].append('термин')                     
                if not result_i[letter] and 'сравнение' in self.choices and ('как ' in x[:10] or 'словно ' in x[:10] or 'будто ' in x[:10] or 'подобно ' in x[:10] or 'точно ' in x[:10]):
                    #print('!!сравнение',x)
                    result_i[letter].append('сравнение')
                   
                if not result_i[letter] and not ' ' in x and x in self.razgov and 'просторечие' in self.choices:
                    #print('!!просторечие',x)
                    result_i[letter].append('просторечие')                
                if ',' in x:
                    x_ = ''.join(x.split(',')[-1:]).strip()
                else:
                    x_ = x
                if ' ' in x_ and len(x_.split())==2:
                    #print('-|',x_,'|-')
                    x_ = x_.split()
                    x0 = self.morph.parse(x_[-2].replace('…',''))
                    x1 = self.morph.parse(x_[-1])[0]
                    if x1.tag.POS == 'NOUN' and 'эпитет' in self.choices:
                        for xxx in x0:
                            if xxx.tag.POS == 'ADJF':
                                #print('!!эпитет',x)
                                result_i[letter].append('эпитет')
                                break
                    elif x0[0].tag.POS == 'ADJF' and x1.tag.POS == 'NOUN' and x1.tag.animacy == 'inan' and 'олицетворение' in self.choices:
                        #print('!!олицетворение',x)
                        result_i[letter].append('олицетворение')

                    elif x1.tag.POS == 'VERB' and x0[0].tag.POS == 'NOUN' and x0[0].tag.animacy == 'inan' and 'олицетворение' in self.choices:
                        #print('!!олицетворение',x)
                        result_i[letter].append('олицетворение')
                    elif x1.tag.POS == 'NOUN' and 'метафора' in self.choices:
                        for xxx in x0:
                            if xxx.tag.POS == 'ADJF':
                                #print('!!метафора',x)
                                result_i[letter].append('метафора')
                                break
                        
                if not ' ' in x and self.morph.parse(x)[0].tag.POS == 'ADJF':
                    mybe_pril = True
                        
                mybe_met = True
                if '-' in x and len(x)<25:
                    if ',' in x:
                        x_ = x.split(',')[0].strip()
                    else:
                        x_ = x
                    x = x_.split('-')
                    x[0] = x[0].strip()
                    x[1] = x[1].strip()
                    x0 = self.morph.parse(x[0])[0].normal_form
                    x1 = self.morph.parse(x[1])[0].normal_form
                    #print(x[0],x[1],'|',x0,x1)
                    if 'синоним' in self.choices:
                        for syn in self.synonyms:
                            if (x[0] in syn or x0 in syn) and (x[1] in syn or x1 in syn):
                                result_i[letter].append('синоним')
                                #print('!!синоним',x)
                                break
                    if 'антоним' in self.choices:
                        for syn in self.antonyms:
                            if (x[0] in syn or x0 in syn) and (x[1] in syn or x1 in syn):
                                result_i[letter].append('антоним')
                                #print('!!антоним',x)
                                break                        
            
            
            
            voskl = True
            vopros = True
            citir = False
            f_vopros = False
            kol_INTJ = 0
            parcelyacia = 0
            first_word = []
            k_first_word = 0
            all_kol_word = 0
            all_kol_korotkih_sent = 0
            kol_sent_not_verb = 0
            kol_fraz = 0
            kol_pros = 0
            kol_vvod = 0
            kol_mus = 0
            kol_syn = 0
            kol_ant = 0
            kol_vopros = 0
            kol_nepolnih = 0
            kol_sravnenia  = 0
            kol_povtor = 0
            kol_obr = 0
            #print('_'*10)
            all_word = {}
            for iii,sent in enumerate(sents_):
                #print(sent)
                if sent:
                    
                    sent = ' '+sent.replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()+' '
                    choice_list = sent.split()

                    sent2 = ([str(self.morph.parse(ok)[0].normal_form) for ok in choice_list])
                    for sss in choice_list[:45]:
                        p_s = self.morph.parse(sss)[0].tag.POS
                        if p_s == 'NOUN' or p_s == 'VERB' or p_s == 'ADJF' or p_s == 'ADVB' or p_s == 'NPRO':
                            if not sss in all_word.keys():
                                all_word[sss] = 1
                            else:
                                all_word[sss] += 1                       
                    sent2 =' '.join(sent2)
                    if 'синоним' in self.choices and not result_i[letter]:
                        est_syn = False
                        for syn in self.synonyms:
                            if (' '+syn[0]+' ' in sent2):
                                mem_syn = syn[0]
                                sent0 = sent2.replace(mem_syn,'')
                                for s in syn[1:]:
                                    if (' '+s+' ' in sent0) and s != mem_syn:
                                        kol_syn +=1
                                        est_syn = True
                                        #print(mem_syn,s)
                                        break
                                if est_syn:
                                    break
                    if 'антоним' in self.choices and not result_i[letter]:
                        est_syn = False
                        for syn in self.antonyms:
                            if (' '+self.morph.parse(syn[0])[0].normal_form+' ' in sent2):
                                mem_syn = self.morph.parse(syn[0])[0].normal_form
                                sent0 = sent2.replace(mem_syn,'')
                                for s in syn[1:]:
                                    if (' '+self.morph.parse(s)[0].normal_form+' ' in sent0) and self.morph.parse(s)[0].normal_form != mem_syn:
                                        kol_ant +=1
                                        est_syn = True
                                        #print(mem_syn,s)
                                        break
                                if est_syn:
                                    break
                    if 'сравнение' in self.choices and ('как ' in sent[1:] or 'словно ' in sent[1:] or 'будто ' in sent[1:] or 'подобно ' in sent[1:] or 'точно ' in sent[1:]):
                        kol_sravnenia +=1
 
                    for vvod in self.vvodnoe:
                        if ' '+vvod+' ' in sent and 'вводные слова и конструкции' in self.choices:
                            kol_vvod +=1
                            break
                    for raz in self.razgov2:
                        if ' '+raz+' ' in sent and 'просторечие' in self.choices:
                            kol_pros +=1
                            break
                    for phras in self.phraseology:
                        if ' '+phras+' ' in sent and 'фразеологизм' in self.choices:
                            kol_fraz +=1
                            break
                    for mus in self.musor:
                        if ' '+mus+' ' in sent and 'жаргонная лексика' in self.choices:
                            kol_mus +=1
                            break                           
                    for mus in self.musor:
                        if ' '+mus+' ' in sent and 'устаревшие слова' in self.choices:
                            kol_mus +=1
                            break                    
                    if not '!' in sent:
                        voskl = False
                    if not '?' in sent:
                        vopros = False
                    if '?' in sent: 
                        if iii ==0:
                            f_vopros = True
                        kol_vopros+=1
                    if ': «' in sent  or ', «' in sent or ', что «' in sent or '", -' in sent or ': "' in sent or '», -' in sent or ', -' in sent or ('«' in sent and not '»' in sent):
                        citir = True
                    
                    all_kol_word += len(choice_list)
                    if len(choice_list) <4:
                        all_kol_korotkih_sent +=1
                    pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
                    est_NOUN_nomn = False
                    for ii in range(len(pos)):
                        if pos[ii]=='NOUN' and self.morph.parse(choice_list[ii])[0].tag.case =='nomn':
                            est_NOUN_nomn = True
                            if 'Name' in self.morph.parse(choice_list[ii])[0].tag and ii+1<len(choice_list) and choice_list[ii+1] == ',':
                                kol_obr +=1
                            break
                        if pos[ii]=='VERB':
                            break
                     
                    if 'INTJ' in pos:
                        kol_INTJ +=1
                    
                    if not est_NOUN_nomn:
                        kol_nepolnih +=1

                    if not 'VERB' in pos and not 'PRTS' in pos:
                        kol_sent_not_verb+=1
                        if est_NOUN_nomn:
                            kol_nepolnih +=1
                    if choice_list[0] == 'и':
                        parcelyacia +=1
                    if len(first_word)<4 and choice_list[0] in first_word or (first_word and first_word[0] in choice_list[:4]):
                        k_first_word +=1
                        #print('++',choice_list[0],first_word)
                    else:
                        first_word.append(choice_list[0])
            if 'повтор' in self.choices:
                for zzx in all_word:
                    if all_word[zzx]>1:
                        #print('!-',zzx)
                        kol_povtor+=1
                        
                     
            if kol_INTJ*2>len(sent_nums) and 'междометие' in self.choices:
                #print('!!междометие')
                result_i[letter].append('междометие')                 

            
            if len(sents_)>0 and vopros and 'риторический вопрос' in self.choices:
                #print('!!риторический вопрос')
                result_i[letter].append('риторический вопрос')            
            if len(sents_)>0 and vopros and 'вопрос' in self.choices:
                #print('!!вопрос')
                result_i[letter].append('вопрос')             
            if citir and 'цитирование' in self.choices:
                #print('!!цитирование')
                result_i[letter].append('цитирование')
                
            if not result_i[letter] and len(sents_)>0 and k_first_word>=len(sents_)/2 and 'анафора' in self.choices:
                #print('!!анафора')
                result_i[letter].append('анафора')
            if len(sent_nums)>7 and 'диалог' in self.choices:
                #print('!!диалог')
                result_i[letter].append('диалог')            
            if kol_vopros*2.4>=len(sent_nums) and not vopros and 'вопросно-ответная форма изложения' in self.choices:
                #print('-!!вопросно-ответная форма изложения',kol_vopros,len(sent_nums))
                result_i[letter].append('вопросно-ответная форма изложения')
            if kol_mus*2>len(sent_nums) and 'устаревшие слова' in self.choices:
                #print('-!!устаревшие слова')
                result_i[letter].append('устаревшие слова')
            if kol_mus*2>len(sent_nums) and 'жаргонная лексика' in self.choices:
                #print('-!!жаргонная лексика')
                result_i[letter].append('жаргонная лексика')             
            if kol_vvod*2>len(sent_nums) and 'вводные слова и конструкции' in self.choices:
                #print('-!!вводные слова и конструкции')
                result_i[letter].append('вводные слова и конструкции')            
            if kol_pros*2>len(sent_nums) and 'просторечие' in self.choices:
                #print('-!!просторечие')
                result_i[letter].append('просторечие')            
            if kol_ant*2>len(sent_nums) and 'антоним' in self.choices:
                #print('-!!антоним')
                result_i[letter].append('антоним')            
            if not result_i[letter] and kol_syn*2>len(sent_nums) and 'синоним' in self.choices:
                #print('-!!синоним')
                result_i[letter].append('синоним')
            if kol_obr*2>len(sent_nums) and 'обращение' in self.choices:
                #print('-!!обращение')
                result_i[letter].append('обращение')                
            if len(sent_nums)>1 and kol_nepolnih==len(sent_nums) and 'неполное предложение' in self.choices:
                #print('-!!неполное предложение')
                result_i[letter].append('неполное предложение')           
            if not result_i[letter] and kol_povtor>=len(sent_nums) and kol_povtor>0 and 'повтор' in self.choices:
                #print('!!повтор')
                result_i[letter].append('повтор')
            if not result_i[letter] and kol_sravnenia*1.6>len(sent_nums) and 'сравнение' in self.choices:
                #print('!!сравнение')
                result_i[letter].append('сравнение')                                
            if not result_i[letter] and kol_fraz*2>len(sent_nums) and 'фразеологизм' in self.choices:
                #print('-!!фразеологизм')
                result_i[letter].append('фразеологизм')             
            if not result_i[letter] and len(sents_)>0 and  voskl and 'риторические восклицания' in self.choices:
                #print('!!риторические восклицания')
                result_i[letter].append('риторические восклицания')                   
            if not result_i[letter] and len(sents_)>0 and  voskl and 'риторическое восклицание' in self.choices:
                #print('!!риторическое восклицание')
                result_i[letter].append('риторическое восклицание') 
            if not result_i[letter] and len(sents_)>0 and voskl and 'восклицание' in self.choices:
                #print('!!восклицание')
                result_i[letter].append('восклицание') 
            if len(sent_nums)>1 and sent_nums[1] == sent_nums[0]+1 and not result_i[letter] and len(sents_)>1 and ((all_kol_korotkih_sent*2>=len(sents_) or kol_sent_not_verb*2>=len(sents_)) and all_kol_word/4<len(sents_) or parcelyacia*2>=len(sents_)) and 'парцелляция' in self.choices:
                
                #print('!!парцелляция',all_kol_korotkih_sent,kol_sent_not_verb,all_kol_word/4,parcelyacia*2,len(sents_))
                result_i[letter].append('парцелляция')
            if mybe_pril and not result_i[letter] and 'Эпитет' in self.choices:
                #print('-!!Эпитет')
                result_i[letter].append('Эпитет')
            if mybe_pril and not result_i[letter] and 'эпитет' in self.choices:
                #print('-!!эпитет')
                result_i[letter].append('эпитет')                
            #if mybe_met and not result_i[letter] and 'метафора' in self.choices:
            #    #print('-!!метафора')
            #    result_i[letter].append('метафора')                
            if not result_i[letter] and f_vopros and not vopros and 'вопросно-ответная форма изложения' in self.choices:
                #print('-!!вопросно-ответная форма изложения',kol_vopros,len(sent_nums))
                result_i[letter].append('вопросно-ответная форма изложения')

        result = {key: list(set(value)) for key, value in result.items()}
        result_i = {key: list(set(value)) for key, value in result_i.items()}
        return result, result_i

    def fit(self, tasks):
        self.corpus, self.types = list(), list()
        for task in tasks:
            letters_to_phrases,_ = self.extract_phrases(task)
            for key in "ABCD":
                questions = letters_to_phrases[key]
                answer_number = task["solution"]["correct"][key]
                answer = next(
                    answ["text"] for answ in task["question"]["choices"] if
                    answ["id"] == answer_number)
                if answer.isdigit():
                    continue
                answer = self.unify_type(answer)
                self.corpus.extend(questions)
                self.types.extend([answer] * len(questions))
        start = time.time()
        print("Encoding sentences with bert...")
        X = np.vstack(self.sentence_embedding(self.corpus))
        print("Encoding finished. This took {} seconds".format(time.time() - start))
        y = self.label_encoder.fit_transform(self.types)
        self.classifier.fit(X, y)

    def load(self, path="data/models/solver26.pkl"):
        model = joblib.load(path)
        self.classifier = model["classifier"]
        self.label_encoder = model["label_encoder"]

    def save(self, path="data/models/solver26.pkl"):
        model = {"classifier": self.classifier,
                 "label_encoder": self.label_encoder}
        joblib.dump(model, path)
