from ufal.udpipe import Model, Pipeline
from difflib import SequenceMatcher
from string import punctuation
import pymorphy2
import random
import re
import sys
from sys import getdefaultencoding

def get_gerund(features):
    """деепричастие """
    hypothesys = []

    for feature in features:
        for row in feature:
            if row[4] == "VERB":
                if "VerbForm=Conv" in row[5]:
                    hypothesys.append(" ".join([row[2] for row in feature]))

    return hypothesys

def get_indirect_speech(features):
    """ косвенная речь """
    morph = pymorphy2.MorphAnalyzer()
    hypothesys = []
    mybe = []
    mybe2 = []
    mybe3 = []    
    pred = []
    for i,feature in enumerate(features):
        bad_sen,bad_sen1,bad_sen2,bad_sen3,bad_sen4 = False,False,False,False,False
        chto = False
        est_dash = False
        est_glag = 0
        id_g = 0
        for j,row in enumerate(feature):
            #print(row[2])
            nf = morph.parse(row[2].replace(',',''))[0].normal_form
            if est_glag>0 and row[4] == "VERB"  and (morph.parse(row[2])[0].tag.person =='2per' or morph.parse(row[2])[0].tag.person =='1per'):
                bad_sen1 = True
                #print('2per',row[2])            
            if (row[8] == '1' or nf =='спросить' or nf =='пообещать' or nf =='спросить' or nf =='завещать' or  nf =='рассказывать'or 
                nf =='вспоминать' or nf =='писать' or nf =='написать' or nf =='заметить' or nf =='утверждать'or nf =='объявить' or 
                nf =='подтвердить' or nf =='произносить' or nf =='уточнить' or nf =='ответить'):
                #print(row[2])
                #hypothesys.append(" ".join([row[2] for row in feature]))
                id_g = j
                bad_sen = True
                est_glag = 4 
            if ',' in row[2]:
                est_glag -=1
            if '»' in row[2]:
                est_glag =0            
            if '–' in row[2]:
                est_dash = True            
            if row[2] =='что' and id_g+6>j and est_glag>0:
                chto = True            
            if '«' in row[2] and id_g+6>j and est_glag>0 and chto:
                #print('«')
                bad_sen3 = True

            if morph.parse(row[2])[0].tag.POS =='NPRO' and est_glag>0 and est_glag!=4 and (morph.parse(row[2])[0].tag.person =='1per' or morph.parse(row[2])[0].tag.person =='2per'):
                bad_sen2 = True
                #print('NPRO',row[2])                
            if morph.parse(row[2])[0].tag.POS =='NPRO' and (morph.parse(row[2])[0].tag.person =='3per'):
                est_glag =0
            if row[2] =='моей' and est_glag>0 and est_glag!=4:
                bad_sen2 = True
                #print('моей',row[2])                 
            if nf =='мнение':
                bad_sen4 = True
            if (row[4] == "VERB"  and morph.parse(row[2])[0].tag.person =='2per'):
                bad_sen4 = True
                #print('!!2per',row[2])
            if morph.parse(row[2])[0].tag.POS =='NPRO' and (morph.parse(row[2])[0].tag.person =='1per' or morph.parse(row[2])[0].tag.person =='2per'):
                bad_sen4 = True
                #print('!!NPRO',row[2])                
            if row[2] =='моей':
                bad_sen4 = True
                #print('!!моей',row[2])                 

        if bad_sen1 or (bad_sen3 and bad_sen2):
            hypothesys.append(str(i+1))
        if bad_sen2:    
            mybe.append(str(i+1))
        if bad_sen and not est_dash:    
            mybe2.append(str(i+1))
        if bad_sen4:
            mybe3.append(str(i+1)) 
    #print(hypothesys,mybe,mybe2,mybe3)
    if not hypothesys:
        hypothesys = mybe
    if not hypothesys:
        hypothesys = mybe2
    if not hypothesys:
        hypothesys = mybe3
    return hypothesys

def get_app(features):
    """ Приложение """
    hypothesys = []
    for i,feature in enumerate(features):
        for row1, row2, row3 in zip(feature, feature[1:], feature[2:]):
            if row1[2] == "«" and row3[2] == "»" and row2[1] == '1':
                hypothesys.append(i+1)
                #hypothesys.append(" ".join([row[2] for row in feature]))
            if "«" in row1[2]:
                if row1[2][1:][0].isupper():
                    hypothesys.append(str(i+1))
                    #hypothesys.append(" ".join([row[2] for row in feature]))
    return hypothesys

def get_predicates(features):
    """ связь подлежащее сказуемое root + subj = number """
    #hypothesys = set()
    hypothesys = []
    for i,feature in enumerate(features):
        head, number = None, None
        for row in feature:
            if row[7] == 'root':
                head = row[0]
                for s in row[5].split('|'):
                    if "Number" in s:
                        number = s.replace("Number=", "")
        for row in feature:
            row_number = None
            for s in row[5].split('|'):
                if "Number" in s:
                    row_number = s.replace("Number=", "")
            if row[0] == head and number != row_number:
                hypothesys.append(str(i+1))
                #hypothesys.add(" ".join([row[2] for row in feature]))
    return hypothesys

def get_clause(features):
    """ сложные предложения """
    #hypothesys = set()
    hypothesys = []
    for i,feature in enumerate(features):
        for row in feature:
            if row[3] == 'который':
                #hypothesys.add(" ".join([row[2] for row in feature]))
                hypothesys.append(str(i+1))
    return hypothesys


def get_participle(features):
    """причастие """
    hypothesys = []
    for i,feature in enumerate(features):
        for row in feature:
            if row[4] == "VERB":
                if "VerbForm=Part" in row[5]:
                    hypothesys.append(str(i+1))
                    #hypothesys.append(" ".join([row[2] for row in feature]))
    return hypothesys

def get_verbs(features):
    """ вид и время глаголов """
    #hypothesys = set()
    hypothesys = []
    for i,feature in enumerate(features):
        head, aspect, tense = None, None, None
        for row in feature:
            if row[7] == 'root':
                # head = row[0]
                for s in row[5].split('|'):
                    if "Aspect" in s:
                        aspect = s.replace("Aspect=", "")
                    if "Tense" in s:
                        tense = s.replace("Tense=", "")

        for row in feature:
            row_aspect, row_tense = None, None
            for s in row[5].split('|'):
                if "Aspect" in s:
                    row_aspect = s.replace("Aspect=", "")
            for s in row[5].split('|'):
                if "Tense" in s:
                    row_tense = s.replace("Tense=", "")
            if row[4] == "VERB" and row_aspect != aspect: # head ?
                hypothesys.append(str(i+1))
                #hypothesys.add(" ".join([row[2] for row in feature]))

            if row[4] == "VERB" and row_tense != tense:
                hypothesys.append(str(i+1))
                #hypothesys.add(" ".join([row[2] for row in feature]))
    return hypothesys

def get_nouns(features):
    """ формы существительных ADP + NOUN"""
    #hypothesys = set()
    hypothesys = []
    apds = ["благодаря", "согласно", "вопреки", "подобно", "наперекор",
            "наперерез", "ввиду", "вместе", "наряду", "по"]
    for i,feature in enumerate(features):
        for row1, row2 in zip(feature, feature[1:]):
            if row1[3] in apds:
                if row2[4] == 'NOUN':
                    hypothesys.append(str(i+1))
                    #hypothesys.add(" ".join([row[2] for row in feature]))
    return hypothesys

def get_numerals(features):
    hypothesys = []
    for i,feature in enumerate(features):
            for row in feature:
                if row[4] == "NUM":
                    hypothesys.append(str(i+1))
                    #hypothesys.append(" ".join([row[2] for row in feature]))
    return hypothesys


def get_homogeneous(features):
    #hypothesys = set()
    #1) не только ... но и (а и; но даже; а еще; а к тому же); не только не ... но (но скорее, скорее; напротив, наоборот); а не только; 2) не то что ... но (а; просто; даже, даже не); даже ... не то что; даже не ... не то что; даже не ... тем более не;
    #3) мало того ... еще и; мало того что ... еще и; мало того; более того, больше того; хуже того; а то и.    
    hypothesys = []
    for i,feature in enumerate(features):
        sent = " ".join([token[2] for token in feature]).lower()
        for double_conj in ["не только","но и", "если не", "не сколько", "не столько", "не то чтобы"]:
            if double_conj in sent:
                hypothesys.append(str(i+1))
    return hypothesys


class Solver():

    def __init__(self, seed=42):
        self.morph = pymorphy2.MorphAnalyzer()
        self.categories = set()
        self.has_model = True
        self.model = Model.load("data/udpipe_syntagrus.model")
        self.process_pipeline = Pipeline(self.model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        #self.model = Model.load("data/udpipe_syntagrus.model".encode())
        #self.process_pipeline = Pipeline(self.model, 'tokenize'.encode(), Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu'.encode())            
        self.seed = seed
        self.label_dict = {
            'деепричастный оборот': "get_gerund",
            'косвенный речь': "get_indirect_speech",
            'несогласованный приложение': "get_app",
            'однородный член': "get_homogeneous",
            'причастный оборот': "get_participle",
            'связь подлежащее сказуемое': "get_predicates",
            'сложноподчинённый': "get_clause",
            'сложный': "get_clause",
            'соотнесённость глагольный форма': "get_verbs",
            'форма существительное': "get_nouns",
            'числительное': "get_numerals"
        }
        self.init_seed()

    def init_seed(self):
        return random.seed(self.seed)

    def get_syntax(self, text):
        processed = self.process_pipeline.process(text)
        content = [l for l in processed.split('\n') if not l.startswith('#')]
        #processed = self.process_pipeline.process(text.encode())
        #content = [l for l in processed.decode().split('\n') if not l.startswith('#')]        
        tagged = [w.split('\t') for w in content if w]
        return tagged

    def tokens_features(self, some_sent):

        tagged = self.get_syntax(some_sent)
        features = []
        for token in tagged:
            _id, token, lemma, pos, _, grammar, head, synt, _, _, = token #tagged[n]
            capital, say = "0", "0"
            if lemma[0].isupper():
                    capital = "1"
            if lemma in ["сказать", "рассказать", "спросить", "говорить"]:
                    say = "1"
            feature_string = [_id, capital, token, lemma, pos, grammar, head, synt, say]
            features.append(feature_string)
        return features

    def normalize_category(self, cond):
        """ {'id': 'A', 'text': 'ошибка в построении сложного предложения'} """
        condition = cond["text"].lower().strip(punctuation)
        condition = re.sub("[a-дabв]\)\s", "", condition).replace('членами.', "член")
        norm_cat = ""
        for token in condition.split():
            lemma = self.morph.parse(token)[0].normal_form
            if lemma not in [
                    "неправильный", "построение", "предложение", "с", "ошибка", "имя",
                    "видовременной", "видо-временной", "предложно-падежный", "падежный",
                    "неверный", "выбор", "между", "нарушение", "в", "и", "употребление",
                    "предлог", "видовременный", "временной"
                ]:
                norm_cat += lemma + ' '
        self.categories.add(norm_cat[:-1])
        return norm_cat

    def parse_task(self, task):

        assert task["question"]["type"] == "matching"

        conditions = task["question"]["left"]
        choices = task["question"]["choices"]

        good_conditions = []
        X = []
        for cond in conditions:  # LEFT
            good_conditions.append(self.normalize_category(cond))
                    
        for choice in choices:
            choice = re.sub("[0-9]\\s?\)", "", choice["text"])
            X.append(choice)
        return X, choices, good_conditions

    def match_choices(self, label2hypothesys, choices):
        final_pred_dict = {}
        for key, value in label2hypothesys.items():
            if len(value) == 1:
                variant = list(value)[0]
                variant = variant.replace(' ,', ',')
                for choice in choices:
                    ratio = SequenceMatcher(None, variant, choice["text"]).ratio()
                    if ratio > 0.9:
                        final_pred_dict[key] = choice["id"]
                        choices.remove(choice)

        for key, value in label2hypothesys.items():
            if key not in final_pred_dict.keys():
                variant = []
                for var in value:
                    for choice in choices:
                        ratio = SequenceMatcher(None, var, choice["text"]).ratio()
                        if ratio > 0.9:
                            variant.append(var)
                if variant:
                    for choice in choices:
                        ratio = SequenceMatcher(None, variant[0], choice["text"]).ratio()
                        if ratio > 0.9:
                            final_pred_dict[key] = choice["id"]
                            choices.remove(choice)
                else:
                    variant = [choice for choice in choices]
                    if variant:
                        final_pred_dict[key] = variant[0]["id"]
                        for choice in choices:
                            ratio = SequenceMatcher(None, variant[0]["text"], choice["text"]).ratio()
                            if ratio > 0.9:
                                choices.remove(choice)

        for key, value in label2hypothesys.items():
            if key not in final_pred_dict.keys():
                variant = [choice for choice in choices]
                if variant:
                    final_pred_dict[key] = variant[0]["id"]
                    for choice in choices:
                        ratio = SequenceMatcher(None, variant[0]["text"], choice["text"]).ratio()
                        if ratio > 0.9:
                            choices.remove(choice)
        return final_pred_dict

    def predict_random(self, task):
        """ Test a random choice model """
        conditions = task["question"]["left"]
        choices = task["question"]["choices"]
        pred = {}
        for cond in conditions:
            pred[cond["id"]] = random.choice(choices)["id"]
        return pred

    def predict(self, task):
        if not self.has_model:
            return self.predict_random(task)
        else:
            return self.predict_from_model(task)

    def fit(self, tasks):
        pass

    def load(self, path="data/models/solver8.pkl"):
        pass

    def save(self, path="data/models/solver8.pkl"):
        pass

    def get_gerund1(self, task):


        pred = []
        may_be = []
        may_be2 = []
        may_be3 = []
        may_be4 = []
        may_be5 = []
        may_be6 = []
        for i, choice in enumerate(task):
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            case = ([str(self.morph.parse(ok)[0].tag.case) for ok in choice_list])
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            all_cases = ([self.morph.parse(ok) for ok in choice_list])

            if 'GRND' in pos:
                may_be6.append(str(i+1))
                may_be5.append(str(i+1))
                #print(i+1,choice)
                #print(pos)            
                #print(case)
                deepr,glag = False,False
                deepr_sy,glag_sy,glag_tsy = False,False,False
                
                prts = False
                glag_te = False
                infn = False
                glag_em = False
                bad_sen,bad_sen1,bad_sen2,bad_sen4,bad_sen5 = False,False,False,False,False
                noun_nomn = False
                bit = False
                deepr_impf = False
                kol_pos_zap=100
                glag_impf = False
                glag_perf = False
                deepr_zap  = False
                deepr_do_glag  = False
                #PRCL_do_GRND  = False
                for j,l in enumerate(pos):
                    if l == 'GRND' and choice_list[j] != 'душа':
                        # or choice_list[j][-2:] =='ся'
                        if j>1: 
                            for cas in all_cases[j-1]:
                                if cas.tag.POS =='CONJ':
                                    #PRCL_do_GRND = True
                                    #print('==========Союз до деепричастия')
                                    bad_sen4 = True
                        if (choice_list[j][-2:] =='сь'):
                            deepr_sy = True
                        for cas in all_cases[j]:
                            if cas.tag.aspect== 'impf':
                                deepr_impf = True
                        if not glag:
                            deepr_do_glag = True
                        deepr = True
                        #for cas in all_cases[j]:
                        #    if cas.tag.POS !='GRND':
                        #        deepr = False
                        #        break
                        #if not deepr:
                        #    break
                    if (deepr  and self.morph.parse(choice_list[j])[0].normal_form == 'быть'):
                        bit = True
                    for cas in all_cases[j]:
                        if cas.tag.POS =='VERB':
                            l = 'VERB'
                    if choice_list[j] == ',' and deepr_do_glag and glag and deepr and (glag_impf == deepr_impf or deepr_impf != glag_perf):
                        #print('+!+',choice_list[j-1])
                        break
                    if (l == 'VERB' or l == 'INFN'):
                        for cas in all_cases[j]:
                            if cas.tag.person == '3per':
                                #print('-----------3per')
                                bad_sen1 = True
                        #print(deepr_impf,glag_impf,glag_perf)
                        if glag and deepr and (glag_impf == deepr_impf or deepr_impf != glag_perf):
                            #print('+++',choice_list[j])
                            break
                        for cas in all_cases[j]:
                            if cas.tag.aspect== 'impf':
                                glag_impf = True
                            else:
                                glag_perf = True
                        if (choice_list[j][-2:] =='сь' or choice_list[j][-2:] =='ся'):
                            glag_sy = True
                        if (not glag)  and (choice_list[j][-3:] =='тсь' or choice_list[j][-3:] =='тся'):
                            #print('-'*10,choice_list[j])
                            if (j<len(pos)-1 and pos[j+1] != 'INFN'):
                                glag_tsy = True
                            elif  (j==len(pos)-1):
                                glag_tsy = True   
                        if (not glag or not deepr) and (choice_list[j][-2:] =='те'):
                            glag_te = True                    
                        if (not glag or not deepr) and (choice_list[j][-2:] =='ем'):
                            glag_em = True                   
                        if (l == 'INFN' and choice_list[j] !='быть'):
                            infn = True
                        if (l == 'VERB'):
                            #print('глагол',choice_list[j])
                            glag = True
                    kol_pos_zap+=1
                    if choice_list[j] == ',' and deepr and not glag:
                        noun_nomn = False
                        kol_pos_zap=0
                    if glag and not deepr:
                        noun_nomn = True 


                    if (l == 'PRTS'):
                        prts = True
                    if (deepr and (not glag or not deepr_do_glag)) and choice_list[j] == ',':
                        deepr_zap = True
                    if deepr and deepr_zap and (not noun_nomn or not deepr_do_glag) and ('меня' == choice_list[j] or 'моей' == choice_list[j] or 'мне' == choice_list[j] or 'её' == choice_list[j] or 'оно' == choice_list[j] or 'это' == choice_list[j]):
                        bad_sen = True
                        #may_be3.append(str(i+1))
                        #print('+'*20,'меня, моей, мне',i+1)
                    elif  ('меня' == choice_list[j] or 'моей' == choice_list[j] or 'мне' == choice_list[j] or 'её' == choice_list[j] or 'оно' == choice_list[j] or 'это' == choice_list[j]):
                        bad_sen5 = True
                        #may_be3.append(str(i+1))
                        #print('-'*20,'меня, моей, мне',i+1)
                    if deepr and not glag and kol_pos_zap<6:
                        for cas in all_cases[j]:
                            if cas.score > 0.01 and (cas.tag.POS =='NOUN' or cas.tag.POS =='NPRO') and cas.tag.case =='nomn':
                                noun_nomn = True 
                                #print('Подлежащее',choice_list[j])
                        #print(i,choice_list[j])
                    #if choice_list[j] == ',' and not glag:
                    #    bit = False

                        
                    #print(choice_list[j],kol_pos_zap,noun_nomn)
                    """if choice_list[j] == ',' and not glag:
                        noun_do_glag = False
                    if (l == 'NOUN' or l == 'NPRO') and choice_list[j] != 'мной' and not glag:
                        noun_do_glag = True  """            
                #noun_do_glag = False
                
                        
                #print(glag,deepr,glag_sy,deepr_sy)    

                #if (glag and deepr and not glag_sy and  deepr_sy or  glag_sy and not deepr_sy):
                #    print('ТИП 1. Деепричастие и глагольное сказуемое, выраженное глаголом без постфикса -ся')
                #    bad_sen1 = True
                if (deepr and prts and not (glag or infn)):
                    #print('ТИП 2. Деепричастие относится к сказуемому в форме краткого страдательного причастия')
                    bad_sen = True    
                #if (glag and deepr and  glag_tsy and not deepr_sy):
                #    print('ТИП 3. Деепричастный оборот прикреплён к сказуемому- возвратному глаголу в страдательном значении, имеющему постфикс ся')
                #    bad_sen2 = True

                if (glag and deepr and glag_te and not infn and not glag_em):
                    #print('ТИП 6. Деепричастный оборот относится к глаголу в повелительном наклонении')
                    bad_sen = False
                    if str(i+1) in may_be5:
                        may_be5.remove(str(i+1))
                elif (infn and deepr and not glag_te  and not glag and not glag_em  and not prts):
                    #print('ТИП 7. Деепричастный оборот относится к инфинитиву')
                    bad_sen = False
                    if str(i+1) in may_be5 :
                        may_be5.remove(str(i+1))                    
                elif (glag and deepr and not glag_te and not infn and glag_em):
                    #print('ТИП 8. Деепричастный оборот в определённо-личном или обобщенно-личном предложении')
                    bad_sen = False
                    if str(i+1) in may_be5 :
                        may_be5.remove(str(i+1))
                elif bit and deepr:
                    #print('Есть слово быть')
                    #bad_sen = True 
                    may_be.append(str(i+1))                    
                elif deepr_impf != glag_impf and deepr_impf == glag_perf and deepr:
                    #print('разный несовершенный вид')
                    may_be2.append(str(i+1))
                elif bad_sen4:
                    may_be2.append(str(i+1))                    
                elif ((infn or glag) and deepr and  not noun_nomn and not glag_te and not glag_em):
                    if (pos[0] !='NOUN' and pos[0] == 'NPRO') or choice_list[1] != ',':
                        #print('ТИП 4 b 5.Деепричастный оборот в безличном предложении Деепричастный оборот в неопределённо-личном предложении')
                        #may_be.append(str(i+1))   
                        bad_sen2 = True    
                        if bad_sen1:
                            may_be.append(str(i+1))  
                #print('-'*30)
                if bad_sen:
                    pred.append(str(i+1))
                if bad_sen1  or bad_sen5:
                    may_be4.append(str(i+1))
                if bad_sen2:
                    may_be3.append(str(i+1))                    
        #print(pred,may_be,may_be2,may_be3,may_be4,may_be5)
        if not pred:
            pred = may_be
        if not pred:
            pred = may_be2    
        if not pred:
            pred = may_be3             
        if not pred:
            pred = may_be4
        if not pred:
            pred = may_be5           
        if not pred:
            pred = may_be6         
        return pred
    def get_indirect_speech1(self, task):
        #-------------------------------МУСОР------------------------------------
        sv = ['что', 'будто', 'чтобы', 'кто', 'что', 'какой', 'как', 'где', 'когда', 'почему', 'ли']

        pred = []
        may_be = []
        for i, choice in enumerate(task):
            bad_sen = False
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.split()
           
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)            
            #for j,l in enumerate(pos):
            for s in sv:
                if s in choice_list:
                    bad_sen = True
            if bad_sen:
                pred.append(str(i+1))
        return pred
        
    def get_app1(self, task):
        pred = []
        may_be = []
        for i, choice in enumerate(task):
            bad_sen = False
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            case = ([str(self.morph.parse(ok)[0].tag.case) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)
            #print(case)
            nach_app = False
            end_app = False
            glag = False
            no_nomn = False
            for j,l in enumerate(pos):
                if nach_app and not end_app:
                    if (case[j]!='nomn' and case[j]!='None'):
                        no_nomn =  True
                        #print(choice_list[j],pos[j],case[j])                
                if choice_list[j] =='«':
                    nach_app = True
                if choice_list[j] =='»':
                    end_app = True

                if nach_app and not end_app and (l == 'VERB' or l == 'INFN'):
                    glag = True
            if nach_app and end_app and not glag and no_nomn:
                pred.append(str(i+1))
            #if not i+1 in pred:
            #    if nach_app and end_app and not glag:
            #        print('Ahtung!!!')
            #        pred.append(i+1)
        return pred
    def get_homogeneous1(self, task):
    
        double_conjs =[["не только","но и",'но даже','а еще','а к тому же'],
            ['не только не','но','но скорее','скорее','напротив','наоборот','а не только'],
            ['не то что','но','а','просто','даже','даже не'],
            ['даже','не то что'],
            ['даже не','не то что'],
            ['даже не','тем более не'],
            ['мало того','еще и'],
            ['мало того что','еще и'],
            ['мало того'],
            ['более того'],
            ['больше того'],
            ['хуже того'],
            ['а то и'],
            ['не сколько','сколько'],
            ['не столько','столько'],
            ['не столько','сколько'],
            ['не то чтобы'],
            ['если не','то'],
            ['как','так и']
        ]
        #,'а и'
        pred = []
        may_be1,may_be2,may_be3,may_be4,may_be5,may_be6,may_be7,may_be0 = [],[],[],[],[],[],[],[]
        for i, choice in enumerate(task):
            est_PRTF = False
            est_VERB = False            
            bad_sen = False
            est_CONJ = False
            
            choice = ' '+choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()+' '
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            case = ([str(self.morph.parse(ok)[0].tag.case) for ok in choice_list])
            transitivity = ([str(self.morph.parse(ok)[0].tag.transitivity) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)
            #print(case) 
            #print(transitivity)
            transitivitys = []
            id_CONJ = 0
            bad_sen0,bad_sen,bad_sen1,bad_sen2,bad_sen3,bad_sen4 = False, False, False, False, False, False
            bad_sen5,bad_sen6,bad_sen7,bad_sen10 = False, False, False, False
            for j,l in enumerate(pos):
                if (j>1 and j+3<len(pos) and choice_list[j]==',' and pos[j-1] == 'NOUN' and pos[j-2] == 'PREP' and pos[j+1] == 'NOUN' 
                and choice_list[j+2]==',' and pos[j+3] == 'NOUN'):
                    bad_sen6 = True
                    #print('-0.1 Прелог+сущ')
                
                
                
                
                if l == 'CONJ' and not id_CONJ:
                    #id_CONJ = choice_list[j]
                    id_CONJ = j
                
                if (l == 'VERB' or l =='INFN') and not transitivity[j] in transitivitys:
                    transitivitys.append(transitivity[j])
                if choice_list[j]==',' and len(transitivitys)==1:
                    transitivitys = []       
                    est_PRTF = False
                    est_VERB = False
                    est_CONJ = False
                if l == 'PRTF':
                    est_PRTF = True
                if l == 'VERB':
                    est_VERB = True    
                if est_PRTF and est_VERB and est_CONJ:
                    #print('-2.1 est_PRTF и est_VERB')
                    bad_sen2 = True
                if l == 'CONJ' and est_PRTF != est_VERB:
                    est_CONJ = True
                if (j>1 and j+1<len(pos)) and choice_list[j]==':' and pos[j-1] == 'NOUN' and pos[j-1] == 'NOUN' and case[j-1] != case[j+1]:
                    #print('-1.1 Есть :')
                    bad_sen4 = True                
                if self.morph.parse(choice_list[j])[0].normal_form =='который':
                    #print('+1.4 который')
                    bad_sen10 = True 
                
                #Eif (j>1 and j+1<len(pos) and (pos[j-1] == 'NOUN' or pos[j-1] == 'NPRO') and
                #   choice_list[j]=='и' and (pos[j+1] == 'NOUN' or pos[j+1] == 'NPRO')):
                #    if case[j-1] != case[j+1]:
                #        print('+1.4 соседни сущ')
                #        bad_sen3 = True                
                if (j>1 and j+1<len(pos) and (pos[j-1] == 'VERB' or pos[j-1] == 'INFN') and
                   choice_list[j]=='и' and (pos[j+1] == 'VERB' or pos[j+1] == 'INFN')):
                    #print('+1.2 соседни')
                    bad_sen3 = True
                if (j>1 and j+2<len(pos) and (pos[j-2] == 'VERB' or pos[j-1] == 'VERB' or pos[j-2] == 'INFN' or pos[j-1] == 'INFN') and
                   choice_list[j]=='и' and (pos[j+2] == 'VERB' or pos[j+1] == 'VERB' or pos[j+2] == 'INFN' or pos[j+1] == 'INFN') and
                    choice_list[j-1]!=','):
                        trans_do = None
                        trans_after = None
                        if (pos[j-1] != 'NOUN' and pos[j-1] != 'NPRO')and (pos[j-2] == 'VERB' or pos[j-2] == 'INFN') and choice_list[j-2] !='был':
                            trans_do = transitivity[j-2]
                        if (pos[j-1] == 'VERB' or pos[j-1] == 'INFN'):
                            trans_do = transitivity[j-1]                        
                        if (pos[j+2] == 'VERB' or pos[j+2] == 'INFN'):
                            trans_after = transitivity[j+2]
                        if pos[j+1] == 'VERB' or pos[j+1] == 'INFN':
                            if not trans_after or trans_do != trans_after:
                                trans_after = transitivity[j+1]
                        #print(trans_do,trans_after,choice_list[j-1],choice_list[j+1])
                        if trans_do and trans_after and trans_do != trans_after:
                            #print('-0.2 Разные соседние transitivitys')
                            bad_sen0 = True
            if len(transitivitys)>1:
                #print(transitivitys)
                #print('-3.1 Разные transitivitys')
                bad_sen1= True
            if id_CONJ:
                j = id_CONJ
                if pos[j-1] != pos[j+1] and (pos[j+1] =='INFN' or pos[j+1] =='VERB' or pos[j+1] =='NOUN') and (pos[j-1] =='INFN' or pos[j-1] =='VERB' or pos[j-1] =='NOUN'):
                    #print('-1.3 Разные части речи')
                    bad_sen = True
                """sen_CONJ = choice.split(id_CONJ)
                sen_CONJ_list0 = sen_CONJ[0].split()
                sen_CONJ_poss0 = ([str(self.morph.parse(ok)[0].tag.POS) for ok in sen_CONJ_list0])
                sen_CONJ[1] = re.sub('\,.*','',sen_CONJ[1])
                sen_CONJ_list1 = sen_CONJ[1].split()
                sen_CONJ_poss1 = ([str(self.morph.parse(ok)[0].tag.POS) for ok in sen_CONJ_list1])
                for j,l in enumerate(sen_CONJ_poss1):
                    if l == 'INFN' or l == 'VERB' or l == 'NOUN':
                        if not l in sen_CONJ_poss0:
                            print('Разные части речи')
                            pred.append(i+1)"""
                
            
            
            
            
            for double_conj in double_conjs:
                if ' '+double_conj[0]+' ' in choice:
                    est_2 = False
                    #print(double_conj)
                    choice1 = re.sub('.*'+double_conj[0],'',choice)
                    choice2 = re.search('.*'+double_conj[0],choice).group(0)
                    choice2 =  re.sub(double_conj[0],'',choice2)[-20:]
                    if len(double_conj)>1:
                        for double_con in double_conj[1:]:
                            if ' '+double_con+' ' in choice1:
                                est_2 = True
                                mem = double_con
                                
                        if not est_2 and double_conj[0] !='как' and double_conj[0] !='даже' and double_conj[0] !='даже не' :
                            bad_sen3 = True
                            #print('-1.5 Одно из двух',double_conj[0])
                        elif est_2:
                            #print('--Есть оба',double_conj[0],mem)
                            sens = re.sub('.*'+double_conj[0],'',choice)
                            sens = re.sub(double_conj[1],'',sens)
                            sens = sens.split(',')
                            mem_sen_VERB = False
                            mem_sen_NOUN = False
                            VERB_do_NOUN = False
                            #print(sens)
                            #mem_kol_sen_NOUN = 0
                            mem_kol_sen = 0
                            for kk,sen in enumerate(sens[:2]):
                                sen_VERB = False
                                sen_NOUN = False
                                sen_list = sen.split()
                                sen_poss = ([str(self.morph.parse(ok)[0].tag.POS) for ok in sen_list])
                                #print(sen_poss)

                                k = 0
                                #kol_sen_NOUN = 0
                                for jj,sen_pos in enumerate(sen_poss[:len(sen_poss[0])+2]):
                                    if sen_pos == 'VERB' or sen_pos == 'INFN':
                                        sen_VERB = True
                                        k +=1
                                        if kk ==1 and not sen_NOUN:
                                            VERB_do_NOUN = True
                                    elif sen_pos == 'NOUN' or sen_pos == 'NPRO':
                                        sen_NOUN = True
                                        #kol_sen_NOUN +=1
                                        k +=1
                                    elif sen_pos != 'INTJ' and sen_pos != 'PRCL' and sen_pos != 'CONJ'and sen_pos != 'PREP'and sen_pos != 'None' and sen_pos != 'PNCT':    
                                        k +=1
                                    if kk==1 and ' '+sen_list[jj]+' ' in choice2:
                                        #print('!!!',sen_list[jj],choice2)
                                        bad_sen7 = True  
                                #if kk ==1 and mem_kol_sen_NOUN and mem_kol_sen_NOUN != kol_sen_NOUN:
                                #    bad_sen5 = True
                                #    print('-1.5 Разное число существительных') 
                                #mem_kol_sen_NOUN = kol_sen_NOUN
                                #print('!',k,sen)
                                if mem_kol_sen and ( mem_kol_sen-k>=2 or mem_kol_sen/k>=2):
                                    bad_sen5 = True
                                    #print('-1.4 Разное число слов')                                    
                                mem_kol_sen = k
                                if kk ==1 and (mem_sen_VERB == True and sen_VERB == False or mem_sen_NOUN == True and sen_NOUN == False):
                                    #print('-0.3 Разное число сущ+гл')
                                    bad_sen7 = True
                                if kk ==1 and (not mem_sen_VERB and sen_VERB and VERB_do_NOUN):
                                    #print('-0.3 Разное число сущ+гл. Глагол до сущ')
                                    bad_sen7 = True                                    
                                
                                mem_sen_VERB = sen_VERB
                                mem_sen_NOUN = sen_NOUN

                    else:
                        #print('-1.5 Одно')
                        bad_sen = True
                                
            if bad_sen10 and (bad_sen or bad_sen1 or bad_sen2):
                may_be7.append(str(i+1)) 
            
            if  bad_sen6:
                pred.append(str(i+1))
            if  bad_sen7:
                may_be0.append(str(i+1))                
            if bad_sen5 or bad_sen4:
                may_be1.append(str(i+1))             
            if bad_sen3:
                may_be6.append(str(i+1))  
            if bad_sen:
                may_be7.append(str(i+1))               
            if bad_sen2:
                may_be2.append(str(i+1))                
            if bad_sen1 or bad_sen10:
                may_be3.append(str(i+1))
            if bad_sen0:
                may_be5.append(str(i+1))
                
        #print(pred,may_be0,may_be5,may_be1,may_be6,may_be7,may_be2,may_be3)
        if not pred:
            pred = may_be0        
        if not pred:
            pred = may_be5
        if not pred:
            pred = may_be1        
        if not pred:
            pred = may_be6  
        if not pred:
            pred = may_be7            
        if not pred:
            pred = may_be2
        if not pred:
            pred = may_be3            
        return pred        
    def get_participle1(self, task):
        pred = []
        may_be = []
        for i, choice in enumerate(task):
            bad_sen = False
            bad_sen1 = False
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            case = ([str(self.morph.parse(ok)[0].tag.case) for ok in choice_list])
            
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            """all_cases = []
            for ok in choice_list:
                all_case = []
                zzzz=self.morph.parse(ok)
                for j,zzz in enumerate(zzzz):
                    #zzzz[j].score>0.2 and 
                    if not zzzz[j].tag.case in all_case:
                        all_case.append(zzzz[j].tag.case)
                all_cases.append(all_case)   """ 
            all_cases = ([self.morph.parse(ok) for ok in choice_list])

            #print(i+1,choice)
            
            #print(pos)
            #print(case)
            #print(all_cases)
            for j,l in enumerate(pos):
                ex = False
                if l =='PRTF':
                    zap = False
                    #----Доделать, если два причастия относятся к одному сущ--
                    for z in range(j-1,-1,-1): 
                        if choice_list[z]==',':
                            zap = True
                            break
                        if pos[z]=='NOUN' or z == 0:
                            zap = False
                            break
                    #print(zap)
                    bad_sen1 = True
                    if zap:
                        
                        
                        for z in range(j-2,-1,-1):
                            if pos[z]=='NOUN':
                                for all_cases_j in all_cases[j]:
                                    for all_cases_z in all_cases[z]:
                                        #print(choice_list[z],all_cases_z.tag.case,all_cases_z.tag.gender,all_cases_z.tag.number,choice_list[j],all_cases_j.tag.case,all_cases_j.tag.gender,all_cases_j.tag.number)
                                        od_case, od_gender, od_number = False, False, False 
                                        if (str(all_cases_j.tag.case)=='None' or str(all_cases_z.tag.case)=='None' or all_cases_j.tag.case == all_cases_z.tag.case): 
                                            od_case = True
                                        if (str(all_cases_j.tag.gender)=='None' or str(all_cases_z.tag.gender)=='None' or all_cases_j.tag.gender == all_cases_z.tag.gender):
                                            od_gender = True
                                        if (str(all_cases_j.tag.number)=='None' or str(all_cases_z.tag.number)=='None' or all_cases_j.tag.number == all_cases_z.tag.number):
                                            od_number = True
                                        #print(od_case,od_gender,od_number)
                                        if od_case and od_gender and od_number:
                                            ex = True
                                            break
                                    if ex:
                                        break        
                                if ex:
                                    break
                                bad_sen = True
                                #print('---1',choice_list[z],pos[z],case[z],choice_list[j],pos[j],case[j])
                                break

                    else:
                
                        for z in range(j,len(pos)-1):
                            if pos[z]=='NOUN':
                                if choice_list[z]==',':
                                    break 
                                for all_cases_j in all_cases[j]:
                                    for all_cases_z in all_cases[z]: 
                                        #print(choice_list[z],all_cases_z.tag.case,all_cases_z.tag.gender,all_cases_z.tag.number,choice_list[j],all_cases_j.tag.case,all_cases_j.tag.gender,all_cases_j.tag.number)
                                        od_case, od_gender, od_number = False, False, False 
                                        if (str(all_cases_j.tag.case)=='None' or str(all_cases_z.tag.case)=='None' or all_cases_j.tag.case == all_cases_z.tag.case): 
                                            od_case = True
                                        if (str(all_cases_j.tag.gender)=='None' or str(all_cases_z.tag.gender)=='None' or all_cases_j.tag.gender == all_cases_z.tag.gender):
                                            od_gender = True
                                        if (str(all_cases_j.tag.number)=='None' or str(all_cases_z.tag.number)=='None' or all_cases_j.tag.number == all_cases_z.tag.number):
                                            od_number = True
                                        if od_case and od_gender and od_number:
                                            ex = True
                                            break
                                    if ex:
                                        break                                        
                                if ex:
                                    if z-j<2 and choice_list[z+1]!='.' and choice_list[z+1]!=',':
                                        #print('---2',choice_list[z],pos[z],case[z],choice_list[j],pos[j],case[j],choice_list[z+1])
                                        bad_sen = True
                                        break
                                    else:
                                        #print('Всё хорошо',choice_list[j],choice_list[z])
                                        bad_sen = False
                                        break
                                else:
                                    bad_sen = True
                                    #print('---3',choice_list[z],pos[z],case[z],choice_list[j],pos[j],case[j])
                if bad_sen:
                    break
            #for j,l in enumerate(pos):
            #if 'PRTF' in pos:
            #    bad_sen = True
            if bad_sen1:
                may_be.append(str(i+1))            
            if bad_sen:
                pred.append(str(i+1))
        #print(pred,may_be)
        if not pred:
            pred = may_be       
        return pred
    def get_numerals1(self, task):
        pred = []
        may_be = []
        for i, choice in enumerate(task):
            bad_sen = False
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)
            if 'NUMR' in pos:
                bad_sen = True
            if bad_sen:
                pred.append(str(i+1))
        return pred            
    def get_nouns1(self, task):
        apds = ["благодаря", "согласно", "вопреки", "подобно", "наперекор", "наперерез",'вслед']
        #, "вместе", "наряду"
        #predlogi1 = ['ввиду', 'вследствие', 'в случае', 'при условии', 'при помощи']
        posle = ['окончанию','возвращению','уходу','прилёту','истечению','прибытию','приезду','завершению']
        predlogi1 = ['ввиду', 'вследствие', 'случае', 'условии', 'помощи']
        predlogi = {'от':['gent','gen2'],'без':['gent','gen2'],'у':['gent','gen2'],'до':['gent','gen2'],'возле':['gent','gen2'],'для':['gent','gen2'],'вокруг':['gent','gen2'],'из':['gent','gen2'],
            'около':['gent','gen2'],'с':['gent','ablt'],'из-за':['gent','gen2'],'из-под':['gent','gen2'],'вроде':['gent','gen2'],'среди':['gent','gen2'],'кроме':['gent','gen2'],
            'ради':['gent','gen2'], 'навстречу':['datv'], 'вдоль':['gent','gen2'],'прежде':['gent','gen2'],'взамен':['gent','gen2'],
            'по':['datv','loct','loc2'],'к':['datv'],'через':['accs','acc2'],'про':['accs','acc2'],'взгляд':['accs','acc2'],
            'за':['accs','acc2','ablt'],'под':['accs','acc2','ablt'],'над':['ablt'],'перед':['ablt'],'между':['ablt','gent'],
            'о':['loct','loc2'],'об':['loct','loc2'],'на':['accs','acc2','loct','loc2'],'в':['loct','loc2','accs','acc2'],'во':['loct','loc2'],'при':['loct','loc2'],'обо':['loct','loc2']}        
        #,'в':['loct','loc2','accs','acc2']gen2
        pred = []
        may_be0 = []
        may_be1 = []
        may_be2 = []
        may_be3 = []
        may_be4 = []
        may_be5 = []
        may_be6 = []
        for i, choice in enumerate(task):
            bad_sen = False
            bad_sen1,bad_sen2,bad_sen3,bad_sen4,bad_sen5,bad_sen6 = False, False, False, False, False, False
            bad_sen7,bad_sen8,bad_sen0 = False,False,False
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.replace('»',' »').replace('«','« ').lower().split()
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            case = ([str(self.morph.parse(ok)[0].tag.case) for ok in choice_list])
            all_cases = ([self.morph.parse(ok) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)
            #print(case)
            
            for j,l in enumerate(pos):
                if choice_list[j] in apds:
                    odnorod = False
                    for z in range(j+1,len(pos)):
                        if not odnorod and (choice_list[z] == ',' or pos[z] == 'CONJ' or pos[z] == 'PREP'):
                            break
                        if (choice_list[z] == '.' or choice_list[z] == '-' or pos[z] == 'VERB'):
                            break                        
                        if pos[z] == 'NOUN' or pos[z] == 'PRTF' :
                            #print('----',odnorod, choice_list[j], choice_list[z])    
                            for cas in all_cases[z]:
                                if str(cas.tag.case) == 'datv':
                                    bad_sen = True
                            if bad_sen:
                                bad_sen = False
                            else:
                                #print('--0_2', choice_list[j], choice_list[z])                            
                                bad_sen = True
                            if bad_sen  or (z<len(pos)-1 and choice_list[z+1] != ',' and pos[z+1] != 'CONJ'):
                                break
                            elif z<len(pos)-1 and (choice_list[z+1] == ',' or pos[z+1] == 'CONJ'):
                                
                                if (choice_list[z+1] == ','):
                                    for zz in range(z+2,z+5):
                                        if zz<len(pos) and pos[zz] == 'CONJ':
                                            odnorod = True
                                if pos[z+1] == 'CONJ':
                                    odnorod = True
                            else:
                                odnorod = False
                                
                if bad_sen:
                    break
                if choice_list[j] in predlogi1:
                    odnorod = False
                    for z in range(j+1,len(pos)):
                        if not odnorod and (choice_list[z] == ',' or pos[z] == 'CONJ'):
                            break
                        if (choice_list[z] == '.' or choice_list[z] == '-' or pos[z] == 'VERB'):
                            break  
                        if pos[z] == 'NOUN':
                            #print('----',odnorod, choice_list[j], choice_list[z])    
                            for cas in all_cases[z]:
                                if str(cas.tag.case) == 'gent':
                                    bad_sen = True
                            if bad_sen:
                                bad_sen = False
                            else:
                                #print('--0_1', choice_list[j], choice_list[z])                            
                                bad_sen = True
                            break
                            if bad_sen or (choice_list[z+1] != ',' and pos[z+1] != 'CONJ'):
                                break
                            elif (choice_list[z+1] == ',' or pos[z+1] == 'CONJ'):
                                if (choice_list[z+1] == ','):
                                    for zz in range(z,z+3):
                                        if pos[zz] == 'CONJ':
                                            odnorod = True
                                if pos[z+1] == 'CONJ':
                                    odnorod = True
                            else:
                                odnorod = False
                if bad_sen:
                    break                    
            for j,l in enumerate(pos):        
                if choice_list[j] == 'по':
                    for z in range(j,len(pos)):
                        if choice_list[z] == ',' or pos[z] == 'VERB' or pos[z] == 'CONJ':
                            break
                        if pos[z] == 'NOUN':
                            for cas in all_cases[z]:
                                if str(cas.tag.case) == 'loct':
                                    bad_sen1 = True
                            if bad_sen1:
                                bad_sen1 = False
                            else:
                                if choice_list[z] in posle:
                                    #print('--0.3', choice_list[j], choice_list[z])                            
                                    bad_sen1 = True                                    
                                    bad_sen0  = True  
                                else:
                                    #print('--1', choice_list[j], choice_list[z])                            
                                    bad_sen1 = True
                            break    
                if bad_sen1:
                    break
            for j,l in enumerate(pos):

                if l == 'PREP' or l == 'ADVB' or choice_list[j] == 'взгляд':
                    for k in predlogi.keys(): 
                        if  choice_list[j] == k:
                            for z in range(j+1,len(pos)):
                                if (choice_list[z] == ',' or choice_list[z] == '.' or pos[z] == 'PREP' or pos[z] == 'VERB' or pos[z] == 'CONJ') and choice_list[j] != 'взгляд':
                                    break 
                                if self.morph.parse(choice_list[z])[0].normal_form == 'который' or (pos[z] == 'ADVB'):
                                    break
                                #if pos[z] == 'NOUN' or (pos[z] == 'NPRO' and z<len(pos)-1 and pos[z+1] != 'NOUN'):
                                if pos[z] == 'NOUN' or pos[z] == 'NPRO':
                                    for cas in all_cases[z]:
                                        if cas.score>0.01 and (str(cas.tag.case) in predlogi[k]):
                                            bad_sen2 = True
                                            break
                                            
                                    if bad_sen2:
                                        bad_sen2 = False
                                    else:
                                        if choice_list[j] != 'c' or choice_list[z]!='боку':
                                            #print('--2', choice_list[j], choice_list[z])  
                                            bad_sen2 = True  
                                    break  
                        if bad_sen2:
                            break                    
            #-----Если непереходный глагол и существительное в винительном падеже
            for j in range(len(pos)-2):
                #if pos[j+1] == 'ADJF' and pos[j+1] =='PREP' and pos[j+2] 'NOUN':
                #    may_be3.append(i+1) 
                neperehod = False
                if pos[j] == 'VERB' and pos[j+1] =='PREP':
                    for cas in all_cases[j]:
                        if str(cas.tag.transitivity) == 'intr':
                            neperehod = True
                    if neperehod:
                        for z in range(j+2,len(pos)):
                            if choice_list[z] == ',' or choice_list[z] == '.' or pos[z] == 'PREP' or pos[z] == 'VERB' or pos[z] == 'CONJ':
                                break                        
                            if pos[z] == 'NOUN' :
                                for cas in all_cases[z]:
                                    if str(cas.tag.case) !='accs':
                                        bad_sen3 = True
                                if bad_sen3:
                                    bad_sen3 = False
                                else:
                                    #print('--3', choice_list[j], choice_list[z])                                
                                    bad_sen3 = True
                                break                                         
                                
                
                if pos[j] == 'ADJF' and pos[j+1] =='PREP':
                    for z in range(j+2,len(pos)):
                        if pos[z] == 'NOUN' and z<len(pos)-1 and pos[z+1] != 'NOUN':
                            for cas_j in all_cases[j]:
                                for cas_z in all_cases[z]:
                                    if cas_j.tag.case == cas_z.tag.case: 
                                        bad_sen4 = True
                            if bad_sen4:
                                bad_sen4 = False
                            else:
                                #print('--4', choice_list[j], choice_list[z])
                                bad_sen4 = True
                            break                                         
                if pos[j] == 'ADJF' and pos[j+1] =='NOUN':
                    for cas_j in all_cases[j]:
                        for cas_z in all_cases[j+1]:
                            if cas_j.tag.case == cas_z.tag.case: 
                                bad_sen5 = True
                               
                    if bad_sen5:
                        bad_sen5 = False
                    else:
                        #print('--5',choice_list[j],choice_list[j+1]) 
                        bad_sen5 = True
                if pos[j] == 'PRTF': 
                    for z in range(j+1,len(pos)):
                        if choice_list[z] == ',' or pos[z] == 'VERB' or pos[z] == 'CONJ' or pos[z] == 'PREP': 
                            break                    
                        if pos[z] =='NOUN':
                            for cas_j in all_cases[j]:
                                for cas_z in all_cases[z]:
                                    if cas_j.score>0.2 and cas_z.score>0.2 and cas_j.tag.case == cas_z.tag.case: 
                                        bad_sen6 = True
                                        
                            if bad_sen6:
                                bad_sen6 = False
                            else:
                                #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--6',choice_list[j],choice_list[z])
                                bad_sen6 = True                                                       
                            break             
                if pos[j] == 'ADJS' or pos[j] == 'ADJF': 
                    for z in range(j+1,len(pos)):
                        if choice_list[z] == ',' or z-j>4 or pos[z] == 'VERB' or pos[z] == 'CONJ':
                            break
                        if pos[z] =='NOUN':
                            for cas_j in all_cases[j]:
                                for cas_z in all_cases[z]:
                                    if cas_j.tag.number == cas_z.tag.number: 
                                        bad_sen7 = True
                                     
                            if bad_sen7:
                                bad_sen7 = False
                            else:
                                #print('--7',choice_list[j],choice_list[z])   
                                bad_sen7 = True                                                       
                            break                              
                if j<len(pos)-2 and pos[j] == 'NOUN' and choice_list[j+1] == 'в' and pos[j+2] == 'NOUN': 
                    for cas_j in all_cases[j]:
                        for cas_z in all_cases[j+2]:
                            if cas_z.tag.case == 'loct' : 
                                bad_sen8 = True
                             
                    if bad_sen8:
                        bad_sen8 = False
                    else:
                        #print('--8',choice_list[j],choice_list[z])   
                        bad_sen8 = True                                                       
                    break            
            
            
            if bad_sen0:
                pred.append(str(i+1))
            if bad_sen:
                may_be0.append(str(i+1))                
            if bad_sen1:
                may_be1.append(str(i+1))                
            if bad_sen8:
                may_be2.append(str(i+1)) 
            if bad_sen2:
                may_be3.append(str(i+1)) 
            if bad_sen4 or bad_sen5:
                may_be4.append(str(i+1)) 
            if bad_sen3 or bad_sen6:
                may_be6.append(str(i+1))                
            if bad_sen7:
                may_be5.append(str(i+1))
        #print(pred,may_be0,may_be3,may_be4,may_be6, may_be1,may_be5,may_be2)
        if not pred:
            pred = may_be0        
        if not pred:
            pred = may_be3
        if not pred:
            pred = may_be4     
        if not pred:
            pred =  may_be6  
        if not pred:
            pred = may_be1
        if not pred:
            pred = may_be5
        if not pred:
            pred = may_be2         
        return pred    
    def get_verbs1(self, task):
        pred = []
        may_be = []
        may_be2 = []
        for i, choice in enumerate(task):
            bad_sen = False
            bad_sen1 = False
            bad_sen2 = False
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            all_cases = ([self.morph.parse(ok) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)
            for j,l in enumerate(pos):
                if l == 'VERB':
                    for z in range(j+1,len(pos)):
                        if choice_list[z] ==',' or choice_list[z] =='.':
                            break
                        if pos[z] == 'VERB':
                            for cas_j in all_cases[j]:
                                for cas_z in all_cases[z]:
                                    if (cas_j.tag.aspect == cas_z.tag.aspect) and (cas_j.tag.tense == cas_z.tag.tense): 
                                        bad_sen = True
                            if bad_sen:
                                bad_sen = False
                            else:
                                #print(choice_list[j],choice_list[z])  
                                bad_sen = True                                                       
                            break
                    if bad_sen:
                        break    
            if bad_sen:
                pred.append(str(i+1))
            for j,l in enumerate(pos):
                if l == 'VERB' and j<len(pos)-1 and choice_list[j+1] !=',':
                    for z in range(j+1,len(pos)):
                        if pos[z] == 'VERB':
                            for cas_j in all_cases[j]:
                                for cas_z in all_cases[z]:
                                    if cas_j.tag.aspect == cas_z.tag.aspect and (cas_j.tag.tense == cas_z.tag.tense): 
                                        bad_sen2 = True
                            if bad_sen2:
                                bad_sen2 = False
                            else:
                                #print(choice_list[j],choice_list[z])  
                                bad_sen2 = True                                                       
                            break
                    if bad_sen2:
                        break    
            if bad_sen2:
                may_be2.append(str(i+1))
            for j,l in enumerate(pos):
                if l == 'VERB':
                    for z in range(j+1,len(pos)):
                        if pos[z] == 'VERB':
                            for cas_j in all_cases[j]:
                                for cas_z in all_cases[z]:
                                    if cas_j.tag.aspect == cas_z.tag.aspect and (cas_j.tag.tense == cas_z.tag.tense): 
                                        bad_sen1 = True
                            if bad_sen1:
                                bad_sen1 = False
                            else:
                                #print(choice_list[j],choice_list[z])  
                                bad_sen1 = True                                                       
                            break
                    if bad_sen1:
                        break             
            if bad_sen1:
                may_be.append(str(i+1))
        #print(pred,may_be2,may_be)
        if not pred:
            pred = may_be2        
        if not pred:
            pred = may_be        
        return pred  
    def get_clause1(self, task):
        pred = []
        may_be = []
        may_be2 = []
        for i, choice in enumerate(task):
            bad_sen = False
            bad_sen1 = False
            bad_sen2 = False
            kol_VERB = 0
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            all_cases = ([self.morph.parse(ok) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)
            for j,l in enumerate(pos):
                if (j< len(pos)-2 and self.morph.parse(choice_list[j])[0].normal_form =='тот' or self.morph.parse(choice_list[j])[0].normal_form =='то') and choice_list[j+1] == ',' and self.morph.parse(choice_list[j+2])[0].normal_form =='что':
                    #print('-------то , что------------')
                    bad_sen = True
                
                
                if choice_list[j] == 'ли':
                    #print('-------ли------------')
                    bad_sen = True
                if pos[j] == 'CONJ' and j< len(pos)-1 and pos[j+1] == 'CONJ' and (choice_list[j] != 'так' or choice_list[j+1] != 'что') and choice_list[j+1] != 'и' and choice_list[j] != 'но':
                    #print('-------CONJ + CONJ------------')
                    bad_sen1 = True
                if pos[0] == 'CONJ' and choice_list[j] == ',' and pos[j+1] == 'CONJ':
                    #print('-------хз------------')
                    bad_sen2 = True
                if l == 'VERB':
                    
                    kol_VERB +=1
            if bad_sen:
                pred.append(str(i+1))
            if kol_VERB>1:
                #print('-------Много глаголов------------')
                may_be2.append(str(i+1))
            if bad_sen1 or bad_sen2:
                may_be.append(str(i+1))      
        #print(pred,may_be,may_be2)
        if not pred:
            pred = may_be 
        if not pred:
            pred = may_be2             
        return pred              
    def get_podch1(self, task):
        souzs = ['с тех пор как', 'что', 'когда','как','тот', 'этот', 'такой', 'там', 'туда', 'оттуда', 'тогда', 'так', 
            'настолько', 'столько', 'потому', 'оттого','весь', 'все', 'каждый', 'всякий', 'везде', 'всюду', 'всегда',
             'Никто', 'ничто', 'нигде', 'никогда','кто-то', 'что-то', 'где-то', 'когда-то',
            'что', 'чтобы', 'как', 'когда', 'когда', 'едва', 'пока', 'как', 'с тех пор как', 'лишь только',
            'потому что', 'так как', 'ввиду того что', 'ибо','если', 'коли', 'кабы', 'когда', 'раз',
            'хотя', 'несмотря на то что', 'вопреки тому что','так что','поэтому'
            'чтобы', 'с тем чтобы', 'для того чтобы','как', 'словно', 'как будто', 'чем', 'точно', 'подобно тому как']
        pred = []
        may_be = []
        may_be2 = []
        may_be3 = []
        for i, choice in enumerate(task):
            bad_sen = False
            bad_sen1 = False
            bad_sen2 = False
            kol_kotorii = 0
            choice = choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            choice_list_norm = ([str(self.morph.parse(ok)[0].normal_form) for ok in choice_list])
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            all_cases = ([self.morph.parse(ok) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)
            for j,l in enumerate(pos):
                if choice_list_norm[j] == 'который':
                    kol_kotorii+=1
                    if choice_list_norm[j-1] == ',':
                        for cas_j in all_cases[j]:
                            for cas_z in all_cases[j-2]:
                                if cas_j.tag.gender == cas_z.tag.gender: 
                                    bad_sen2 = True
                        if bad_sen2:
                            bad_sen2 = False
                        else:
                            #print(choice_list[j],choice_list[j-2])  
                            bad_sen2 = True                                                       
                        break                

            for souz in souzs:
                if ', '+souz in choice:
                    bad_sen = True
            if kol_kotorii>1:
                #print('2 который!!')
                pred.append(str(i+1))
            if bad_sen2:
                may_be.append(str(i+1))
            #if bad_sen:
            #    may_be3.append(str(i+1))             
            if bad_sen or kol_kotorii>0 or ', то что' in choice:
                #print('Союзы, который или , то что')
                may_be2.append(str(i+1))       
        #print(pred,may_be,may_be2)
        if not pred:
            pred = may_be 
        if not pred:
            pred = may_be2 
        
        return pred 
    
    def get_predicates1(self, task):
        pred = []
        may_be = []
        may_be1 =  []
        may_be2 = []
        may_be3 = []
        for i, choice in enumerate(task):
            bad_sen = False
            bad_sen1 = False
            bad_sen2 = False
            bad_sen3 = False
            bad_sen0 = False
            kol_kotorii = 0
            choice = ' '+choice.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').lower()+' '
            #choice = re.sub('«.*»','',choice)
            choice_list = choice.replace('»',' »').replace('«','« ').split()
            choice_list_norm = ([str(self.morph.parse(ok)[0].normal_form) for ok in choice_list])
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            case = ([str(self.morph.parse(ok)[0].tag.case) for ok in choice_list])
            number = ([str(self.morph.parse(ok)[0].tag.number) for ok in choice_list])
            gender = ([str(self.morph.parse(ok)[0].tag.gender) for ok in choice_list])
            all_cases = ([self.morph.parse(ok) for ok in choice_list])
            #print(i+1,choice)
            #print(pos)
            id_zap = -1
            podl = False
            est_and = False            
            for j,l in enumerate(pos):
                
                if choice_list[j] == ',':
                    id_zap = j
                    podl = False 
                    est_and = False
                if not podl and (j==0 or pos[j-1] != 'PREP'):
                    for cas_j in all_cases[j]:
                        if (all_cases[j][0].score/cas_j.score<2.5 or cas_j.score>0.1) and (cas_j.tag.POS == 'NOUN' or cas_j.tag.POS == 'NPRO') and cas_j.tag.case == 'nomn':
                            
                            podl = True
                            number[j] = cas_j.tag.number
                            if j+2<len(pos) and choice_list[j+1] =='и' and (pos[j+2] == 'NPRO' or pos[j+2] == 'NOUN' or pos[j+2] == 'ADJF') and choice_list[j+2] !='который' and case[j+2] == 'nomn':
                                number[j] = 'plur'
                                est_and = True
                    if (choice_list[j] =='все' or choice_list[j] =='те') and (pos[j+1] == 'PREP' or choice_list[j+1] ==','):
                        number[j] = 'plur'
                        est_and = True
                        podl = True
                    if (choice_list[j] =='тот' or choice_list[j] =='каждый' or choice_list[j] =='кто') and choice_list[j+1] ==',':
                        number[j] = 'sing'
                        est_and = True
                        podl = True                    
                    if choice_list[j] =='полшколы':
                        number[j] = 'sing'
                        podl = True                        
                    if podl:
                        id_zap_NOUN = j
                        #print('подл',choice_list[j])
                        NOUN_do_zap = False
                        VERB_do_zap = False
                        kol_zap = 0
                        if j+1 !=len(pos) and (pos[j+1] == 'ADJS' and number[j] != number[j+1]):
                            #print('NOUN+ADJS')
                            bad_sen2 = True
                        
                        for z in range(j-1,len(pos)):
                            #if (choice_list[j+1] != ',' and choice_list[z] == ',' and pos[j+1] != 'PRTF' or choice_list[z] == '.'):
                            #    break
                            #print('++',choice_list[z],kol_zap,VERB_do_zap)
                            number_z_plur = False
                            number_z_sing = False
                            if z<0:
                                continue
                            if kol_zap % 2 ==0 and kol_zap>0 and z>j and pos[z]=='CONJ':
                                #print('break союз',choice_list[z])
                                break
                            est_VERB = False
                            if z!=j and (z==0 or pos[z-1] != 'PREP') and (choice_list[z-1] != 'и' or z!=j+2) and (choice_list[z-2] != 'и' or pos[z-1] != 'ADJF' or z!=j+3):
                                #print('++',choice_list[z],kol_zap,VERB_do_zap)
                                for cas_j in all_cases[z]:
                                    if (all_cases[z][0].score/cas_j.score<2 or (cas_j.tag.number == 'plur' and cas_j.score>0.05)) and not 'Name' in cas_j.tag  and  (cas_j.tag.POS == 'NPRO' or  cas_j.tag.POS == 'NOUN') and cas_j.tag.case == 'nomn':
                                        NOUN_do_zap = True
                                        #print('сущ',choice_list[z])
                            if self.morph.parse(choice_list[z])[0].normal_form =='который' or choice_list[z]=='«' or choice_list[z]=='»':
                                NOUN_do_zap = True
                                #print('который')                                
                            
                            if z!=j and kol_zap % 2 ==0 and not VERB_do_zap:
                                for cas_z in all_cases[z]:
                                    if cas_z.tag.POS =='NOUN':
                                        break
                                    if all_cases[z][0].score/cas_z.score<2 and (cas_z.tag.POS =='VERB' or cas_z.tag.POS =='PRTS'):
                                        #print('-',choice_list[z])
                                        est_VERB = True
                                        if cas_z.tag.number == 'sing':
                                            number_z_sing = True
                                        if cas_z.tag.number == 'plur':
                                            number_z_plur = True                                            
                                        #break
                                if est_VERB:
                                    #if NOUN_do_zap:
                                    #    break
                                    #print(choice_list[j],choice_list[z],number[j],number_z_sing,number_z_plur,NOUN_do_zap)
                                    if z>j and gender[j] !='None' and gender[z] !='None' and gender[j]!=gender[z]:
                                        #print('не сходится gender')
                                        bad_sen3 = True
                                        id_bad = z
                                    if not NOUN_do_zap and ((number[j] == 'sing' and not number_z_sing) or (number[j] == 'plur' and not number_z_plur)): 
                                        bad_sen1 = True            
                                        #print('не сходится')
                                        id_bad = z
                                        if est_and:
                                            bad_sen0 = True    
                                            #print(choice_list[z],bad_sen1,bad_sen0,NOUN_do_zap)
                                    VERB_do_zap = True
                                    #if bad_sen1 and not NOUN_do_zap:
                                    #    print('Нашлось') 
                                    #    break
                            if (choice_list[z] == ',' or choice_list[z] == '.' or z+1==len(pos))and bad_sen1 and (z>id_bad+3 or not NOUN_do_zap):
                                #print('break сущ',choice_list[j])  
                                break
                            if choice_list[z] == ',' or choice_list[z] == '.' or z+1==len(pos):
                                bad_sen1 = False
                                NOUN_do_zap = False
                                VERB_do_zap = False
                                if z>j: 
                                    kol_zap+=1
                        if bad_sen1:
                            break  
                if choice_list[j] == 'один' and number[j+1] =='plur':
                    #print('1один+plur')  
                    bad_sen  = True 
                if choice_list[j] == 'один':
                    for z in range(j+1,len(pos)):
                        if choice_list[z] == ',':
                            break
                        if pos[z] == 'VERB' and number[z] =='plur':
                            #print('2один+plur')  
                            bad_sen  = True
                            break
                    for z in range(j-1,-1,-1):
                        if choice_list[z] == ',':
                            break
                        if pos[z] == 'VERB' and number[z] =='plur':
                            #print('3один+plur')  
                            bad_sen  = True
                            break         
            if 'кто , как не' in  choice or 'кто из' in choice or 'кто-то из' in choice   or 'какой из' in choice or 'каждая и' in choice or 'каждый и' in choice or 'какая из' in choice:
                #print('сочетание')
                may_be.append(str(i+1))
                if bad_sen1:
                    bad_sen = True
            #if ' все ' in choice or ' те ' in choice:
            #    #print('все или те')
            #    #may_be.append(str(i+1))                
            if bad_sen or bad_sen0 or bad_sen2:
                pred.append(str(i+1))            
            if bad_sen1 or bad_sen3:
                may_be1.append(str(i+1))
        #print(pred,may_be,may_be1)
        if not pred:
            pred = may_be 
        if not pred:
            pred = may_be1          
        return pred     
    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        question = task['question']
        
        x, choices, conditions = self.parse_task(task)
        all_features = []
        for i,row in enumerate(x):
            row = row.replace('«','« ').replace('»',' »')
            #print(i+1,row)
            all_features.append(self.tokens_features(row))        
        
        pred =[[],[],[],[],[],[],[],[],[],[],[],[]]
        conditions_id =[]
        #print(x)
        #print(choices)
        #print(conditions)    
        for condition in conditions:
            if condition == 'причастный оборот ':
                conditions_id.append(5)
            elif  condition == 'связь подлежащее сказуемое ':
                conditions_id.append(6)
            elif  condition == 'несогласованный приложение ':
                conditions_id.append(3)
            elif  condition == 'форма существительное ':
                conditions_id.append(9)  
            elif  condition == 'косвенный речь ':
                conditions_id.append(2)  
            elif  condition == 'деепричастный оборот ':
                conditions_id.append(1)  
            elif  condition == 'однородный член ':
                conditions_id.append(4)  
            elif  condition == 'сложноподчинённый ':
                conditions_id.append(7)  
            elif  condition == 'сложный ':
                conditions_id.append(11)  
            elif  condition == 'соотнесённость глагольный форма ':
                conditions_id.append(8)  
            elif  condition == 'числительное ':
                conditions_id.append(10)
            else:
                conditions_id.append(0)
        #print(conditions_id)    
        pred[1] = self.get_gerund1(x)
        #print('Деепричастие',pred[1])
        pred[2] = get_indirect_speech(all_features)
        #print('Косвеная речь',pred[2])   
        pred[3] =  self.get_app1(x)
        #print('несогласованный приложение',pred[3])
        pred[4] = self.get_homogeneous1(x)
        #print('однородный член',pred[4])
        pred[5] = self.get_participle1(x)
        #print('причастный оборот',pred[5]) 
        pred[6] = self.get_predicates1(x)
        #print('связь подлежащее сказуемое',pred[6])           
        pred[7] = self.get_podch1(x)
        #print('сложноподчинённый',pred[7]) 
        pred[8] = self.get_verbs1(x)
        #print('соотнесённость глагольный форма',pred[8])
        pred[9] = self.get_nouns1(x)
        #print('форма существительное',pred[9])
        pred[10] = self.get_numerals1(x)
        #print('числительное',pred[10])
        pred[11] = self.get_clause1(x)
        #print('сложный',pred[11])
        #if 'деепричастный оборот ' in conditions:
        #    pred[1] = self.get_gerund1(x)
        #if 'причастный оборот ' in conditions:
        #    pred[5] = self.get_participle1(x)       
        #if 'однородный член ' in conditions:
        #    pred[4] = self.get_homogeneous1(x)       
        #if 'связь подлежащее сказуемое ' in conditions:
        #    pred[6] = self.get_predicates1(x)       
        #if 'косвенный речь ' in conditions:
        #    pred[2] = get_indirect_speech(all_features)       
        #if 'форма существительное ' in conditions:
        #    pred[9] = self.get_nouns1(x)
        #answer = task['solution']['correct']
        result = []
        all_id = ['1','2','3','4','5','6','7','8','9']
        i = 0
        for condition, key in zip(conditions_id, ["A", "B", "C", "D", "E"]):
            result.append(pred[condition])
            #if conditions[i] == 'форма существительное ':
            #print( conditions[i],key, pred[condition] , answer[key])
            i+=1
        pred_dict = {}
        #print(result)
        for __ in range(5):
            for _ in range(10):
                for condition, key in zip(result, ["A", "B", "C", "D", "E"]):
                    if len(condition) == 1:
                        zzz = condition[0]
                        pred_dict[key] = zzz
                        for i in range(len(result)):
                            if zzz in result[i]:
                                result[i].remove(zzz)
                            if zzz in all_id:
                                all_id.remove(zzz)
                        break
            #print(pred_dict)
            #print(result)
            for condition, key in zip(result, ["A", "B", "C", "D", "E"]):
                if len(condition) > 1:
                    zzz = random.choice(condition)
                    #
                    pred_dict[key] = zzz
                    result[result.index(condition)] = []
                    for i in range(len(result)):
                        if zzz in result[i]:
                            result[i].remove(zzz)
                        if zzz in all_id:
                            all_id.remove(zzz)    
                    break
            #print(pred_dict)
            #print(result)
            if len(pred_dict)==5:
                break
        #print(pred_dict)
        #print(all_id)
        
        for key in ["A", "B", "C", "D", "E"]:
            if not key in pred_dict:
                pred_dict[key] = random.choice(all_id)
                #pred_dict[key] = 0
        fin_dict = {}
        for condition, key in zip(sorted(pred_dict.keys()), ["A", "B", "C", "D", "E"]):
            fin_dict[key] = pred_dict[condition]
        
               
        return fin_dict