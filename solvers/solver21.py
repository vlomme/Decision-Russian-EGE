import re
import pymorphy2
class Solver():

    def __init__(self, seed=42, path_to_model="data/models/siameise_model.h5"):
        self.morph = pymorphy2.MorphAnalyzer()
    def fit(self, tasks):
        pass    
    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        test_sentence_pairs = []
        choices, label = self.parse_task(task)
        #print(choices)
        #print(label)
        choices_dict = {}
        for n, choice in enumerate(choices, 1):
            choices_dict[choice] = n

        if label == 'тире':
            hypothesys = self.dash_task(choices)
            result = self.dash_pred(hypothesys)
        elif label == 'запятая':
            hypothesys = self.comma_task(choices)
            result = self.comma_pred(hypothesys)
        else:
            hypothesys = self.semicolon_task(choices)
            result = self.semicolon_pred(hypothesys)
        #print(hypothesys)
        return result
    def comma_pred(self, choices):
        pred =[[],[],[],[],[],[],[],[],[],[],[],[]]
        
        for i, choice in choices.items():
            
            choice_list = choice.split()
            #print(i,choice)
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            for j,l in enumerate(choice_list):
                if l == ',':
                    pos_dash = j 
                    break
            #print(pos) 
            kol_zap = 0
            est_none = False
            deepri  = False
            glag_do, glag_after = False,False
            dv = False
            for j,l in enumerate(pos):
                if (l == 'PRCL' or choice_list[j] =='-' or choice_list[j] =='.' or choice_list[j] ==':') and j>pos_dash:
                    est_none = True
                #if (choice_list[j]==',') and j > pos_dash and not est_none:
                #    kol_zap +=1
                if l == 'VERB' and (j>pos_dash and not est_none):
                    glag_after=True
                if l == 'VERB' and j<pos_dash:
                    glag_do=True 
            if 'GRND' in pos:        
                deepri = True
            if '-' in choice_list or ':' in choice_list:        
                dv = True            
            
            
            if deepri:
                pred[3].append(str(i))
            elif glag_do and glag_after:
                pred[4].append(str(i))
            #elif (pos[pos_dash-1]=='NOUN' or pos[pos_dash-1]=='NPRO') and (pos[pos_dash+1]=='PRTF' or pos[pos_dash+1]=='PRTS'):
            elif (pos[pos_dash+1]=='PRTF' or pos[pos_dash+1]=='PRTS') or (pos[pos_dash+2]=='PRTF' or pos[pos_dash+2]=='PRTS')or (pos[pos_dash+3]=='PRTF' or pos[pos_dash+3]=='PRTS') or choice_list[pos_dash+1] =='вызывающей':
                pred[5].append(str(i))
            elif dv:
                pred[2].append(str(i))
            elif choice_list[pos_dash+1] =='как':
                pred[6].append(str(i))                
            elif pos[pos_dash+1]=='CONJ':
                pred[1].append(str(i))

            else:
                pred[0].append(str(i))                
        #print(pred)
        index_result = 0
        for j,l in enumerate(pred):
            if len(l)>1:
                index_result = j
                #break    
        return pred[index_result] 

    
    def semicolon_pred(self, choices):    
        pred =[[],[],[],[],[],[],[],[],[],[],[],[]]
        
        for i, choice in choices.items():
            
            choice_list = choice.split()
            #print(i,choice)
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            for j,l in enumerate(choice_list):
                if l == ':':
                    pos_dash = j 
                    break
            #print(pos)            
            kol_zap = 0
            many_zap = False
            est_none  = False
            glag_after,glag_do = False,False
            
            for j,l in enumerate(pos):
                if (l == 'PRCL' or choice_list[j] =='-' or choice_list[j] =='.') and j>pos_dash:
                    est_none = True
                if (choice_list[j]==',' or choice_list[j]=='и') and j > pos_dash and not est_none:
                    kol_zap +=1                
                if l == 'VERB' and (j>pos_dash and not est_none):
                    glag_after=True
                if l == 'VERB' and j<pos_dash:
                    glag_do=True              
            if kol_zap>1:
                many_zap = True
            if '»' in choice and (': «' in choice):
                #2--При прямой речи------------------------------------
                pred[2].append(str(i))
            elif glag_do and glag_after:
                pred[0].append(str(i))
            elif  many_zap or 'всё' in choice.lower():
                #1--Много запятых------------------------------------
                pred[1].append(str(i))
            else:
                pred[0].append(str(i))
        #print(pred)
        index_result = 0
        for j,l in enumerate(pred):
            if len(l)>1:
                index_result = j
                #break    
        return pred[index_result]        
    def dash_pred(self, choices):    
        pred =[[],[],[],[],[],[],[],[],[],[],[],[]]
        
        for i, choice in choices.items():
            pos_dash = 1
            choice_list = choice.split()
            #print(i,choice)
            pos = ([str(self.morph.parse(ok)[0].tag.POS) for ok in choice_list])
            for j,l in enumerate(choice_list):
                if l == '-':
                    pos_dash = j 
                    break
            #print(pos)            
            #if pos_dash<1:
            #    pos_dash2 = 0
            #else:
            #    pos_dash2 = pos_dash-1
            #0------------Иначе------------------
            #1----------приложение, имеющее пояснительный характер-------------------------
            #2----------между подлежащим и сказуемым(без глаголов)-------------------------
            #3-----------в бессоюзном сложном предложении------------------------
            #4-----------в сложносочинённом предложении------------------------
            #5-----------однородных членов перед обобщающим словом------------------------
            #6-----------в неполном предложении------------------------
            #7-----------между подлежащим и сказуемым, инфинитива.------------------------
            #8-----------Много тире------------------------
            #9-----------для обозначения количественных пределов------------------------
            #10---------------При прямой речи--------------------
            #11-----------------------------------
            
            
            
            glag_all,est_none,glag_after,glag_do,infn_after,infn_do,prtf_all = False,False,False,False,False,False,False
            prtf_do,  prtf_after = False,False
            infn_do2,many_zap,many_dash,many_prep = False,False,False,False
            prep,conj,kol_dash,kol_zap = 0,0,0,0
            for j,l in enumerate(pos):
                if (choice_list[j]==':' or choice_list[j]=='.') and j<pos_dash:
                    glag_all,est_none,glag_after,glag_do,infn_after,infn_do,prtf_all = False,False,False,False,False,False,False
                    kol_zap = 0
                if (choice_list[j]==',') and j<pos_dash:
                    glag_all,est_none,glag_after,glag_do,infn_after,infn_do,prtf_all = False,False,False,False,False,False,False
                 
                
                if choice_list[j]=='-':
                    kol_dash +=1
                if choice_list[j]==',' and j<pos_dash:
                    kol_zap +=1    
                if (l == 'PRTF' or l == 'PRTS') and j<pos_dash:
                    prtf_all = True   
                if l == 'None' and j>pos_dash and choice_list[j]!='-':
                    est_none = True
                if l == 'VERB' and (j>pos_dash and not est_none):
                    glag_after=True
                if l == 'INFN' and (j>pos_dash and not est_none):
                    infn_after=True
                if (l == 'PRTF' or l == 'PRTS') and (j>pos_dash and not est_none):
                    prtf_after=True                    
                if l == 'INFN' and j<pos_dash:
                    infn_do=True
                    infn_do2=True
                if l == 'VERB' and j<pos_dash:
                    glag_do=True                    
                if (l == 'PRTF' or l == 'PRTS') and j<pos_dash:
                    prtf_do=True                 
                if l == 'VERB' and not est_none:
                    glag_all=True
                if l == 'PREP' and j<pos_dash:
                    prep+=1
                if l == 'CONJ' and j<pos_dash:
                    conj+=1                     
            if prep>2 or conj>2 :
                many_prep = True
            #if kol_dash>1:
            #    many_dash = True
            if kol_zap>2:
                many_zap = True                
                
            #print(i,'Всего', glag_all,'до',glag_do,'после',glag_after,'инф до',infn_do,'инф после',infn_after)    
            
            if '»' in choice and ('! -' in choice or ', -' in choice or '? -' in choice or '» -' in choice ):
                #10--При прямой речи------------------------------------
                pred[10].append(str(i))
            elif many_dash:
                pred[8].append(str(i))
            elif many_zap or choice_list[pos_dash+1] =='эти':
                pred[5].append(str(i))                
            elif pos[pos_dash-1] =='NUMR' and pos[pos_dash+1] =='NUMR':
                pred[9].append(str(i))
            elif (not glag_all and not prtf_all and (infn_do == infn_after)):
                #-----------------Нет глаголов------------------------------------
                if many_prep:
                    #5--однородных членов перед обобщающим словом.
                    pred[5].append(str(i))
                else:
                    #2--между подлежащим и сказуемым, которые выражены формами именительного падежа.
                    pred[2].append(str(i))
                #3--

            elif ((infn_after or glag_after or prtf_after or pos[pos_dash+1] =='NPRO') and (infn_do or glag_do or prtf_do )):
                #-----------------глагол с обоих сторон------------------------------------

                
                if pos[pos_dash+1] =='CONJ':
                    #4--Тире в сложносочинённом предложении
                    pred[4].append(str(i))
                else:
                    #3--в бессоюзном сложном предложении
                    pred[3].append(str(i))
            elif not glag_all and (infn_do2 or infn_after):
                pred[7].append(str(i))
            elif (not glag_after and not infn_after or not infn_do and not glag_do ):
                #-----------------Нет глаголов справа------------------------------------

                
                if choice_list[pos_dash-2] =='а' or choice_list[pos_dash-2] ==',':
                    #6-----------в неполном предложении------------------------
                    pred[6].append(str(i))
                
                else:   
                    #1--приложение, имеющее пояснительный характер.
                    pred[1].append(str(i))
            elif not glag_after and infn_after or infn_do and not glag_do:
                    #7--между подлежащим и сказуемым, инфинитива.
                    pred[7].append(str(i))                
            else:
                pred[0].append(str(i))
        #print(pred)
        index_result = 0
        for j,l in enumerate(pred):
            if len(l)>1:
                index_result = j
                #break    
        return pred[index_result]
    
    
    
    def parse_task(self, task):
        """ link multiple_choice """
        assert task["question"]["type"] == "multiple_choice"
        
        choices = task["question"]["choices"]

        links, label = [], ""
        description = task["text"].replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;')
        description = description.replace('»',' »').replace('«','« ')
        if "двоеточие" in description:
            label = "двоеточие"
        if "тире" in description:
            label = "тире"
        if "запят" in description:
            label = "запятая"

        m = re.findall("[0-9]\\)", description)
        for n, match in enumerate(m, 1):
            first, description = description.split(match)
            if len(first) > 1 and "Найдите" not in first:
                links.append(first)
                if n == len(m):
                    description = description.split('\n')[0]
                    links.append(description.replace(' (', ''))

        assert len(links) == len(choices)

        return links, label
    def dash_task(self, choices):
        hypothesys = {}

        for i, choice in enumerate(choices):
            if ' -' in choice or ' −' in choice:
                hypothesys[i+1] = choice
        return hypothesys

    def semicolon_task(self, choices):
        hypothesys = {}
        for i, choice in enumerate(choices):
            if ':' in choice:
                hypothesys[i+1] = choice
        return hypothesys

    def comma_task(self, choices):
        hypothesys = {}
        for i, choice in enumerate(choices):
            if ', ' in choice:
                hypothesys[i+1] = choice
        return hypothesys