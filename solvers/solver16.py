import re
from solvers.utils import Bert2
import pymorphy2
import random
class Solver(Bert2):
    def __init__(self, seed=42, train_size=0.85):
        super(Solver, self).__init__()
        print("h")


    def load(self, path="data/models/solver17.pkl"):
        print("h")


    def predict_from_model(self, task):
        #print('-'*40,task['id'],'-'*40)
        words_all = task["question"]["choices"]
        par_b = [['стар','млад'],  ['жив','мертв'],  ['день','ночь']]
        ans = []
        morph = pymorphy2.MorphAnalyzer()
        for i,words in enumerate(words_all):
            #print('-'*80)
            #words = re.sub(' '," [MASK] ",words)
            words = words["text"].strip().lower()
            #print(words)
            words = re.sub('[0-5][\)\.] ','',words)
            choice_list = words.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').replace('»',' »').replace('«','« ').split()
            #print(choice_list)
            pos = ([str(morph.parse(ok)[0].tag.POS) for ok in choice_list])
            case = ([str(morph.parse(ok)[0].tag.case) for ok in choice_list])
            all_cases = ([morph.parse(ok) for ok in choice_list])
            for j, p in enumerate(pos):            
                eto_NOUN = False
                bad_par = False
                if p == 'PRTF' and (j==0 or pos[j-1] !='CONJ'):
                    choice_list[j] = '[MASK] '+choice_list[j] 
                if p == 'VERB':
                    for iii in range(j+1,min(len(pos),j+12)):
                        if pos[iii] == 'NOUN':
                            for all_cases_z in all_cases[iii]:
                                if  all_cases_z.tag.case =='nomn':
                                    #eto_NOUN = True
                                    eto_NOUN = False
                                    break
                            if eto_NOUN:
                                break
                        if pos[iii] =='VERB' and pos[iii-1] !='CONJ':
                            #print('Глагол',choice_list[j],choice_list[iii])
                            choice_list[iii] = '[MASK] '+choice_list[iii]  
                if p == 'INFN':
                    for iii in range(j+1,min(len(pos),j+5)):
                        if pos[iii] =='INFN' and pos[iii-1] !='CONJ':
                            #print('Инфинитив',choice_list[j],choice_list[iii])
                            choice_list[j] = choice_list[j]+ ' [MASK]'                
                if p == 'ADVB':
                    for iii in range(j+1,min(len(pos),j+3)):
                        if pos[iii] =='ADVB' and pos[iii-1] !='CONJ':
                            #print('Наречие',choice_list[j],choice_list[iii])
                            choice_list[j] = choice_list[j]+ ' [MASK]'
                if p == 'ADJF':
                    for iii in range(j+1,min(len(pos),j+4)):
                        if pos[iii] =='ADJF' and pos[iii-1] !='CONJ':
                            if case[j] ==  case[iii]:
                                #print('Прилогательное',choice_list[j],choice_list[iii])
                                choice_list[j] = choice_list[j]+ ' [MASK]'
                                break
                eto_NOUN = False
                for all_cases_j in all_cases[j]:
                    if all_cases_j.score> 0.05 and all_cases_j.tag.POS == 'NOUN':
                        eto_NOUN = True
                if eto_NOUN:
                    odnorod = False
                    NOUN_est_ADJF = False
                    for ii in range(j+1,len(pos)):
                        if pos[ii] =='NOUN':
                            for all_cases_j in all_cases[j]:
                                for all_cases_z in all_cases[ii]:
                                    if all_cases_z.tag.POS == 'ADJF' or all_cases_j.tag.POS == 'ADJF':
                                        NOUN_est_ADJF = True
                                    #if all_cases_j.score> 0.24 and all_cases_z.score> 0.24 and all_cases_j.tag.case == all_cases_z.tag.case  and all_cases_j.tag.number == all_cases_z.tag.number:
                                    if (all_cases_j.tag.case == all_cases_z.tag.case) and (choice_list[j] != 'свет' or choice_list[ii] != 'заря'):
                                        odnorod = True
                            if odnorod:
                                if not NOUN_est_ADJF:
                                    #print('Сущ',choice_list[j],choice_list[ii])
                                    choice_list[j] = choice_list[j]+ ' [MASK]'
                                break    
                        
                        
                        if pos[ii] !='NOUN' and pos[ii] !='ADJF' and pos[ii] !='PREP' and pos[ii] !='PRCL':
                            break            
                
                if  (p == 'CONJ'or choice_list[j] =='да' or choice_list[j] =='ни')  and j>0 and pos[j-1] != 'CONJ':
                    if choice_list[j] =='то' and choice_list[j-1] =='не':
                            #print('Союз',choice_list[j-1],choice_list[j])
                            choice_list[j-1] = '[MASK] '+choice_list[j-1]  
                          
                    elif choice_list[j] !='ни' or (choice_list[j] =='ни' and (j==0 or choice_list[j-1] !='свет')and (j==len(pos) or choice_list[j+1] !='свет')):
                        #print('Союз',choice_list[j],choice_list[j-1],choice_list[j+1])
                        for pb in par_b:
                            if (j!=0 and choice_list[j-1] ==pb[0]) and (j!=len(pos) and choice_list[j+1] ==pb[1]):
                                bad_par = True
                        if j==1 and choice_list[j] == 'и':
                            bad_par = True
                        if choice_list[j] == 'ни' and (j!=len(pos) or choice_list[j+1] =='то'):
                            bad_par = True                       
                        if not bad_par:  
                            choice_list[j] = '[MASK] '+choice_list[j]
                       
            words = ' '.join(choice_list).replace(' ,',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' :',':').replace(' ;',';').replace(' »','»').replace('« ','«')
            words = words.replace('[MASK] [MASK]','[MASK]').replace('[MASK] [MASK]','[MASK]')
            #print(words)
            result = self.what_mask(words)
            #print(result)
            if len(result)==1:
                ans.append(str(i+1))       
        #print(ans)
        if not ans:
            ans = ['1', '2']
        elif len(ans) ==1:
            if ans[0] != '5':
                ans.append('5')
            else:
                ans = ['1', '5']
        else:
            random.shuffle(ans)
            ans = sorted(ans[:2])
        return ans
        