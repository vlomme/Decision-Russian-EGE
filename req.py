import json
import requests
from threading import Thread
from ast import literal_eval as le
symm = 0
ta = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
tb = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
#train/
#range(1,136)
exam_ticket = []
n = -1
nn= -1
with open('1.csv', encoding='utf-8') as fin:
    for i, lines in enumerate(fin):
        #if nn<4810:
        #    if lines[0] == '|' or lines[0] == '-':
        #        nn+=1
        #    continue
        if not lines or len(lines)<2 or lines.isspace():
            continue
        lines = lines.strip()
        if lines[0] == '!':
            n +=1
            #if n>50:
            #    break    
            exam_ticket.append({})
            exam_ticket[n]["id"] = str(n)
            exam_ticket[n]["text"] = lines[1:]
            exam_ticket[n]["question"] = {}
        elif lines[0] == '+': 
            exam_ticket[n]["text"] += '\n'+lines[1:]
        elif lines[0] == '#': 
            if not "left" in exam_ticket[n]["question"]:
                exam_ticket[n]["question"]["left"] = []
            exam_ticket[n]["question"]["left"].append({"id":'A',"text": lines[1:]})
        elif lines[0] == '*': 
            if not "choices" in exam_ticket[n]["question"]:
                exam_ticket[n]["question"]["choices"] = []
                id_a = 1
            exam_ticket[n]["question"]["choices"].append({"id": str(id_a),"text": lines[1:]})
            id_a+=1
        elif lines[0] == '?': 
            exam_ticket[n]["question"]["type"] = lines[1:]
        elif lines[0] == '-':
            exam_ticket[n]["solution"] = {}
            exam_ticket[n]["solution"]["correct"] = le(lines[1:])
        elif lines[0] == '|':
            exam_ticket[n]["solution"] = {}
            exam_ticket[n]["solution"]["correct_variants"] = []
            lines = lines[1:].split('|')
            #data = json.loads('{"a": "ex1", "b": "ex2", "c": "ex3"}')
            for line in lines:
                if line[0] == '[' or line[0] == '{':
                    exam_ticket[n]["solution"]["correct_variants"].append(le(line))
                else:
                    exam_ticket[n]["solution"]["correct_variants"].append(line)
        else:
            if not "choices" in exam_ticket[n]["question"]:
                exam_ticket[n]["question"]["choices"] = []
                id_a = 1
            exam_ticket[n]["question"]["choices"].append({"id": str(id_a),"text": lines})
            id_a+=1
        #if not "type" in exam_ticket[n]["question"]:
        #    print(exam_ticket[n])
#print(exam_ticket)
if "tasks" in exam_ticket:
    yyy = exam_ticket["tasks"]
    if isinstance(yyy, dict):
        if "tasks" in yyy.keys():
            yyy = yyy["tasks"]
else:
    yyy = exam_ticket
#print('-'*30)
#print(yyy)    
requests.get('http://localhost:8000/ready')
resp = requests.post('http://localhost:8000/take_exam', json=exam_ticket)
xxx = resp.json()['answers']

s = 0
x = 0

#print('-'*30)
#print(xxx)
print('-'*30)
for i in range(len(yyy)):
    score = 1

    #print(yyy[i])
    if ('correct' in yyy[i]['solution']):
        zzz = (yyy[i]['solution']['correct'])
        if type(zzz) is dict:
            for k, v in zzz.items():
                if xxx[str(i)] != '0':
                    x = x + score
                if k in xxx[str(i)] and xxx[str(i)][k] == v:
                    
                    s = s + 1
        elif (xxx[str(i)]==zzz):
            if xxx[str(i)] != '0':
                x = x + score            
            s = s + score

    if ('correct_variants' in yyy[i]['solution']):
        if xxx[str(i)] != '0':
            x = x + score
        zzz = yyy[i]['solution']['correct_variants'][0]
        for variants in yyy[i]['solution']['correct_variants']:
            if (xxx[str(i)]==variants):
                zzz = variants
                s = s + score
                break

    #print (zzz)
    if xxx[str(i)] != '0' and (xxx[str(i)])!=(zzz):
        print(yyy[i]["text"])
        print('-'*30)
        if "choices" in yyy[i]["question"]:
            for ch in yyy[i]["question"]["choices"]:
                print(ch["text"])
        if "left" in yyy[i]["question"]:
            for ch in yyy[i]["question"]["left"]:
                print(ch["text"])
        print (i, ':', xxx[str(i)],'|----|', zzz,str(xxx[str(i)])==str(zzz))
        print('='*30)


print('ИТОГО',s,'|',x)
"""    
for i in range(0,26):
    if (len(ta[i]))>0:
        print(i,len(ta[i]),ta[i])
print('ИТОГО ЗА ВСЁ',symm)
for i in range(0,26):
    if (len(tb[i]))>0:
        print(i,len(tb[i]),tb[i])"""
            
