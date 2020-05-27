import random
from collections import defaultdict
import numpy as np
import json
from ast import literal_eval as le
from utils import *
from solvers import *


import re
import traceback
import time
start_loop1 = time.time()
solver_param = defaultdict(dict)
solver_param[17]["train_size"] = 0.9
solver_param[18]["train_size"] = 0.85
solver_param[19]["train_size"] = 0.85
solver_param[20]["train_size"] = 0.85

class CuttingEdgeStrongGeneralAI(object):

    def __init__(self, train_path='public_set/train'):
        self.train_path = train_path
        self.classifier = classifier.Solver()
        solver_classes = [
            solver1,
            solver2,
            solver3,
            solver4,
            solver5,
            solver6,
            solver7,
            solver8,
            solver9,
            solver10,
            solver10,
            solver10,
            solver13,
            solver14,
            solver15,
            solver16,
            solver17,
            solver17,
            solver17,
            solver17,
            solver21,
            solver22,
            solver23,
            solver24,
            solver25,
            solver26
        ]
        #solver_classes = [solver4]
        self.solvers = self.solver_loading(solver_classes)
        self.clf_fitting()

    def solver_loading(self, solver_classes):
        solvers = []
        start_loop = time.time()
        for i, solver_class in enumerate(solver_classes):
            solver_index = i + 1
            train_tasks = load_tasks(self.train_path, task_num=solver_index)
            solver_path = os.path.join("data", "models", "solver{}.pkl".format(solver_index))
            solver = solver_class.Solver(**solver_param[solver_index])
            try:
                solver.fit(train_tasks)
            except Exception as e:
                print('Exception during fitting: {}'.format(e))            
            """
            if os.path.exists(solver_path):
                print("Loading Solver {}".format(solver_index))
                solver.load(solver_path)
            else:
                print("Fitting Solver {}...".format(solver_index))
                try:
                    print("Fitting Solver")
                    #solver = solver_class.Solver(**solver_param[solver_index])
                    solver.fit(train_tasks)
                    solver.save(solver_path)
                except Exception as e:
                    print('Exception during fitting: {}'.format(e))
            """        
            print("Solver {} is ready!\n".format(solver_index))
            
            solvers.append(solver)
            print(time.time() - start_loop)
        return solvers

    def clf_fitting(self):
        tasks = []
        for filename in os.listdir(self.train_path):
            if filename.endswith(".json"):
                data = read_config(os.path.join(self.train_path, filename))
                tasks.append(data)
        print("Fitting Classifier...")
        self.classifier.fit(tasks)
        print("Classifier is ready!")
        return self

    def not_so_strong_task_solver(self, task):
        question = task['question']
        if question['type'] == 'choice':
            # pick a random answer
            choice = random.choice(question['choices'])
            answer = choice['id']
        elif question['type'] == 'multiple_choice':
            # pick a random number of random choices
            min_choices = question.get('min_choices', 1)
            max_choices = question.get('max_choices', len(question['choices']))
            n_choices = random.randint(min_choices, max_choices)
            random.shuffle(question['choices'])
            answer = [
                choice['id']
                for choice in question['choices'][:n_choices]
            ]
        elif question['type'] == 'matching':
            # match choices at random
            random.shuffle(question['choices'])
            answer = {
                left['id']: choice['id']
                for left, choice in zip(question['left'], question['choices'])
            }
        elif question['type'] == 'text':
            if question.get('restriction') == 'word':
                # pick a random word from the text
                words = [word for word in task['text'].split() if len(word) > 1]
                answer = random.choice(words)

            else:
                answer = ('=(')

        else:
            raise RuntimeError('Unknown question type: {}'.format(question['type']))

        return answer

    def take_exam(self, exam):
        answers = {}
        # pprint.pprint(exam)
        if "tasks" in exam:
            variant = exam["tasks"]
            if isinstance(variant, dict):
                if "tasks" in variant.keys():
                    variant = variant["tasks"]
        else:
            variant = exam
        task_number = self.classifier.predict(variant)
        for i,task_n in enumerate(task_number):
            if "left" in variant[i]["question"] and len(variant[i]["question"]["left"]) == 4:
                task_number[i] = 26
            if "left" in variant[i]["question"] and len(variant[i]["question"]["left"]) == 5:
                task_number[i] = 8                
        #print("Classifier results: ", task_number)
        for i, task in enumerate(variant):
            #if (int(task_number[i])):
            task_id = task['id']
            task_index, task_type = i + 1, task["question"]["type"]
            try:
                prediction = self.solvers[task_number[i] - 1].predict_from_model(task)
                #print("Prediction: ", prediction)
            except Exception as e:
                print(traceback.format_exc())
                prediction = self.not_so_strong_task_solver(task)
            if isinstance(prediction, np.ndarray):
                prediction = list(prediction)
            answers[task_id] = prediction
            #else:
            #    answers[task['id']] = '0'
        return answers





ai = CuttingEdgeStrongGeneralAI()

if __name__ == '__main__':
    exam_ticket = []
    n = -1
    with open('task.csv', encoding='utf-8') as fin:
        for i, lines in enumerate(fin):
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

    if "tasks" in exam_ticket:
        yyy = exam_ticket["tasks"]
        if isinstance(yyy, dict):
            if "tasks" in yyy.keys():
                yyy = yyy["tasks"]
    else:
        yyy = exam_ticket
 
    xxx = ai.take_exam(exam_ticket)

    s = 0
    x = 0


    print('-'*30)
    for i in range(len(yyy)):
        score = 1

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

            
   
    
    
    
