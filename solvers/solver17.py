import re
from solvers.utils import Bert2
import random
class Solver(Bert2):
    def __init__(self, seed=42, train_size=0.85):
        super(Solver, self).__init__()
        print("h")


    def load(self, path="data/models/solver17.pkl"):
        print("h")


    def predict_from_model(self, task):
        words = task['text'].split("\n")
        words = ' '.join(words)
        words = re.sub('\([0-9]*?\)',"[MASK]",words)
        #sentence = words.replace(' [MASK] ','[MASK]'); sentence = sentence.replace('[MASK] ','[MASK]'); sentence = sentence.replace(' [MASK]','[MASK]')  # удаляем лишние пробелы
        #sentence = sentence.split('[MASK]')        
        result = self.what_mask(words)
        if not result:
            result = [str(random.choice(range(5))+1)]
        return result
        