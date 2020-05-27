import os
import random
from functools import wraps
from abc import ABC, abstractmethod
import pickle
import torch
import ufal.udpipe
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertForMaskedLM
import difflib
import numpy as np
import re



def singleton(cls):
    instance = None

    @wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return inner


class AbstractSolver(ABC):
    def __init__(self, seed=42):
        self.seed = seed
        self._init_seed()

    def _init_seed(self):
        random.seed(self.seed)

    def fit(self, tasks):
        pass

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            temp = pickle.load(f)
        assert isinstance(temp, cls)
        return temp

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def predict_from_model(self, task):
        pass



class Bert2(object):
    def __init__(self):
        #self.model_file = "./data/pytorch.tar.gz"
        self.model_file = "./data/bert60.tar.gz"
        self.vocab_file = "./data/vocab_2.txt"
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()

    @singleton
    def bert_model(self):
        model = BertForMaskedLM.from_pretrained(self.model_file).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=False)
        return tokenizer
    
    def what_mask(self, text):
        w = self.tokenizer.tokenize(',')
        w_i = self.tokenizer.convert_tokens_to_ids(w)
        w = self.tokenizer.tokenize('^')
        w_j = self.tokenizer.convert_tokens_to_ids(w)        
        #print(text)
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        #print(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predictsx1 = []
        for i in range(len(mask_input)):
            predictsx1.append(predictions[0,mask_input[i],:])
            predicts1 = predictsx1[i].argsort()[-8:].numpy()
            out1 = self.tokenizer.convert_ids_to_tokens(predicts1)
            #print(out1)
        #print(predictsx1[0].numpy().shape)
        output = []
        a=len(mask_input)
        for i in range(a):
            if predictsx1[i][w_i] > predictsx1[i][w_j]:
                output.append(str(i+1))
        """b = 0
        for i in range(a):
            b += predictsx1[i][w_i]*predictsx1[i][w_i]
        if a > 0:
            c=b/a
        else:
            c= 0
        for i in range(a):
            #print(predictsx1[i][w_i],c)
            if (predictsx1[i][w_i]*predictsx1[i][w_i]>c):
                output.append(str(i+1))"""
        return output

class Bert(object):
    """
    Embedding Wrapper on Bert Multilingual Cased
    """

    def __init__(self):
        self.model_file = "./data/ru_conversational_cased_L-12_H-768_A-12.tar.gz"
        self.vocab_file = "./data/vocab_2.txt"
        #self.model_file = "./data/bert-base-multilingual-cased.tar.gz"
        #self.vocab_file = "./data/bert-base-multilingual-cased-vocab.txt"        
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()

    @singleton
    def bert_model(self):
        model = BertForMaskedLM.from_pretrained(self.model_file).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=False)
        return tokenizer
           
    def token_embedding(self, token_list):
        token_embedding = []
        for token in token_list:
            ontoken = self.tokenizer.tokenize(token)
            segments_ids, indexed_tokens = [1] * len(ontoken), self.tokenizer.convert_tokens_to_ids(ontoken)
            segments_tensors, tokens_tensor = torch.tensor([segments_ids]), torch.tensor([indexed_tokens])
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            ontoken_embeddings = []
            for subtoken_i in range(len(ontoken)):
                hidden_layers = []
                for layer_i in range(len(encoded_layers)):
                    vector = encoded_layers[layer_i][0][subtoken_i]
                    hidden_layers.append(vector)
                ontoken_embeddings.append(hidden_layers)
            cat_last_4_layers = [torch.cat((layer[-4:]), 0) for layer in ontoken_embeddings]
            token_embedding.append(cat_last_4_layers)
        token_embedding = torch.stack(token_embedding[0], 0) if len(token_embedding) > 1 else token_embedding[0][0]
        return token_embedding 
    def what_mask27(self, text):
        if text[-1]==']':
            text = text+' . .'
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        segments_ids = [0] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
            predictsx1 = predictions[0,mask_input[0],:]
            predicts1 = predictsx1.argsort()[-8:].numpy()
            out1 = self.tokenizer.convert_ids_to_tokens(predicts1)
        return out1[0]
        
    def what_mask15(self, text,w_x,delta):
        #print(text)
        w_y = []
        for i in range(len(w_x)):
            w_y.append(self.tokenizer.tokenize(w_x[i].lower()))
            w_y[i] = self.tokenizer.convert_tokens_to_ids(w_y[i])        
        if text[-1]==']':
            text = text+' . .'
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        segments_ids = [0] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predictsx1,predictsx2,predictsx3 = [],[],[]
        for i in range(len(mask_input)):
            predictsx1.append(predictions[0,mask_input[i],:].numpy()+10)
            predictsx2.append(predictions[0,mask_input[i]+1,:].numpy()+10)
            predictsx3.append(predictions[0,mask_input[i]+2,:].numpy()+10)

        ver_w_y2 = 0
        ver_w_y,ver_w2,ver_w3 = [],[],[]
        output = []
        #print(w_x)
        #print(w_y)
        for i in range(len(w_x)):
            if len(w_y[i])>2:
                ver_w_y.append(abs(predictsx1[i][w_y[i][0]]*predictsx2[i][w_y[i][1]]*predictsx3[i][w_y[i][2]])**(1/3))
            elif len(w_y[i])>1:
                ver_w_y.append(abs(predictsx1[i][w_y[i][0]]*predictsx2[i][w_y[i][1]])**(1/2))
            else:
                ver_w_y.append(predictsx1[i][w_y[i][0]])  
            if  'ковану' in w_x[i] or 'кованы' in w_x[i]  or  'золочёно' in w_x[i].replace('е','ё') or  'золочёны' in w_x[i].replace('е','ё') or  'золочёну' in w_x[i].replace('е','ё'):
                ver_w_y[i] +=5
        for i in range(len(w_x)//2):
            ver_w2.append(ver_w_y[i] / ver_w_y[len(w_x)//2 +i])
            #print(ver_w2[i], w_x[i],ver_w_y[i], w_x[len(w_x)//2 +i],ver_w_y[len(w_x)//2 +i])
        max1 = 0
        mem1 = ''
        #print(delta)
        for i in range(len(ver_w2)):    
            if ver_w2[i]>1+delta:
                output.append(w_x[i])
            else:
                output.append(w_x[len(w_x)//2 +i])
        return output
        
    def what_mask14(self, text,w_x):
        w_y = []
        for i in range(len(w_x)):
            w_y.append(self.tokenizer.tokenize(w_x[i].lower()))
            w_y[i] = self.tokenizer.convert_tokens_to_ids(w_y[i])        
        if text[-1]==']':
            text = text+' . .'
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        segments_ids = [0] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predictsx1,predictsx2,predictsx3 = [],[],[]
        for i in range(len(mask_input)):
            predictsx1.append(predictions[0,mask_input[i],:].numpy()+10)
            predictsx2.append(predictions[0,mask_input[i]+1,:].numpy()+10)
            predictsx3.append(predictions[0,mask_input[i]+2,:].numpy()+10)

        ver_w_y2 = 0
        ver_w_y,ver_w2,ver_w3 = [],[],[]
        output = []
        #print(w_x)
        #print(w_y)
        for i in range(len(w_x)):
            if len(w_y[i])>2:
                ver_w_y.append(abs(predictsx1[i][w_y[i][0]]*predictsx2[i][w_y[i][1]]*predictsx3[i][w_y[i][2]])**(1/3))
            elif len(w_y[i])>1:
                ver_w_y.append(abs(predictsx1[i][w_y[i][0]]*predictsx2[i][w_y[i][1]])**(1/2))
            elif len(w_y[i])==1:
                #print(1,w_y[i][0])
                #print(2,predictsx1[i])
                #print(3,predictsx1[i][w_y[i][0]])
                ver_w_y.append(predictsx1[i][w_y[i][0]])  
            else:
                ver_w_y.append(0)
        for i in range(len(w_x)//2):
            ver_w2.append(ver_w_y[i] / ver_w_y[len(w_x)//2 +i])
            if w_x[i]=='вглубь' or w_x[i]=='потом':
                ver_w2[i] = ver_w2[i]-0.4
            if w_x[i]=='чтобы' or w_x[i]=='тоже' or w_x[i]=='также':
                ver_w2[i] = ver_w2[i]-0.2
            if ver_w2[i]>1:
                ver_w2[i] = 1
            #print(ver_w2[i], w_x[i],ver_w_y[i], w_x[len(w_x)//2 +i],ver_w_y[len(w_x)//2 +i])
        for i in range(len(ver_w2)//2):
            ver_w3.append(ver_w2[i*2]*ver_w2[i*2+1])
            #print(ver_w3[i], w_x[i*2], w_x[i*2+1])
        max1 = 0
        mem1 = ''
        for i in range(len(ver_w3)):    
            if ver_w3[i]>max1:
                max1 = ver_w3[i]
                mem1 = w_x[2*i]+w_x[2*i+1]
        output = mem1
        return output
        
    def what_mask13(self, text,w_x, z):
        w_y = []
        for i in range(len(w_x)):
            w_y.append(self.tokenizer.tokenize(w_x[i].lower()))
            w_y[i] = self.tokenizer.convert_tokens_to_ids(w_y[i])        
        if text[-1]==']':
            text = text+' . .'
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        segments_ids = [0] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predictsx1,predictsx2,predictsx3 = [],[],[]
        for i in range(len(mask_input)):
            predictsx1.append(predictions[0,mask_input[i],:].numpy()+10)
            predictsx2.append(predictions[0,mask_input[i]+1,:].numpy()+10)
            predictsx3.append(predictions[0,mask_input[i]+2,:].numpy()+10)

        ver_w_y2 = 0
        ver_w_y,ver_w2 = [],[]
        output = []
        #print(w_x)
        #print(w_y)
        for i in range(len(w_x)):
            if len(w_y[i])>2:
                ver_w_y.append(abs(predictsx1[i][w_y[i][0]]*predictsx2[i][w_y[i][1]]*predictsx3[i][w_y[i][2]])**(1/3))
            elif len(w_y[i])>1:
                ver_w_y.append(abs(predictsx1[i][w_y[i][0]]*predictsx2[i][w_y[i][1]])**(1/2))
            else:
                ver_w_y.append(predictsx1[i][w_y[i][0]])  
        for i in range(len(w_x)//2):
            ver_w2.append(ver_w_y[i] / ver_w_y[len(w_x)//2 +i])
            #if w_x[i] in z:
            #    print((z[w_x[i]]-0.95)*10)
            #    ver_w2[i] = ver_w2[i] + (z[w_x[i]]-0.95)*10
            #print(ver_w2[i], w_x[i],ver_w_y[i], w_x[len(w_x)//2 +i],ver_w_y[len(w_x)//2 +i])
        max1 = 0
        mem1 = ''
        for i in range(len(ver_w2)):    
            if ver_w2[i]>max1:
                max1 = ver_w2[i]
                mem1 = w_x[i]
        output = mem1
        return output
    def what_mask5(self, text,w_x):
        w_y = []
        for i in range(len(w_x)):
            w_y.append(self.tokenizer.tokenize(w_x[i].lower()))
            w_y[i] = self.tokenizer.convert_tokens_to_ids(w_y[i])        
        if text[-1]==']':
            text = text+' . .'
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        segments_ids = [0] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predictsx1,predictsx2,predictsx3 = [],[],[]
        for i in range(len(mask_input)):
            predictsx1.append(predictions[0,mask_input[i],:].numpy()+10)
            predictsx2.append(predictions[0,mask_input[i]+1,:].numpy()+10)
            predictsx3.append(predictions[0,mask_input[i]+2,:].numpy()+10)

        ver_w_y2 = 0
        ver_w_y,ver_w2 = [],[]
        output = []
        #print(w_x)
        #print(w_y)
        for i in range(len(w_x)):
            if len(w_y[i])>2:
                ver_w_y.append(abs(predictsx1[i][w_y[i][0]]*predictsx2[i][w_y[i][1]]*predictsx3[i][w_y[i][2]])**(1/3))
            elif len(w_y[i])>1:
                ver_w_y.append(abs(predictsx1[i][w_y[i][0]]*predictsx2[i][w_y[i][1]])**(1/2))
            else:
                ver_w_y.append(predictsx1[i][w_y[i][0]])  
        for i in range(len(w_x)//2):
            ver_w2.append(ver_w_y[i] / ver_w_y[len(w_x)//2 +i])
            #print(ver_w2[i], w_x[i],ver_w_y[i], w_x[len(w_x)//2 +i],ver_w_y[len(w_x)//2 +i])
        max1 = 0
        max2 = 0
        mem1 = ''
        mem2 = ''
        for i in range(len(ver_w2)):    
            if ver_w2[i]>max1 and ver_w2[i]>0.8:
                max2 = max1
                max1 = ver_w2[i]
                mem2 = mem1
                mem1 = w_x[i]
            elif ver_w2[i]>max2 and ver_w2[i]>0.8:
                max2 = ver_w2[i]
                mem2 = w_x[i]
        output =[mem1,mem2]
        return output
        
    def del_what_mask(self, text):
        w = self.tokenizer.tokenize(',')
        w_i = self.tokenizer.convert_tokens_to_ids(w)
        w = self.tokenizer.tokenize('^')
        w_j = self.tokenizer.convert_tokens_to_ids(w)        
        #print(text)
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        #print(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predictsx1 = []
        for i in range(len(mask_input)):
            predictsx1.append(predictions[0,mask_input[i],:])
            predicts1 = predictsx1[i].argsort()[-8:].numpy()
            out1 = self.tokenizer.convert_ids_to_tokens(predicts1)
            #print(out1)
        #print(predictsx1[0].numpy().shape)
        output = []
        a=len(mask_input)
        for i in range(a):
            if predictsx1[i][w_i] > predictsx1[i][w_j]:
                output.append(str(i+1))
        """b = 0
        for i in range(a):
            b += predictsx1[i][w_i]*predictsx1[i][w_i]
        if a > 0:
            c=b/a
        else:
            c= 0
        for i in range(a):
            #print(predictsx1[i][w_i],c)
            if (predictsx1[i][w_i]*predictsx1[i][w_i]>c):
                output.append(str(i+1))"""
        return output
    def what_mask2(self, text,z,type_a = 0):
        #print(text)
        #print(z)
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        #print(tokenized_text)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        out2, out3 ='',''
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
            predictsx1 = predictions[0,mask_input[0],:]
            predicts1 = predictsx1.argsort()[-50:].numpy()
            out1 = self.tokenizer.convert_ids_to_tokens(predicts1)
            if (len(mask_input)>1):
                predictsx2 = predictions[0,mask_input[1],:]
                predicts2 = predictsx2.argsort()[-50:].numpy()
                out2 = self.tokenizer.convert_ids_to_tokens(predicts2)
        out = ""
        search = True
        for i in range(len(out1)-1,-1,-1):
            if type_a == 3:
                break
            out1[i] = out1[i].strip().lower()
            if out2:
                for w in z:
                    if type_a == 3:
                        break
                    #print(w,out1[i], (out1[i] in w))
                    if out1[i].lower() in w:
                        w2 = w.replace(out1[i],'')
                        #print (w, w2)
                        for j in range(len(out2)-1,-1,-1):
                            if out2[j].lower() in w2:
                                w3 = w2.replace(out2[j],'').replace(' ','')
                                #print('!',w3,'|')
                                if not w3:
                                    type_a = 3
                                    out = w.replace(' ','')
                                    break
        #print (out,out1,out2)

        
        if type_a==2:
            out = "не знаю"
        elif type_a==4:
            for w in z:
                if '-' in w or ' ' in w:
                    out = w.replace(' ','').replace('-','')
                    break
            if not out:
                search = False
                if z:
                    out = z[0]
                else:
                    out = out1[-1]
        else:
            for i in range(len(out1)-1,-1,-1): 
                if (len(out1[i])<2 and (out1[i]!='и' and out1[i]!='а')) or out1[i] =='вот':
                    #if (len(out1[i])<1) or out1[i] =='вот':
                    out1.pop(i)
                elif out1[i] in z:
                    out = out1[i]
                    break
            if not out and type_a==1:
                for w in z:
                    if '-' in w or ' ' in w:
                        out = w.replace(' ','').replace('-','')
                        break
            if not out:
                search = False
                if z:
                    out = z[0]
                else:
                    out = out1[-1]
        if out == 'ксожалению':
            out = 'такимобразом'
        if out == 'вовсене':
            out = 'именно' 
        return out,search
class BertEmbedder2(object):
    """
    Embedding Wrapper on Bert Multilingual Cased
    """

    def __init__(self):
        self.model_file = "./data/ru_conversational_cased_L-12_H-768_A-12.tar.gz"
        self.vocab_file = "./data/vocab_2.txt"
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()

    @singleton
    def bert_model(self):
        model = BertModel.from_pretrained(self.model_file).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=False)
        return tokenizer

    def sentence_embedding(self, text_list):
        embeddings = []
        for text in text_list:
            token_list = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
            segments_ids, indexed_tokens = [1] * len(token_list), self.tokenizer.convert_tokens_to_ids(token_list)
            segments_tensors, tokens_tensor = torch.tensor([segments_ids]), torch.tensor([indexed_tokens])
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            sent_embedding = torch.mean(encoded_layers[11], 1)
            embeddings.append(sent_embedding)
        return embeddings
        
class BertEmbedder(object):
    """
    Embedding Wrapper on Bert Multilingual Cased
    """

    def __init__(self):
        self.model_file = "./data/ru_conversational_cased_L-12_H-768_A-12.tar.gz"
        self.vocab_file = "./data/vocab_2.txt"
        #self.model_file = "./data/bert-base-multilingual-cased.tar.gz"
        #self.vocab_file = "./data/bert-base-multilingual-cased-vocab.txt"
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()
        self.embedding_matrix = self.get_bert_embed_matrix()

    @singleton
    def bert_model(self):
        model = BertModel.from_pretrained(self.model_file).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=False)
        return tokenizer

    @singleton
    def get_bert_embed_matrix(self):
        bert_embeddings = list(self.model.children())[0]
        bert_word_embeddings = list(bert_embeddings.children())[0]
        matrix = bert_word_embeddings.weight.data.numpy()
        return matrix

    def sentence_embedding(self, text_list):
        embeddings = []
        for text in text_list:
            token_list = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
            segments_ids, indexed_tokens = [1] * len(token_list), self.tokenizer.convert_tokens_to_ids(token_list)
            segments_tensors, tokens_tensor = torch.tensor([segments_ids]), torch.tensor([indexed_tokens])
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            sent_embedding = torch.mean(encoded_layers[11], 1)
            embeddings.append(sent_embedding)
        return embeddings

    def token_embedding(self, token_list):
        token_embedding = []
        for token in token_list:
            ontoken = self.tokenizer.tokenize(token)
            segments_ids, indexed_tokens = [1] * len(ontoken), self.tokenizer.convert_tokens_to_ids(ontoken)
            segments_tensors, tokens_tensor = torch.tensor([segments_ids]), torch.tensor([indexed_tokens])
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            ontoken_embeddings = []
            for subtoken_i in range(len(ontoken)):
                hidden_layers = []
                for layer_i in range(len(encoded_layers)):
                    vector = encoded_layers[layer_i][0][subtoken_i]
                    hidden_layers.append(vector)
                ontoken_embeddings.append(hidden_layers)
            cat_last_4_layers = [torch.cat((layer[-4:]), 0) for layer in ontoken_embeddings]
            token_embedding.append(cat_last_4_layers)
        token_embedding = torch.stack(token_embedding[0], 0) if len(token_embedding) > 1 else token_embedding[0][0]
        return token_embedding
    
    def compare_text_with_variants2(self, sentences, t = 0):
        #print(sentences)
        stop_words = []
        diff = np.zeros(len(sentences)-1)
        with open('./data/stop_words.csv', encoding='utf-8') as f:
            for line in f:
                stop_words.append(line.replace("\n",""))
        token_input,tokens,len_sen = [],[],[]
        sen2 = np.zeros((len(sentences),120000)) 
        for j, sentence in enumerate(sentences):
            sentence = re.sub('\([0-9]*\)','',sentence).replace("<…>","")
            sentence = sentence.replace(",","").replace(".","").replace(":","").replace("»","").replace("«","").replace("-"," ").lower()
            sentence = ' '+sentence+' '
            for stop_word in stop_words:
                sentence = sentence.replace(' '+stop_word+' ',' ')
            sentence = sentence.replace('     ',' ').replace('    ',' ').replace('   ',' ').replace('  ',' ')
            
            tokens.append([])
            len_sen.append(0)
            #print(sentence)
            sentence = sentence.split()
            for x, _ in enumerate(sentence):
                sentence[x] = self.morph.parse(sentence[x])[0].normal_form
                if sentence[x] in stop_words:
                    sentence[x] = ""
                else: 
                    len_sen[j] +=1
            sentence = " ".join(sentence)
            #print(sentence)
            sentence = [sentence]
            for i in range(len(sentence)):
                tokens[j] = tokens[j] + self.tokenizer.tokenize(sentence[i]) 
            token_input.append(self.tokenizer.convert_tokens_to_ids(tokens[j]))
            for token in token_input[j]:
                sen2[j,token] +=1            
        for i in range(1,len(sentences)):
            diff[i-1] = np.linalg.norm(sen2[0] - sen2[i])

        #print(1,diff)
        """if t ==0:
            diff = diff.argsort()[:2]+1
        else:
            diff = diff.argsort()[-2:]+1
        print(2,diff)
        output = diff.tolist()
        print(3,output)
        output = [str(item) for item in output]
        #print(output)"""
        output ={}
        for i in range(len(diff)):
            output[i] = diff[i]
        return output

class UDPipeError(Exception):
    def __init__(self, err):
        self.err = err

    def __str__(self):
        return self.err


def iter_words(sentences):
    for s in sentences:
        for w in s.words[1:]:
            yield w


class Pipeline(object):
    def __init__(self, input_format='conllu', model=None, output_format=None, output_stream=None,
                 tag=False, parse=True):
        self.model = model

        # if model:
        #    self.input_format = model.newTokenizer(model.DEFAULT)
        # else:
        self.input_format = ufal.udpipe.InputFormat.newInputFormat(input_format)

        self.pipes = []

        self.pipes.append(self.read_input)

        if tag:
            self.pipes.append(self.tag)
        if parse:
            self.pipes.append(self.parse)
        if output_format:
            self.output_format = ufal.udpipe.OutputFormat.newOutputFormat(output_format)
            self.output_stream = output_stream
            self.pipes.append(self.write_output)

    def read_input(self, data):
        # Input text
        self.input_format.setText(data)

        # Errors will show up here
        error = ufal.udpipe.ProcessingError()

        # Create empty sentence
        sentence = ufal.udpipe.Sentence()

        # Fill sentence object
        while self.input_format.nextSentence(sentence, error):
            # Check for error
            if error.occurred():
                raise UDPipeError(error.message)

            yield sentence

            sentence = ufal.udpipe.Sentence()

    def tag(self, sentences):
        """Tag sentences adding lemmas, pos tags and features for each token."""

        for sentence in sentences:
            self.model.tag(sentence, self.model.DEFAULT)
            yield sentence

    def parse(self, sentences):
        """Tag sentences adding lemmas, pos tags and features for each token."""

        for sentence in sentences:
            self.model.parse(sentence, self.model.DEFAULT)
            yield sentence

    def write_output(self, sentences):
        output = ""

        for sentence in sentences:
            output += self.output_format.writeSentence(sentence)

        output += self.output_format.finishDocument()

        return output

    def process(self, inputs):
        for fn in self.pipes:
            inputs = fn(inputs)

        return inputs


def standardize_task(task):
    if "choices" not in task:
        if "question" in task and "choices" in task["question"]:
            task["choices"] = task["question"]["choices"]
        else:
            parts = task["text"].split("\n")
            task["text"] = parts[0]
            task["choices"] = []
            for i in range(1, len(parts)):
                task["choices"].append({"id": str(i), "text": parts[i]})
    for i in range(len(task["choices"])):
        parts = [x.strip() for x in task["choices"][i]["text"].split(",")]
        task["choices"][i]["parts"] = parts
    return task


def check_solution(task, solution):
    if "correct_variants" in task["solution"]:
        correct = set(task["solution"]["correct_variants"][0])
    elif "correct" in task["solution"]:
        correct = set(task["solution"]["correct"])
    else:
        raise ValueError("Unknown task format!")
    return float(set([str(x) for x in solution]) == correct)


def random_solve_task(task):
    """
    :param task: standardized task
    :return: list of string labels
    """
    choice_decisions = []
    for ch in task["choices"]:
        if random.randint(0, 1):
            choice_decisions.append(ch["id"])
    return choice_decisions
