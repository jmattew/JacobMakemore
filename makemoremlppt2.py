#this line helps us open up the names.txt file and then allows us to read what's in it and then put the results on new lines
from typing import Any
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

words = open('names.txt', 'r').read().splitlines()  

g = torch.Generator().manual_seed(2147483647) 

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # mapping of s to i in the 2d array, rows is s, columns is i
stoi['.'] = 0
N = torch.zeros((27,27),dtype=torch.int32) # pytorch automatically sets the data to be 32 bit floats, but we want to use integers as we are storing counts

itos = {i:s for s,i in stoi.items()}

def build_dataset(words):
    block_size = 4 # context length, how many characters do we take in to predict the next character?
    X, Y = [], [] # X is the training input, each row is a context of the number of characters specified in block_size
    # Y is the training output, each row is the next character in the context, basically the matching entry in Y that you want the model to predict

    for w in words:
        #print(w)
        context = [0] * (block_size)
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)

            #print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xva, Yva = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# print(Xtr.shape, Ytr.shape)
# print(Xva.shape, Yva.shape)
# print(Xte.shape, Yte.shape)
