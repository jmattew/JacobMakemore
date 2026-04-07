#this line helps us open up the names.txt file and then allows us to read what's in it and then put the results on new lines
from typing import Any
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()  

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # mapping of s to i in the 2d array, rows is s, columns is i
stoi['.'] = 0
N = torch.zeros((27,27),dtype=torch.int32) # pytorch automatically sets the data to be 32 bit floats, but we want to use integers as we are storing counts

itos = {i:s for s,i in stoi.items()}

#this file is for the multi-layer perceptron model version of the bigram language model

block_size = 3 # context length, how many characters do we take in to predict the next character?
X, Y = [], [] # X is the training input, each row is a context of the number of characters specified in block_size
# Y is the training output, each row is the next character in the context, basically the matching entry in Y that you want the model to predict

for w in words[:5]:
    print(w)
    context = [0] * (block_size - 1)
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)

        print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

#C is the character embedding matrix where each row is a learned vector for one symbol(one of 27 characters)
#before training we initialize the character embedding matrix with random values, from a normal distribution with mean 0 and standard deviation 1
#the entries of C must be floats as they are learned continuous values which we will use to compute dot products with the one hot encoded input to get logits

#So X is what the model sees, Y is the correct next character that comes after the X data and the learned outputs live in parameters like C
C = torch.randn((27,2)) # our lookup table, 27 rows for the 27 characters, 2 columns for the 2 hidden units
print(C)

emb = C[X] # each row of emb is the embedding of the corresponding row of X
W1 = torch.randn((6,100))
b1 = torch.randn(100)
