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
n1 = int(0.8*len(words)) # 80% of the words are in our dataset for training
n2 = int(0.9*len(words)) # 90% of the words are in our dataset for validation

Xtr, Ytr = build_dataset(words[:n1]) # training data, about 80% of data
Xva, Yva = build_dataset(words[n1:n2]) # validation data, about 10% of data
Xte, Yte = build_dataset(words[n2:]) # test data, about 10% of data

#C is the character embedding matrix where each row is a learned vector for one symbol(one of 27 characters)
#before training we initialize the character embedding matrix with random values, from a normal distribution with mean 0 and standard deviation 1
#the entries of C must be floats as they are learned continuous values which we will use to compute dot products with the one hot encoded input to get logits

#So X is what the model sees, Y is the correct next character that comes after the X data and the learned outputs live in parameters like C
C = torch.randn((27,2), generator = g) # our lookup table, 27 rows for the 27 characters, 2 columns for the 2 hidden units

# Now for Y, we have to pluck out the character we want and the actual probability we have for predicting it
#prob[torch.arange(Y.shape[0]), Y]

W1 = torch.randn((6,100), generator = g)
b1 = torch.randn(100, generator = g)


W2 = torch.randn((100,27), generator = g) 
b2 = torch.randn(27, generator = g) 


parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True 

#We want to find the best leanring rate for the model, so we will try different learning rates and see which one works best
lre = torch.linspace(-3,0,1000) # learning rate exponent
lrs = 10**lre # learning rate(10 to the power of learning rate exponent)

lri = [] # to keep track of learning rates we've used
lossi = [] # to keep track of losses we've seen

for k in range(10000): # we go through 10 iterations to train the model

    ix = torch.randint(0, Xtr.shape[0], (32,)) # use XTrain data, making minibatches of 32 to train the model all rather than using 32,000 datapoints
    #FORWARD PASS
    emb = C[Xtr[ix]] # use XTrain data, each row of emb is the embedding of the corresponding row of X, X[ix] is [32,3,2]
    h = torch.tanh(emb.view(-1,6) @ W1 + b1)
    logits = h @ W2 + b2 # logits are the unnormalized probabilities for each character
    loss = F.cross_entropy(logits, Ytr[ix]) # cross_entropy function also calculates the loss with logits and Y tensor
    #print(k, '   ', loss.item())

    #BACKWARD PASS
    for p in parameters:#initialize the gradients to 0
        p.grad = None
    loss.backward() # backpropgation

    #Update Parameters
    #lr = lrs[k]
    lr = 0.1# we found that the best learning rate is about 0.1 based on tracking the stats for a more generalized model
    # we decrease learning rate for the model to understand more specific patterns in later stages of training
    for p in parameters:
        p.data += -lr * p.grad # learning rate * gradients

    #track stats
    #lri.append(lr)
    #lossi.append(loss.item())

print(loss.item())

