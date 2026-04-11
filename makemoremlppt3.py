from typing import Any
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

#this line helps us open up the names.txt file and then allows us to read what's in it and then put the results on new lines
words = open('names.txt', 'r').read().splitlines()  

g = torch.Generator().manual_seed(2147483647) 

#build the vocabulary of characters in the dataset
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # mapping of s to i in the 2d array, rows is s, columns is i
stoi['.'] = 0
N = torch.zeros((27,27),dtype=torch.int32) # pytorch automatically sets the data to be 32 bit floats, but we want to use integers as we are storing counts
itos = {i:s for s,i in stoi.items()}

vocab_size = len(itos)
block_size = 3

def build_dataset(words):
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


n_embd = 10 # number of character embedding vectors
n_hidden = 200 # number of neurons in hidden layer of mlp

#C is the character embedding matrix where each row is a learned vector for one symbol(one of 27 characters)
#before training we initialize the character embedding matrix with random values, from a normal distribution with mean 0 and standard deviation 1
#the entries of C must be floats as they are learned continuous values which we will use to compute dot products with the one hot encoded input to get logits

#So X is what the model sees, Y is the correct next character that comes after the X data and the learned outputs live in parameters like C
C = torch.randn((27,2), generator = g) # our lookup table, 27 rows for the 27 characters, 2 columns for the 2 hidden units

# Now for Y, we have to pluck out the character we want and the actual probability we have for predicting it
#prob[torch.arange(Y.shape[0]), Y]

W1 = torch.randn((n_embd * block_size,n_hidden), generator = g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden, generator = g) * 0.01


W2 = torch.randn((n_hidden,vocab_size), generator = g) * 0.01
b2 = torch.randn(27, generator = g) * 0  

#these 4 are batch normalization parameters--------------------------------
bngain = torch.ones((1,n_hidden))
bnbias = torch.zeros((1,n_hidden))
# gain and bias are trained usind backpropogation

#these 2 are buffers
bnstd_running = torch.zeros((1,n_hidden))
bnmean_running = torch.ones((1,n_hidden))
#running mean and standard deviation are used to keep track of the mean and standard deviation of the hidden layer,
#running mean and standard deviation are not trained with backpropogation, they are trained with the small updates in the training pass
#--------------------------------------------------------------------------------

parameters = [C, W1, b1, W2, b2, bngain, bnbias]

for p in parameters:
    p.requires_grad = True 


max_steps = 20000
batch_size = 32
lossi = []
for i in range(max_steps): # we go through 10 iterations to train the model

    #minibatch construction
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g) # use XTrain data, making minibatches of 32 to train the model all rather than using 32,000 datapoints
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X, Y

    #FORWARD PASS
    emb = C[Xb] # use XTrain data, each row of emb is the embedding of the corresponding row of X, X[ix] is [32,3,2]
    embcat = emb.view(emb.shape[0],-1) # concatenate the vectors

    #linear layer
    hpreact = embcat @ W1 #+ b1 # hidden layer pre activation

    #batch normalization layer--------------------------------------------------------
    bnmeani = hpreact.mean(0,keepdims=True)
    bnstdi = hpreact.std(0,keepdims=True)
    hpreact = bngain * (hpreact - bnmeani)/bnstdi + bnbias # in our forward passwe estimate the mean and standard deviation of the hidden layer 
    #and then scale the hidden layer to have a mean of 0 and a standard deviation of 1
    with torch.no_grad():
        bmean_running = 0.999 * bnmean_running + 0.001 * bnmeani # to keep track of the running mean 
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi # to keep track of the running standard deviation
    #----------------------------------------------------------------------------------------

    #non-linearity layer
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2 # logits are the unnormalized probabilities for each character, this is the output layer
    loss = F.cross_entropy(logits, Ytr[ix]) # cross_entropy function also calculates the loss with logits and Y tensor
    #print(k, '   ', loss.item())

    #BACKWARD PASS
    for p in parameters:#initialize the gradients to 0
        p.grad = None
    loss.backward() # backpropgation

    #Update Parameters
    # we found that the best learning rate is about 0.1 based on tracking the stats for a more generalized model
    # we decrease learning rate for the model to understand more specific patterns in later stages of training
    lr = 0.1 if i<100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad # learning rate * gradients

    #track stats
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d} loss: {loss.item():.4f}')

    lossi.append(loss.log10().item())
print('training loss:', loss.item())

#-torch.tensor(1.0/27.0).log() # this should be the probability of any character in the vocabulary

#4 dimensional example of the issue:
#logits = torch.tensor([0.0, 0.0, 0.0, 0.0]) # if all the values are the same we see the uniform distribution
##probs = F.softmax(logits, dim=0)
#loss = -probs[2].log() # this is the loss for the correct character
#print(probs, '      loss: ', loss) # the value for the correct character's index should have a very high probability

with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0],-1)
    hpreact = embcat @ W1 + b1
    bnmean =  hpreact.mean(0,keepdims=True)
    bnstd = hpreact.std(0,keepdims=True)
    

@torch.no_grad() # this disables gradient tracking for the following code
def split_loss(split):

    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xva, Yva),
        'test': (Xte, Yte),
    }[split]

    emb = C[x] # use XTrain data, each row of emb is the embedding of the corresponding row of X, X[ix] is [32,3,2]
    embcat = emb.view(emb.shape[0],-1) # concatenate the vectors
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean) / (bnstd + 1e-5) + bnbias
    h = torch.tanh(bngain * (hpreact - bnmean) / (bnstd + 1e-5) + bnbias) # hidden layer
    logits = h @ W2 + b2 # logits are the unnormalized probabilities for each character, this is the output layer
    loss = F.cross_entropy(logits, y) # cross_entropy function also calculates the loss with logits and Y tensor
    print(split, loss.item())

split_loss('train')
split_loss('val')