from typing import Any
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()  

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # mapping of s to i in the 2d array, rows is s, columns is i
stoi['.'] = 0
N = torch.zeros((27,27),dtype=torch.int32) # pytorch automatically sets the data to be 32 bit floats, but we want to use integers as we are storing counts

P = (N+1).float() # set it to N+1 to ensure that no probability is 0 so log likelihood is not undefined (like infinity)
P = P/P.sum(1, keepdims=True) # normalize each row to sum to 1


#GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
#equivalient to maximizing the log_likelihood(because log is monotonic)
#equivalent to minimize the negative log_likelihood
#equivalent to minimizing the average negative log_likelihood
#log(a*b*c) = log(a) + log(b) + log(c)
log_likelihood = 0.0
n = 0

#create a training set of bigrams:
# for w in words: 
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs,chs[1:]): 
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1,ix2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         n+=1
#         #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
# print(f'{log_likelihood=}')
# nll = -log_likelihood
# print(f'{nll=}') # negative log_likelihood
# print(f'{nll/n}') #average negative log_likelihood, in practice we want to minimize this as it reflects the quality of your model
# the lowest the average negative log_likelihood can go is 0, the lower it is the better as it means higher probabilities for values

#create training set of all bigrams (x,y): x is the first character, y is the predicted character
xs = []; ys = []

for w in words[:1]: 
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]): 
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
# example with emma: xs = [0,5,13,13,1], ys = [5,13,13,1,0], if 0 or . is inputted we should expect the probility for character 5 or e to be high
# if 5 is inputted we should expect the probility for character 13 or m to be high
# if 13 is inputted we should expect the probility for character 13 or 1 to be high

#the neural net is made up of neurons and the neurons have weights and biases which act multiplicatively on the inputs
#before we were feeding in the integers of the character's place in the alphabet into the neural net but that won't really give an accurate value
#so we can use one hat encoding to represent the characters and feed them into the neural net

#in one hat encoding we will take an integer like 13, then make a vector of all 0s except the 13th dimension of it which will be 1
xenc = F.one_hot(xs, num_classes=27)
plt.imshow(xenc)
plt.show()
