#this line helps us open up the names.txt file and then allows us to read what's in it and then put the results on new lines
from typing import Any
import torch
import matplotlib.pyplot as plt



words = open('names.txt', 'r').read().splitlines() 
#print(len(words))
#print(min(len(w) for w in words))
#print(max(len(w) for w in words))


# in the bigram languange model, we keep a count of how many pairs of characters we see in different names in the dataset
b = {}
for w in words: # in the first column we are printing the number of words specified in the square brackets after words
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs,chs[1:]): # zip takes two iterators, pairs them up and then creates an iterator over the tuples of entries
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1# if the bigram is not in the dictionary we return 0, then we add 1 as it is the first occurrence of that bigram

#print(sorted(b.items(), key = lambda kv: -kv[1])) # print out the bigrams in descending order for counts, if we want to do ascending order we change it to kv[1]
# it's more efficient to use a 2d array rather than a dictionary, so here we will make a 2d array where rows are the first character of the bigram and columns are the second character of the bigram
# so each entry in the 2d array will tell us how many times the second character follows the first one


chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)} # mapping of s to i in the 2d array, rows is s, columns is i
stoi['.'] = 0 # <S> and <E> are not normal characters in the alphabet so we have to manually add them in 
# we have 26 letters + the S and E special characters so we want a 28x28 2d array
N = torch.zeros((27,27),dtype=torch.int32) # pytorch automatically sets the data to be 32 bit floats, but we want to use integers as we are storing counts
for w in words: 
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]): 
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1 # it automatically starts at 0, and we increment the count by 1 each time we find the bigram again 

itos = {i:s for s,i in stoi.items()}

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(N[0].float(), num_samples=1, replacement=True, generator=g).item()
#itos[ix]
P = (N+1).float() # set it to N+1 to ensure that no probability is 0 so log likelihood is not undefined (like infinity)
P = P/P.sum(1, keepdims=True) # normalize each row to sum to 1
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        #p = torch.ones(27)/27.0 # uniform distributionwhere every output is equally likely(model is untrained here, output will be terrible)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        #print(itos[ix])
        if ix == 0:
            break;
    #print(''.join(out))


#GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
#equivalient to maximizing the log_likelihood(because log is monotonic)
#equivalent to minimize the negative log_likelihood
#equivalent to minimizing the average negative log_likelihood
#log(a*b*c) = log(a) + log(b) + log(c)

log_likelihood = 0.0
n = 0
for w in words[:3]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n+=1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}') # negative log_likelihood
print(f'{nll/n}') #average negative log_likelihood, in practice we want to minimize this as it reflects the quality of your model
# the lowest the average negative log_likelihood can go is 0, the lower it is the better as it means higher probabilities for values




#to help us visualize the 2d array, we can plot it as a heatmap
# plt.figure(figsize=(20,9))
# plt.imshow(N, cmap='Blues', aspect = 'auto')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j,i,chstr, ha="center", va="bottom", color="gray")
#         plt.text(j,i,str(N[i,j].item()), ha="center", va="top", color="gray")
# plt.axis('off')
# plt.show()