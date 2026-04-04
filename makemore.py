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
stoi = {s:i for i,s in enumerate(chars)} # mapping of s to i in the 2d array, rows is s, columns is i
stoi['<S>'] = 26 # <S> and <E> are not normal characters in the alphabet so we have to manually add them in 
stoi['<E>'] = 27
# we have 26 letters + the S and E special characters so we want a 28x28 2d array
N = torch.zeros((28,28),dtype=torch.int32) # pytorch automatically sets the data to be 32 bit floats, but we want to use integers as we are storing counts
for w in words: 
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs,chs[1:]): 
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1 # it automatically starts at 0, and we increment the count by 1 each time we find the bigram again 

itos = {i:s for s,i in stoi.items()}

#to help us visualize the 2d array, we can plot it as a heatmap
plt.figure(figsize=(20,9))
plt.imshow(N, cmap='Blues', aspect = 'auto')
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j,i,chstr, ha="center", va="bottom", color="gray")
        plt.text(j,i,str(N[i,j].item()), ha="center", va="top", color="gray")
plt.axis('off')
plt.show()