#this line helps us open up the names.txt file and then allows us to read what's in it and then put the results on new lines
from typing import Any


words = open('names.txt', 'r').read().splitlines() 
print(words[:10])
#print(len(words))
#print(min(len(w) for w in words))
#print(max(len(w) for w in words))


# in the bigram languange model, we keep a count of how many pairs of characters we see in different names in the dataset
b = {}
for w in words[:3]: # in the first column we are printing the number of words specified in the square brackets after words
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs,chs[1:]): # zip takes two iterators, pairs them up and then creates an iterator over the tuples of entries
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1# if the bigram is not in the dictionary we return 0, then we add 1 as it is the first occurrence of that bigram
        print(ch1, ch2)
