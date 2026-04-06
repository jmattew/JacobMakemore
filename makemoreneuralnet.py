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
xs = []; ys = [] # x is for input to the neural net, ys represent labels for the actualnext character(output from the neural net)
g = torch.Generator().manual_seed(2147483647) # makes it so we see the same values as Andrej Karpathy's video
w = torch.randn(27,27, generator=g) # randomly initialize 27 neuron's weights, each neuron receives 27 inputs


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


#----------------------One Forward Pass-----------------------------

#in one hot encoding we will take an integer like 13, then make a vector of all 0s except the 13th dimension of it which will be 1
xenc = F.one_hot(xs, num_classes=27)


#W = torch.randn(27,1) # W represents one neuron with a weight, W will be a column vector of 27 vectors, 1 column, 27 rows, the weights in W are multiplied by the inputs
#xenc @ W # the @ is the matrix multiplication operator in pytorch, the output of this operation will be a 5x1 column vector
# when you multiply two matriced together, the number of columns in the first matrix must be equal to the number of rows in the second matrix
# also, the number of rows in the first matrix and the number of columns in the second matrix become the number of rows and columns in the answer

W = torch.randn(27,27) # now we have 27 neurons, each neuron has a weight, W will be a 27x1 column vector of weights 
logits = xenc @ W #get log counts, doing the matrix multiplication will evaluate all 27 neurons(weights) in parallel against the 5 inputs so 5x27 X 27x27 = 5x27 with 5 inputs we get 27 outputs
# our neural net here will only have one layer, so the output of the first layer will be our results
counts = logits.exp() # get something that looks like counts with.exp(), the .exp() is the exponential function, it will convert the output to a probability distribution which are values we can actually use
probs = counts/counts.sum(1, keepdims=True) # normalize the counts so that all the values when summed together sum up to sum to 1
# now these are proper probabilies which we can use to predict the next character

#counts and probs make up the softmax activation function, which takes logits, exponentiates them(e^logit) and then divides by the sum of the 
#exponentiated values to get a probability distribution so now all the values sum up to 1



# all of the above functions are differentiable which means we can backpropogate through them to update the weights


#-------------------------------------------------------


#----------------------Now Backward Pass with Backpropagation---------------------------------

nlls = torch.zeros(5)
for i in range(5):
    #ith bigram
    x = xs[i].item() # input character index
    y = ys[i].item() # output or label character index
    print("--------")
    print(f'bigrame example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
    print(f'input to the neural net: ', x)
    print(f'output probabilities of the neural net: ', probs[i])
    print(f'label or actual next character: ', y)
    p = probs[i,y]
    print('probabilty assigned by the neural net to the actual correct next character:', p.item())
    logp = torch.log(p)
    print('log likelihood: ', logp.item())
    nll = -logp
    print('negative log likelihood: ', nll.item())
    nlls[i] = nll

print('========')
print('average negative log likelihood: ', nlls.mean().item())







#-------------------------------------------------------


plt.imshow(xenc)
plt.show()
