import numpy as np
import pickle

# softmax 
def softmax(inputs):
	inputs = np.array(inputs)
	yo = np.exp(np.dot(W3, inputs) + b3)
	op = yo/np.sum(yo)
	return op


W1 = pickle.load(open("Wxh.pkl", "rb")) #weight matrices
b1 = pickle.load(open("bh.pkl", "rb"))
W2 = pickle.load(open("Wxh1.pkl", "rb"))
b2 = pickle.load(open("bh1.pkl", "rb"))
W3 = pickle.load(open("W3.pkl", "rb"))
b3 = pickle.load(open("b3.pkl", "rb"))


all_inps = pickle.load(open("inp.pkl", "rb"))
all_outs = pickle.load(open("out.pkl", "rb"))
c1 = 0
c2 = 0
for i in range(len(all_inps)):
	if i <= 1600:
		continue
	c1 += 1
	val1 = np.tanh(np.dot(W1, all_inps[i]) + b1)
	val2 = np.tanh(np.dot(W2, val1) + b2)
	ans = softmax(val2)
	all_outs[i] = all_outs[i].tolist()
	if np.argmax(ans) == all_outs[i].index(max(all_outs[i])):
		c2 += 1

print float(c2)/c1
