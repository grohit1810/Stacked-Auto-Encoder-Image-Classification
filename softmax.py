import numpy as np
import pickle

inputs = pickle.load(open("l1.pkl", "rb"))
W3 = np.random.randn(5, 8) * 0.01 #random init of the wt matrix
b3 = np.random.randn(5, 1) * 0.01 #random init of bias	

# softmax 
def softmax(inputs):
	inputs = np.array(inputs)
	yo = np.exp(np.dot(W3, inputs) + b3)
	op = yo/np.sum(yo)
	return op


num_epochs = 20
target = pickle.load(open("out.pkl", "rb"))

# backpropagation
def backprop(i, op, target):
	global W3
	target = np.array(target).reshape(5, 1)
	error = op - target
	dW3 = np.zeros_like(W3)
	dW3 = np.dot(error, i.T)
	W3 += -0.01 * dW3
for i in range(num_epochs):
	for j in range(len(inputs)):
		if i > 1600:
			break
		else:
			outputs = softmax(inputs[j])
			backprop(inputs[j], outputs, target[j])
	pickle.dump(W3, open("W3.pkl", "wb"))
	pickle.dump(b3, open("b3.pkl", "wb"))

