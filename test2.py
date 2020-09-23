import numpy as np
import pickle

#targets=pickle.load(open("targets.pkl","rb"))
Wxh=pickle.load(open("Wxh1.pkl","rb"))
bh=pickle.load(open("bh1.pkl","rb"))
inputs=pickle.load(open("l.pkl","rb"))

l=[]
for X in inputs[:1600]:
    
    h_a = np.tanh(np.dot(Wxh, np.reshape(X,(len(X),1))) + bh)
    
    l.append(h_a)
    
print len(l)
print l[500]

pickle.dump(l,open("l1.pkl","wb"))
