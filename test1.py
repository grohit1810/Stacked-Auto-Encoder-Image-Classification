import numpy as np
import pickle

#targets=pickle.load(open("targets.pkl","rb"))
Wxh=pickle.load(open("Wxh.pkl","rb"))
bh=pickle.load(open("bh.pkl","rb"))
inputs=pickle.load(open("inp.pkl","rb"))

l=[]
for X in inputs[:1600]:
    
    h_a = np.tanh(np.dot(Wxh, np.reshape(X,(len(X),1))) + bh)
    
    l.append(h_a)
    
print len(l)
print l[500]

pickle.dump(l,open("l.pkl","wb"))
