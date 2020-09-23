import pickle
import numpy as np
inp=pickle.load(open("i.pkl","rb"))

for i in xrange(len(inp)):
inp[i]=np.array(inp[i]).reshape(768,1)
out=pickle.load(open("o.pkl","rb"))


for i in xrange(len(out)):
	out[i]=np.array(out[i]).reshape(5,1)

# pickle.dump(inp,open("inp.pkl","wb"))
pickle.dump(out,open("out.pkl","wb"))


