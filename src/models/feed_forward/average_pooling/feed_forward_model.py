import random
import numpy as np
import collections
from sklearn.metrics import accuracy_score
from operator import add
random.seed(0)

#hyperparameters 

emb_size = 50

fin = open("../../../dataset/web_reviews/psst-tokens.tsv",'r')

data=fin.readlines()

x_tups = []

for line in data:
	words = line.strip("\n").strip().split("\t")
	lid = words[0].strip()
	prep = words[1].strip().split()
	sense = words[2].strip()
	sentence = words[3].strip()

	sent_split = sentence.split("|")
	left_context = sent_split[0].strip().split()
	right_context = sent_split[-1].strip().split()

	tup = (lid, prep, left_context, right_context, sense)

	x_tups.append(tup)


print("input taken in. size is - ", len(x_tups))
#calc. uniques

uniques = set()

u_ids = set()
u_senses = set()

for tup in x_tups:
	
	u_ids.add(tup[0])
	u_senses.add(tup[-1])
	all_words = tup[1] + tup[2] + tup[3]
	for word in all_words:
		final_word = word.lower()
		uniques.add(final_word)

print("unique_ids are - ", len(u_ids))
print("unique senses are - ", len(u_senses))
#from word vectors file


filevec = "../../../word_vectors/glove.6B.50d.txt"
word_vectors = []
word2vec = {}

with open(filevec,"r") as fin:
	for line in fin:
		words = line.strip("\n").strip().split()
		if(len(words) == (emb_size + 1)):
			word_vectors.append(words[0])
			word2vec[words[0]] = [float(i) for i in words[1:]]

#checking in word vectors file.
found = 0
total = 0

for word in uniques:
	total += 1
	if word in word_vectors:
		found += 1

print("vocab size is - ", total, "found vectors for - ", found)

u_senses = list(u_senses)

sense2id = {}
id2sense = {}

for i in range(len(u_senses)):
	sense2id[u_senses[i]] = i
	id2sense[i] = u_senses[i]

uniques = list(uniques)

w2id = {}
id2w = {}

for i in range(len(uniques)):
	w2id[uniques[i]] = i
	id2w[i] = uniques[i]

w2id["<PAD>"] = len(uniques)
id2w[len(uniques)] = "<PAD>"

x_tups_again = []

for tup in x_tups:
	prepn = []
	for w in tup[1]:
		prepn.append(w2id[w.lower()])

	lcn = []
	for w in tup[2]:
		lcn.append(w2id[w.lower()])

	rcn = []
	for w in tup[3]:
		rcn.append(w2id[w.lower()])

	sensen = sense2id[tup[4]]

	x_tups_again.append((tup[0], prepn, lcn, rcn, sensen))

x_tups = x_tups_again


'''
x_tups_again = []

for tup in x_tups:
	sent = tup[2] + tup[1] + tup[3]
	x_tups_again.append((tup[0], sent, tup[-1]))

x_tups = x_tups_again
'''
print("Till ids replacement done")

sentMaxl = -1
sentMaxp = -1
sentMaxr = -1

for tup in x_tups:
	if(len(tup[1]) > sentMaxp):
		sentMaxp = len(tup[1])

	if(len(tup[2]) > sentMaxl):
		sentMaxl = len(tup[2])

	if(len(tup[3]) > sentMaxr):
		sentMaxr = len(tup[3])


x_tups_pad = []
id2lengths = {}

for tup in x_tups:
	paddedp = tup[1] + [w2id["<PAD>"] for i in range(sentMaxp - len(tup[1]) )]
	paddedl = tup[2] + [w2id["<PAD>"] for i in range(sentMaxl - len(tup[2]) )]
	paddedr = tup[3] + [w2id["<PAD>"] for i in range(sentMaxr - len(tup[3]) )]
	
	id2lengths[tup[0]] = (len(tup[1]), len(tup[2]), len(tup[3]))
	x_tups_pad.append((tup[0], paddedp, paddedl, paddedr, tup[-1]))

x_tups = x_tups_pad

print("padding done.")


#getting word vectors
wv = np.random.rand(len(uniques)+1, emb_size).tolist()

for word, wid in w2id.items():
	if word in word2vec:
		wv[wid] = word2vec[word]

print("wv created")

#process x_tups by converting everything into ids.
#finally I have w2id, id2w, sense2id, and x_tups with same form but all replaced with ids.

#now divide in train,test, dev.

ftest = open("../../../dataset/web_reviews/psst-test.sentids", "r")

test_ids = ftest.readlines()

test_set_ids = []

ids_considered = set()

for tid in test_ids:
	tid = tid.strip("\n").strip()

	test_set_ids.append(tid)

x_tups_test = []
x_tups_train = []

for tup in x_tups:
	if(tup[0].split(":")[0] in test_set_ids) and (tup[0].split(":")[0] not in ids_considered):
		x_tups_test.append(tup)
		ids_considered.add(tup[0].split(":")[0])
	else:
		x_tups_train.append(tup)

random.shuffle(x_tups_train)

x_tups_dev = x_tups_train[:450]
x_tups_train = x_tups_train[450:]

random.shuffle(x_tups_train)
random.shuffle(x_tups_dev)
random.shuffle(x_tups_test)

print("different datasets prepared. sizes are - ", len(x_tups_train), len(x_tups_dev), len(x_tups_test))


#####################################
#####################################
#training

#datasets preparation

def process(tuples):
	
	X_lids = []
	X_pids = []
	X_rids = []
	X_llen = []
	X_plen = []
	X_rlen = []

	y = []

	for tup in tuples:
		X_lids.append(tup[2])
		X_pids.append(tup[1])
		X_rids.append(tup[3])

		X_llen.append(id2lengths[tup[0]][1])
		X_plen.append(id2lengths[tup[0]][0])
		X_rlen.append(id2lengths[tup[0]][2])

		y.append(tup[-1])


	return np.array(X_lids), np.array(X_pids), np.array(X_rids), np.array(X_llen), np.array(X_plen), np.array(X_rlen), np.array(y)


X_train_lids, X_train_pids, X_train_rids, X_train_llen, X_train_plen, X_train_rlen, y_train = process(x_tups_train)
X_dev_lids, X_dev_pids, X_dev_rids, X_dev_llen, X_dev_plen, X_dev_rlen, y_dev = process(x_tups_dev)
X_test_lids, X_test_pids, X_test_rids, X_test_llen, X_test_plen, X_test_rlen, y_test = process(x_tups_test)

##########################
#TRAINING
from tensorflow_class import *

#hyper parameters
learning_rate = 1e-2
num_classes = len(u_senses)
emb_size = 50
num_epochs = 100
l2_reg_lambda = 0.0001
batch_size = 100
drop_out = 0.7
max_iterations = int(len(X_train_lids)/batch_size)
test_after = 5

print("epochs - ", num_epochs, "max_iterations - ", max_iterations)
print("Now lets do it man")

#create model

model = MainModel(sentMaxl, sentMaxp, sentMaxr, num_classes, wv, emb_size, l2_reg_lambda, learning_rate)
print("Model created. Training the model.")
#train the model
print_after = 10

fout = open("feed_forward_output.txt","w")

print("hyperparameters",learning_rate, num_epochs ,l2_reg_lambda ,batch_size, drop_out )

for epoch in range(num_epochs):
	for iteration in range(max_iterations):
		b_start = iteration*batch_size
		b_end = (iteration+1)*batch_size

		batch_lids = X_train_lids[b_start:b_end,:] 
		batch_pids = X_train_pids[b_start:b_end,:]
		batch_rids = X_train_rids[b_start:b_end,:]
		batch_llen = X_train_llen[b_start:b_end]
		batch_plen = X_train_plen[b_start:b_end]
		batch_rlen = X_train_rlen[b_start:b_end]
		batch_y = y_train[b_start:b_end]

		batch_yy = []
		for ele in batch_y:
			yy = [0.0 for i in range(num_classes)]
			yy[ele] = 1.0
			batch_yy.append(yy)
		batch_y = np.array(batch_yy)

		l, acc = model.train_step(batch_lids, batch_pids, batch_rids, batch_llen, batch_plen, batch_rlen, batch_y, drop_out)

	##test on training data
	batch_y = y_train
	batch_yy = []
	for ele in batch_y:
		yy = [0.0 for i in range(num_classes)]
		yy[ele] = 1.0
		batch_yy.append(yy)
	batch_y = np.array(batch_yy)

	acc_train = model.test_step(X_train_lids, X_train_pids, X_train_rids, X_train_llen, X_train_plen, X_train_rlen, batch_y)
	print("accuracy on training in epoch, " ,epoch, "is - ", acc_train)

	#evaluate model on dev and test. while printing out on the console. > to a file.
	if (epoch%test_after == 0):
		fout.write("in epoch - " + str(epoch)+ "\n")

		batch_y = y_dev
		batch_yy = []
		for ele in batch_y:
			yy = [0.0 for i in range(num_classes)]
			yy[ele] = 1.0
			batch_yy.append(yy)
		batch_y = np.array(batch_yy)

		acc_dev = model.test_step(X_dev_lids, X_dev_pids, X_dev_rids, X_dev_llen, X_dev_plen, X_dev_rlen, batch_y)
		
		batch_y = y_test
		batch_yy = []
		for ele in batch_y:
			yy = [0.0 for i in range(num_classes)]
			yy[ele] = 1.0
			batch_yy.append(yy)
		batch_y = np.array(batch_yy)


		acc_test = model.test_step(X_test_lids, X_test_pids, X_test_rids, X_test_llen, X_test_plen, X_test_rlen, batch_y)

		fout.write("accuracy on dev set - " + str(acc_dev) + "\n")
		fout.write("accuracy on test set - " + str(acc_test) + "\n")
		fout.write("\n")

fout.close()