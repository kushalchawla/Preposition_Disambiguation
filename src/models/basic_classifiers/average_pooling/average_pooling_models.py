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

sentMax = -1

for tup in x_tups:
	if(len(tup[1]) > sentMax):
		sentMax = len(tup[1])


'''
x_tups_pad = []

for tup in x_tups:
	padded = tup[1] + [w2id["<PAD>"] for i in range(sentMax - len(tup[1]) )]
	x_tups_pad.append((tup[0], padded, tup[2]))

x_tups = x_tups_pad

print("padding done.")
'''

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

def compress(my_tuple):
	
	vector = [0.0 for i in range(emb_size)]
	for wid in my_tuple:
		vector = map(add, vector, wv[wid])

	l1 = len(my_tuple)
	vector = [i/l1 for i in vector]
	
	return vector

def merge(v1,v2,v3):
	mm = v2 + v1 + v3
	return mm

def process(tuples):
	X = []
	y = []

	for tup in tuples:

		if(len(tup[1])>0):
			v1 = compress(tup[1])
		else:
			v1 = [0.0 for i in range(emb_size)]

		if(len(tup[2])>0):
			v2 = compress(tup[2])
		else:
			v2 = [0.0 for i in range(emb_size)]

		if(len(tup[3])>0):
			v3 = compress(tup[3])
		else:
			v3 = [0.0 for i in range(emb_size)]

		merged = merge(v1,v2,v3)
		X.append(merged)
		y.append(tup[-1])

	return np.array(X), np.array(y)



X_train, y_train = process(x_tups_train)
X_dev, y_dev = process(x_tups_dev)
X_test, y_test = process(x_tups_test)


##########################
#TRAINING
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

models = [svm.SVC(kernel="linear"), neighbors.KNeighborsClassifier(), RandomForestClassifier(n_estimators=100), MLPClassifier()]

for model in models:
	clf = model

	clf = clf.fit(X_train, y_train)

	######################################
	#assign class as a prediction for all samples.

	#predictions

	#devset
	predictions_dev = clf.predict(X_dev)

	#testset
	predictions_test = clf.predict(X_test)

	#evaluation

	#devset

	true_classes = []

	for tup in x_tups_dev:
		true_classes.append(tup[-1])

	accuracy_dev = accuracy_score(y_true=true_classes, y_pred=predictions_dev)

	

	#testset

	true_classes = []

	for tup in x_tups_test:
		true_classes.append(tup[-1])

	accuracy_test = accuracy_score(y_true=true_classes, y_pred=predictions_test)

	print("Model is - ", clf)
	print("accuracy on dev set is - ", accuracy_dev)
	print("accuracy on test set is - ", accuracy_test)
	print("\n")



