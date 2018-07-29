import random

random.seed(0)
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
dimensions = 50
word_vectors = []

with open(filevec,"r") as fin:
	for line in fin:
		words = line.strip("\n").strip().split()
		if(len(words) == (dimensions + 1)):
			word_vectors.append(words[0])

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


#sentMax padding
sentMax = -1



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
import collections
import numpy as np
from sklearn.metrics import accuracy_score

all_classes = {}

for tup in x_tups_train:
	if(tuple(tup[1]) in all_classes):
		all_classes[tuple(tup[1])].append(tup[-1])
	else:
		all_classes[tuple(tup[1])] = [tup[-1]]


class_freq = {}

for k,v in all_classes.items():
	coll = collections.Counter(v)
	
	coll_class_list = list(coll.keys())
	coll_class_freq = list(coll.values())

	class_freq[k] = (coll_class_list, coll_class_freq)

max_class = {}

for k,v in class_freq.items():
	max_class[k] = v[0][v[1].index(max(v[1]))]

###
all_classes1 = []

for tup in x_tups_train:
	all_classes1.append(tup[-1])

class_freq1 = collections.Counter(all_classes1)

class_list1 = list(class_freq1.keys())
class_freq1 = list(class_freq1.values())

max_freq1 = max(class_freq1)

max_class1 = class_list1[class_freq1.index(max_freq1)]
###

print("model training done.")

######################################
#assign max class as a prediction for all samples.

#predictions

#devset
predictions_dev = []

for tup in x_tups_dev:
	if (tuple(tup[1]) in max_class):
		predictions_dev.append(max_class[tuple(tup[1])])
	else:
		predictions_dev.append(max_class1)

#testset
predictions_test = []

for tup in x_tups_test:
	if (tuple(tup[1]) in max_class):
		predictions_test.append(max_class[tuple(tup[1])])
	else:
		predictions_test.append(max_class1)

#evaluation
#devset

true_classes = []

for tup in x_tups_dev:
	true_classes.append(tup[-1])

accuracy_dev = accuracy_score(y_true=true_classes, y_pred=predictions_dev)

print("accuracy on dev set is - ", accuracy_dev)

#testset

true_classes = []

for tup in x_tups_test:
	true_classes.append(tup[-1])

accuracy_test = accuracy_score(y_true=true_classes, y_pred=predictions_test)

print("accuracy on test set is - ", accuracy_test)


