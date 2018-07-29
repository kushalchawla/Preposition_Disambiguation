#emb_Size - word embedding size

import tensorflow as tf
import numpy as np

class MainModel(object):
	def __init__(self, sentMaxl, sentMaxp, sentMaxr, num_classes, wv, emb_size, l2_reg_lambda, learning_rate):
		tf.reset_default_graph()

		#inputs
		self.X_lids = tf.placeholder(tf.int32, [None, sentMaxl], name="X_lids")
		self.X_pids = tf.placeholder(tf.int32, [None, sentMaxp], name="X_pids")
		self.X_rids = tf.placeholder(tf.int32, [None, sentMaxr], name="X_rids")

		self.X_llen = tf.placeholder(tf.int32, [None], name='X_llen')
		self.X_plen = tf.placeholder(tf.int32, [None], name='X_plen')
		self.X_rlen = tf.placeholder(tf.int32, [None], name='X_rlen')

		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		#initialization
		W_wemb = tf.Variable(wv)

		print("W_wemb, ", W_wemb.get_shape(), "--, self.X_lids ", self.X_lids.get_shape())
		emb_l = tf.nn.embedding_lookup(W_wemb, self.X_lids) #N X sentMax X 50
		print("emb_l ", emb_l.get_shape())

		emb_p = tf.nn.embedding_lookup(W_wemb, self.X_pids)
		emb_r = tf.nn.embedding_lookup(W_wemb, self.X_rids)

		emb_l = tf.expand_dims(emb_l, -1)
		print("emb_l expanded, ", emb_l.get_shape())

		emb_p = tf.expand_dims(emb_p, -1)
		emb_r = tf.expand_dims(emb_r, -1)

		#AVG POOLING
		pooled_l = tf.nn.avg_pool(emb_l, ksize=[1, sentMaxl, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pooll")	#Nx1x50x1
		pooled_p = tf.nn.avg_pool(emb_p, ksize=[1, sentMaxp, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="poolp")
		pooled_r = tf.nn.avg_pool(emb_r, ksize=[1, sentMaxr, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="poolr")

		print("pooled_l " , pooled_l.get_shape())

		final_l = tf.reshape(pooled_l, [-1, emb_size])
		final_p = tf.reshape(pooled_p, [-1, emb_size])
		final_r = tf.reshape(pooled_r, [-1, emb_size])

		print("final_l is ", final_l.get_shape())

		XX = tf.concat(1, [final_l, final_p, final_r])
		print(XX.get_shape())

		#########################
		# dropout layer	 
		h = tf.nn.dropout(XX, self.dropout_keep_prob)
		h = tf.nn.relu(h)

		#hidden to output layer operations.
		W = tf.Variable(tf.truncated_normal([3*emb_size, num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

		scores = tf.nn.xw_plus_b(h, W, b, name="scores")			#200x8
		print('score', scores.get_shape())

		self.predictions = tf.argmax(scores, 1, name="predictions")
		losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
		self.loss = tf.reduce_mean(losses)  + l2_reg_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

		self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

		self.optimizer = tf.train.AdamOptimizer(learning_rate)
		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)  
		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver(max_to_keep=50)


	def train_step(self, batch_lids, batch_pids, batch_rids, batch_llen, batch_plen, batch_rlen, batch_y, drop_out):
		feed_dict = {
		self.X_lids: batch_lids,
		self.X_pids: batch_pids,
		self.X_rids: batch_rids,

		self.X_llen: batch_llen,
		self.X_plen: batch_plen,
		self.X_rlen: batch_rlen,

		self.input_y: batch_y,
		self.dropout_keep_prob: drop_out
		}

		_, step, loss, accuracy, predictions = self.sess.run([self.train_op, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
		
		return loss, accuracy

	def test_step(self, batch_lids, batch_pids, batch_rids, batch_llen, batch_plen, batch_rlen, batch_y):
		feed_dict = {
		self.X_lids : batch_lids,
		self.X_pids : batch_pids,
		self.X_rids : batch_rids,

		self.X_llen : batch_llen,
		self.X_plen : batch_plen,
		self.X_rlen : batch_rlen,

		self.input_y: batch_y,
		self.dropout_keep_prob: 1.0
		}

		accuracy = self.sess.run([self.accuracy], feed_dict)
		
		return accuracy

