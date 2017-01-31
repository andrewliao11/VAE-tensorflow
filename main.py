import tensorflow as tf
import numpy as np
import os, argparse, pdb, multiprocessing
import nn, construct_binary, reader
from utils import *

class Model(object):

    def __init__(self, z_dims=20, scope='mnist'):
	self.z_dims = z_dims
	self.scope = scope


    def _get_latent_code(self, x):
	x = tf.to_float(x)/255.
	fc1 = nn.fc(x, 400, nl=tf.nn.relu, scope='e_fc1')
	mu = nn.fc(fc1, self.z_dims, scope='e_fc_mu')
	log_sigma_sq = nn.fc(fc1, self.z_dims, scope='e_fc_sigma')
	return mu, log_sigma_sq

    def _reconstruct(self, z):
	fc1 = nn.fc(z, 400, nl=tf.nn.relu, scope='d_fc1')
	_x = nn.fc(fc1, self.x_dims, nl=tf.nn.sigmoid, scope='d_fc2')
	return _x

    def _build_graph(self, x):
	with tf.variable_scope(self.scope):
	    input_shape = x.get_shape().as_list()
	    assert len(input_shape) == 2
	    batch_size, self.x_dims = input_shape
	    mu, log_sigma_sq = self._get_latent_code(x)
	    # N(z;0,I), unit gaussian
	    eps = tf.random_normal((batch_size, self.z_dims), 0, 1, 
				dtype=tf.float32) 
	    # z = mu+sigma*eps
	    self.z = tf.add(mu, tf.mul(tf.sqrt(tf.exp(log_sigma_sq)), eps))
	    self._x = self._reconstruct(self.z)
	    # likelihood of Bernoulli
	    # reference: https://onlinecourses.science.psu.edu/stat504/node/27
	    x_float = tf.to_float(x)/255.
	    likelihood_loss = -tf.reduce_sum(x_float*tf.log(1e-8 + self._x) + \
				(1-x_float)*tf.log(1e-8 + 1 - self._x), 1)	# B
	    # kl divergence (prior: gaussian)
	    # reference: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
	    kl_divergence = -0.5 * tf.reduce_sum(1 + log_sigma_sq \
				- tf.square(mu) - tf.exp(log_sigma_sq), 1)	# B
	    self.cost = tf.reduce_mean(tf.add(likelihood_loss, kl_divergence))
	    tf.summary.image("input_images", tf.reshape(x_float, [-1,28,28,1]))
	    tf.summary.image("output_images", tf.reshape(self._x, [-1,28,28,1]))
	    tf.summary.scalar("likelihood_loss", tf.reduce_mean(likelihood_loss))
	    tf.summary.scalar("kl_divergence", tf.reduce_mean(kl_divergence))
	    tf.summary.scalar("cost", self.cost)

    def _get_all_params(self):
	params = []
	for param in tf.global_variables():
	    if self.scope in param.name:
		params.append(param)
	return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-thread mnist classifier')
    parser.add_argument('--lr', type=float, default=3e-4,
                    help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    bin_filepath = 'mnist.tfrecords'
    if not os.path.exists(bin_filepath):
    	construct_binary.mnist(bin_filepath)
    # convert the string into tensor (represent a single tensor)
    single_label, single_image = reader.read_to_tensor(bin_filepath) 
    images_batch, labels_batch = tf.train.shuffle_batch(
    	[single_image, single_label], batch_size=args.batch_size,
   	capacity=2000,
    	min_after_dequeue=1000
    )
    # build graph
    M = Model()
    M._build_graph(images_batch)
    global_step = tf.get_variable('global_step', [], 
			initializer=tf.constant_initializer(0), trainable=False)
    train_op = tf.train.AdamOptimizer(args.lr).minimize(M.cost, global_step=global_step)
    if not os.path.exists(bin_filepath):
	os.makdir('./logs')
    summary_writer = tf.summary.FileWriter('./logs')
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=20)

    config = get_session_config(0.3, multiprocessing.cpu_count()/2)
    sess = tf.Session(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # creates threads to start all queue runners collected in the graph
    # [remember] always call init_op before start the runner
    tf.train.start_queue_runners(sess=sess)
    step = 0
    while True:
  	_, summary_str, loss= sess.run([train_op, summary_op, M.cost])
	summary_writer.add_summary(summary_str, step)
	if step%100 == 0:
	    if not os.path.exists('./checkpoints'):
		os.mkdir('./checkpoints')
	    saver.save(sess, os.path.join('./checkpoints', 'mnist'), global_step=global_step)
	    print "==================================="
	    print "[#] Iter", step
	    print "[L] Loss =", loss
	step += 1
