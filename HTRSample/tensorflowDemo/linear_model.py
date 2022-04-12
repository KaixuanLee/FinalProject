import numpy as np
#import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf

import numpy as np
import matplotlib.pylab as plt

# Preparing to learn data
train_X = np.random.normal(1, 5, 200)
train_Y = 0.5*train_X+2+np.random.normal(0, 1, 200)
L = len(train_X) 

epoch = 200 
learn_rate = 0.005 

temp_graph = tf.Graph()
with temp_graph.as_default():
   X = tf.placeholder(tf.float32) 
   Y = tf.placeholder(tf.float32)
   k = tf.Variable(np.random.randn(), dtype=tf.float32)
   b = tf.Variable(0, dtype=tf.float32) 
   linear_model = k*X+b # linear model
   cost = tf.reduce_mean(tf.square(linear_model - Y)) # cost 
   optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate) # GradientDescent
   train_step = optimizer.minimize(cost)
   init = tf.global_variables_initializer()
train_curve = []
var = tf.get_variable("var_name", [5], initializer = tf.zeros_initializer) # 定义
saver = tf.train.Saver({"var_name": var}) # 不指定变量字典时保存所有变量
with tf.Session(graph=temp_graph) as sess:
   sess.run(init)
   for i in range(epoch):
       sess.run(train_step, feed_dict={X: train_X, Y: train_Y}) 
       temp_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
       train_curve.append(temp_cost) 
   kt_k = sess.run(k); kt_b = sess.run(b)
   Y_pred = sess.run(linear_model, feed_dict={X: train_X})
   file_writer = tf.summary.FileWriter('./HTRSample/user_log_path', sess.graph) # 输出文件
saver.save(sess, "./HTRSample/model_save/model.ckpt")
#saver.restore(sess, "./model.ckpt")

ax1 = plt.subplot(1, 2, 1); ax1.set_title('Linear model fit');
ax1.plot(train_X, train_Y, 'b.'); ax1.plot(train_X, Y_pred, 'r-')
ax2 = plt.subplot(1, 2, 2); ax2.set_title('Training curve');
ax2.plot(train_curve, 'r--')
plt.show();
