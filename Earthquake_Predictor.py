
# coding: utf-8

# # EarthQauke Predictor

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt




data = pd.read_csv("data/earthquake_data.csv")


col1 = data[["Latitude","Longitude", "Depth"]]
col2= data["Magnitude"]




InputX = col1.as_matrix()
Inputy = col2.as_matrix()






InputX.astype(float, copy = False)
Inputy.astype(float, copy = False)





X_min = np.amin(InputX,0)
X_max = np.amax(InputX,0)
y_min = np.amin(Inputy,0)
y_max = np.amax(Inputy,0)





InputX_norm = (InputX-X_min)/ (X_max-X_min)






Inputy_norm = Inputy


X_feature = 3
y_feature = 1
sample = 23000 
InputX_reshape = np.resize(InputX_norm,(sample,X_feature))
Inputy_reshape = np.resize(Inputy_norm,(sample,y_feature))



batch_size = 20000
InputX_train = InputX_reshape[0:batch_size,:]
InputX_train.shape





Inputy_train = Inputy_reshape[0:batch_size,:]





v_size = 2500
InputX_test = InputX_reshape[batch_size: batch_size+v_size,:]





Inputy_test = Inputy_reshape[batch_size: batch_size+v_size,:]



# ### Build and Train Model

print("Building Model........")

learning_rate = 0.01
training_iterations = 10000
display_iterations = 20000


X = tf.placeholder(tf.float32,shape=(None,X_feature))
y = tf.placeholder(tf.float32, shape=(None,y_feature))

L1 = 3 
L2 = 3
L3 = 3
w_fc1 = tf.Variable(tf.random_uniform([X_feature,L1]))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[3]))
w_fc2 = tf.Variable(tf.random_uniform([L1,L2]))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[L2]))
w_fc3 = tf.Variable(tf.random_uniform([L2,L3]))
b_fc3 = tf.Variable(tf.constant(0.1, shape=[L3]))
W_fo = tf.Variable(tf.random_uniform([L3,y_feature]))
b_fo = tf.Variable(tf.constant(0.1,shape=[y_feature]))

matmul_fc1 = tf.matmul(X,w_fc1) + b_fc1
h_fc1 = tf.nn.relu(matmul_fc1) 
matmul_fc2 = tf.matmul(h_fc1,w_fc2) + b_fc2
h_fc2 = tf.nn.relu(matmul_fc2)
matmul_fc3 = tf.matmul(h_fc2,w_fc3) + b_fc3
h_fc3 = tf.nn.relu(matmul_fc3)
matmul_fc4 = tf.matmul(h_fc3,W_fo) + b_fo
output_layer = matmul_fc4

mean_sq = tf.reduce_mean(tf.square(y-output_layer))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_sq)


saver = tf.train.Saver()




init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("Training_Loss:", sess.run([mean_sq], feed_dict={X:InputX_train,y:Inputy_train}))
    for i in range(training_iterations):
        sess.run([train_step], feed_dict={X:InputX_train,y:Inputy_train})
        if i%display_iterations ==0:
            print("Training_loss is :", sess.run(mean_sq, feed_dict={X:InputX_train,y:Inputy_train}),"at_iteration:", i)
            print("validation_loss is :", sess.run(mean_sq, feed_dict={X:InputX_test, y:Inputy_test}),"at_iteration:", i)
    
    save_path = saver.save(sess, "data/earthquake_model.ckpt")
    print("Model saved in file: %s" % save_path)
    print("final_training_loss:", sess.run(mean_sq, feed_dict={X:InputX_train, y:Inputy_train}))
    print("final_validation_loss", sess.run(mean_sq, feed_dict={X:InputX_test, y:Inputy_test}))


# ### Make Prediction



lat = input("Enter latitude between -77 to 86 :")
long = input("Enter Longitude between -180 to 180:")
depth = input("Enter Depth between 0 to 700:")
InputX2 = np.asarray([lat, long, depth], dtype = np.float32)
InputX2_norm = (InputX2-X_min)/ (X_max-X_min)
InputX1_test = np.resize(InputX2_norm,(1,X_feature))
with tf.Session() as sess:
        saver.restore(sess, "/tmp/earthquake_model.ckpt")
        print("Model Restored.")
        print("output:", sess.run(output_layer, feed_dict={X:InputX1_test}), "Magnitude")

