from __future__ import division
import cPickle
import numpy as np
import math
import random
import os as os
from scipy import misc
from skimage import color
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#import matplotlib.pyplot as plot1

#def graph_plot(x, y, xlab, ylab):
    #plot1.figure(num = 1, figsize =(15,10), dpi = 72)
    #plot1.subplot(321)
    #plot1.scatter(CS_Score,Res_OH)
#    plot1.plot(x, y, 'g^')
#    plot1.xlabel(xlab)
#    plot1.ylabel(ylab)
#    plot1.show()

def oneHotEncoding(target):
    print ("oneHotEncoding")
    print (np.shape(target))
    t = np.zeros((len(target),10))
    print (np.shape(t))
   # print "entering for:"
    for i in range(len(target)):
        index = target[i]
    #    print target[i]
    #    print index
        t[i][index] = 1
    #    print t
    return t

def gradientErrorFunction(x, t, y):
    print ("gradientErrorFunction:")
    print (np.shape(y))
    print (np.shape(t))
    print (np.shape(x))
    temp = y - t
    xMat = np.matrix(x)
    tempMat = np.matrix(temp)
    
    deltaE = np.dot(tempMat.transpose(),xMat)
    print (np.shape(deltaE))
    return deltaE

def SGD_w(deltaE, eta, w):
    print ("SGD_w:")
    print (np.shape(deltaE))
    print (np.shape(w))
    
  #  print len(deltaE)
  #  print len(deltaE[0]) 
    
   # print len(w)
    #print len(w[0])
    
   # deltaE = (eta * deltaE)
    wnew = w - (eta * deltaE)
   # print len(deltaE)
   # print len(deltaE[0]) 
    print (np.shape(wnew))
    print (len(wnew))
    print (len(wnew[0]))
    return wnew
    
def activationfn(x, w, b):
    print ("activation:")
    a = np.zeros(10)
    
    print (np.shape(w))
    print (np.shape(x))
    xMat = np.matrix(x)
    a = np.dot(w,xMat.transpose())
    print (np.shape(a))
    return a
    
def calculate_y(a):
    print ("calculate_y:")
    print (len(a))
    print (np.shape(a))

    sum_a = 0
    c=max(a)
    
    for i in range(len(a)):
        sum_a = sum_a + (math.exp(a[i]-c))
    
    y = np.zeros(len(a))    
    
    for i in range(len(a)):
        y[i] = (math.exp(a[i]-c))/sum_a
        
    sum_y = sum(y)
    
    print (sum_y)

    return y
    
def hiddenLayerActivation(x, w, b):
    print ("hiddenLayerActivation:")
    print (np.shape(w))
    print (np.shape(x))
    print (len(w))
    print (np.shape(w[0]))
    
    
    row,col = np.shape(w)

    
    z = np.zeros(row)
    
  #  print "**************************************************"
    
    for i in range(row):
        for j in range(col):
      #      print x[j]
       #     print w[i][j], i, j
        #    print "print"
            z[i] = z[i] + (w[i][j] * x[j])
        z[i] = z[i] + b
        #print "##########################################"
    
    return z
    
def hFunction(z):
    hz = np.zeros(len(z))
    hdashA = np.zeros(len(z))
    for i in range(len(z)):
        hz[i] = 1/(1 + math.exp(-z[i]))
        hdashA[i] = hz[i] * (1 - hz[i])
    return hz,hdashA
    
def gradientErrorFunctionNNLayer2(y, t, z):
    print ("gradientErrorFunctionNNLayer2:")
    d = y - t
    dk = np.matrix(d)
  
    print (np.shape(dk))
  
    zmat = np.matrix(z)
    
    print (np.shape(zmat))
    
    error_dk = np.dot(dk.transpose(), zmat)
    
   # row, col = np.shape(error)
    
  #  error_dk = np.zeros((row,col))
    
   # for i in range(row):
    #    temp = error[i][:]
     #   print (temp)
      #  print np.shape(temp)
       # print len(temp[0][:])
        #temp1 = temp[0][:]
       # print temp1
    #    for j in range(col):
     #       error_dk[i][j] = temp1[j]
    
    print (np.shape(error_dk))
    print (len(error_dk))
    print (len(error_dk[0]))
   
    return dk.transpose(), error_dk    
    
def gradientErrorFunctionNNLayer1(hdashA, w, dk, x):
    print ("gradientErrorFunctionNNLayer1:")
    print (np.shape(hdashA))
    print (np.shape(w))
    print (len(w))
    print (len(w[0]))
    print (np.shape(dk))
    dj = np.zeros(len(hdashA))
    
    for j in range(len(dj)):
        sum_w = 0
        for k in range(len(w)):
            sum_w = sum_w + (w[k][j] * dk[k])
        dj[j] = hdashA[j] * sum_w
    print (np.shape(dj))
    xmat = np.matrix(x)
    print (np.shape(xmat))
    djmat = np.matrix(dj)
    print (np.shape(djmat))
    error_dj = np.dot(djmat.transpose(), xmat)
    print (np.shape(error_dj))
    print (len(error_dj))
    print (len(error_dj[0]))
    return djmat.transpose(), error_dj
    
def softmax(y):
    print ("SoftMax:")
    print (np.shape(y))
    maximum = -1.0
    value = -1
    for i in range(len(y)):
        if(maximum < y[i]):
            maximum = y[i]
            value = i
  #  print value
  #  print "end softmax"
    return value
    
def logRegression(x, t, b):
    print ("logregression:")
    print (len(x[0]))
    
    w = np.ones((10,len(x[0])))
    eta = 0.01
    count = 0
    for j in range(5):
        for i in range(len(x)):
            a = activationfn(x[i][:], w, b)
            y = calculate_y(a)
            deltaE = gradientErrorFunction(x[i][:],t[i][:],y)
            w = SGD_w(deltaE, eta, w)
            count = count + 1
        print ("count:")
        print (count)
    return w
    
def logRegressionValidate(x, t, w, b):
    print ("logRegressionValidate:")
    found = 0.0
    y_value = np.zeros(len(x))
    for i in range(len(x)):
        a = activationfn(x[i][:], w, b)
        y = calculate_y(a)
        value = softmax(y)
 #       print t[i]
        y_value[i] = value
        if(value==t[i]):
            found  = found + 1.0
            
    print ("found:")
    print (found)
    
    accuracy = (found/len(t))*100
    print ("accuracy:")
    print (accuracy)
    return y, y_value, accuracy
    
def logRegressionTest(x, w, b):
    print ("logRegressionTest:")
    y_value = np.zeros(len(x))
    for i in range(len(x)):
        a = activationfn(x[i][:], w, b)
        y = calculate_y(a)
        value = softmax(y)
 #       print t[i]
        y_value[i] = value
    
    return y, y_value
    
def neuralnetwork(x, t, b):
    print ("neuralnetwork:")
    eta = 0.01
   # x = np.insert(input_x,0,0,axis =1)
    print (np.shape(x))
    
    w1 = np.ones((100,len(x[0])))

    print (np.shape(w1))
    print (len(w1[0]))

    for i in range(len(w1)):
        for j in range(len(w1[0])):
            w1[i][j] = random.randrange(0,100,1)
            w1[i][j] = w1[i][j] / 10000
    w2 = np.ones((10,100))
    print (np.shape(w2))

    for i in range(len(w2)):
        for j in range(len(w2[0])):
            w2[i][j] = random.randrange(0,100,1)
            w2[i][j] = w2[i][j] / 10000   
      
   
    for i in range(len(x)):
        z = hiddenLayerActivation(x[i][:], w1, b)
        #z = np.insert(z,0,0)
        hz, hdashA = hFunction(z)
        a = hiddenLayerActivation(hz, w2, b)

        y = calculate_y(a)
        dk, error_dk = gradientErrorFunctionNNLayer2(y, t[i], z)
        dj, error_dj = gradientErrorFunctionNNLayer1(hdashA, w2, dk, x[i][:])

        w1 = SGD_w(error_dj, eta, w1)
        w2 = SGD_w(error_dk, eta, w2)
    
    return w1, w2
    
def neuralnetworkValidate(input_x, t, w1, w2, b):
    print ("neuralnetworkValidate:")
   # x = np.insert(input_x,0,0,axis =1)
    x = np.matrix(input_x)
    print (np.shape(x)) 
    
    found  = 0.0
    y_value = np.zeros(len(x))
   
    for i in range(len(x)):
        z = hiddenLayerActivation(x[i][:], w1, b)
        #z = np.insert(z,0,0)
        hz, hdashA = hFunction(z)
        a = hiddenLayerActivation(hz, w2, b)
        y = calculate_y(a)
        
        value = softmax(y)

        y_value[i] = value
        if(value==t[i]):
            found  = found + 1.0

    
    accuracy = (found/len(t))*100
    print ("accuracy NN:")
    print (accuracy)
    
    return y, y_value, accuracy
    
def neuralnetworkTest(input_x, w1, w2, b):
    print ("neuralnetworkTest:")
   # x = np.insert(input_x,0,0,axis =1)
    x = np.matrix(input_x)
    print (np.shape(x)) 

    y_value = np.zeros(len(x))
   
    for i in range(len(x)):
        z = hiddenLayerActivation(x[i][:], w1, b)
        #z = np.insert(z,0,0)
        hz, hdashA = hFunction(z)
        a = hiddenLayerActivation(hz, w2, b)
        y = calculate_y(a)
        
        value = softmax(y)

        y_value[i] = value
        
    return y, y_value
    
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def cnn():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.global_variables_initializer())
    y = tf.matmul(x,W) + b
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for i in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    

if __name__ == "__main__":
    print ("UBitName = jruvikam")
    print ("personNumber = 50207613")
    pickleFile = open('mnist.pkl','rb')
    train_set_MNIST, valid_set_MNIST, test_set_MNIST = cPickle.load(pickleFile)
    train_x_MNIST = train_set_MNIST[0]
    train_target_MNIST = train_set_MNIST[1]
    train_t_MNIST = oneHotEncoding(train_target_MNIST)
    valid_x_MNIST = valid_set_MNIST[0]
    valid_target_MNIST = valid_set_MNIST[1]
    test_x_MNIST = test_set_MNIST[0]
    test_target_MNIST = test_set_MNIST[1]
    
    b = 1
 # TUNE HYPERPARAMETER ETA
    w_logRegress_MNIST = logRegression(train_x_MNIST, train_t_MNIST, b)
    yOneHot_validate_MNIST, y_value_validate_MNIST, accuracy_validate_MNIST = logRegressionValidate(valid_x_MNIST, valid_target_MNIST, w_logRegress_MNIST, b)
    yOneHot_test_MNIST, y_value_test_MNIST = logRegressionTest(test_x_MNIST, w_logRegress_MNIST, b)
    print ("accuracy MNIST validation:")
    print (accuracy_validate_MNIST)
    
    
    

 
    path = "USPSdata/Numerals/"
    count = 0
    validate_x_USPS = np.zeros((1,784))
    target_set_USPS = np.zeros((1,1))
    print (np.shape(validate_x_USPS))
    for i in range(10):
        new_path = path
        new_path = new_path + str(i) + "/"
       
        for name in os.listdir(new_path):
            final_path = new_path
            final_path = final_path + name
           # print count
            #print final_path
            if ".list" not in name:
                if (name != "Thumbs.db"):
                 #   if count < 5:
                    img = misc.imread(final_path)
                    gray_img = color.rgb2gray(img)
                    resized_img = misc.imresize(gray_img,(28,28))
                #    print "resized img:"
                 #   print len(resized_img)
                  #  print np.shape(resized_img)
                    flat_img = np.ravel(resized_img)
                    validate_x_USPS = np.insert(validate_x_USPS,len(validate_x_USPS),flat_img,axis=0)
                    target_set_USPS = np.insert(target_set_USPS,len(target_set_USPS),int(i),axis=0)
                    #print "resized img:"
                    #print len(flat_img)
                    #print np.shape(flat_img)
                    count = count + 1
                    if((count%1000) == 0):
                        print (count) 
                    
                 
                 #   else:
                  #      break
    print ("count:")
    print (count)
    validate_x_USPS = np.delete(validate_x_USPS,0,axis=0)
    target_set_USPS = np.delete(target_set_USPS,0,axis=0)

    
    yOneHot_validate_USPS, y_value_validate_USPS, accuracy_validate_USPS = logRegressionValidate(validate_x_USPS, target_set_USPS, w_logRegress_MNIST, b)
    
    
    path = "USPSdata/Test/"
    count = 0
    test_x_USPS = np.zeros((1,784))

    for i in range(10):
        new_path = path
       
        for name in os.listdir(new_path):
            final_path = new_path
            final_path = final_path + name
           # print count
            #print final_path
            if ".list" not in name:
                if (name != "Thumbs.db"):
                 #   if count < 5:
                    img = misc.imread(final_path)
                    gray_img = color.rgb2gray(img)
                    resized_img = misc.imresize(gray_img,(28,28))
                #    print "resized img:"
                 #   print len(resized_img)
                  #  print np.shape(resized_img)
                    flat_img = np.ravel(resized_img)
                    test_x_USPS = np.insert(test_x_USPS,len(validate_x_USPS),flat_img,axis=0)
                  
                    #print "resized img:"
                    #print len(flat_img)
                    #print np.shape(flat_img)
                    count = count + 1
                    if((count%1000) == 0):
                        print (count)
                    
                  
                 #   else:
                  #      break
    print ("count:")
    print (count)
    test_x_USPS = np.delete(test_x_USPS,0,axis=0)

    yOneHot_test_USPS, y_value_test_USPS = logRegressionTest(test_x_USPS, w_logRegress_MNIST, b)   

    cnn()
    print ("accuracy USPS validation:")
    print (accuracy_validate_USPS)
    
    print ("accuracy MNIST validation:")
    print (accuracy_validate_MNIST)
    
  #  w1_nn_MNIST, w2_nn_MNIST = neuralnetwork(train_x_MNIST, train_t_MNIST, b)
  #  yOneHot_nn_MNIST, y_value_nn_MNIST, accuracy_nn_MNIST = neuralnetwork(valid_x_MNIST, valid_target_MNIST, w1_nn_MNIST, w2_nn_MNIST, b)
  #  yOneHot_test_nn_MNIST, y_value_test_nn_MNIST = neuralnetwork(test_x_MNIST, w1_nn_MNIST, w2_nn_MNIST, b)
    
  #  yOneHot_nn_USPS, y_value_nn_USPS, accuracy_nn_USPS = neuralnetwork(validate_x_USPS, target_set_USPS, w1_nn_MNIST, w2_nn_MNIST, b)

   # yOneHot_test_nn_USPS, y_value_test_nn_USPS = neuralnetwork(test_x_USPS, w1_nn_MNIST, w2_nn_MNIST, b)   
   
    
    print ("PROGRAM COMPLETED")
