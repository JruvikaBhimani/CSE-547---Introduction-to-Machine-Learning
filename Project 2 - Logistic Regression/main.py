import numpy as np
#import xlrd
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot1

def cluster(M, data_set):
    kmeans = KMeans(n_clusters=M, random_state=0).fit(data_set)
    mu = kmeans.cluster_centers_
    return mu
    
def feature_mat(M, data_set, mu):
    phi=np.zeros((len(data_set),M))
    mu_x = np.zeros(len(data_set[1,:]))

    for i in range(len(data_set)):
        for j in range(M):
            mu_x = data_set[i] - mu[j]
            mu_transpose = mu_x.transpose()
            temp = np.dot(mu_x, mu_transpose)
            phi[i,j] = math.exp((-0.5)* temp)
            #phi[i,j] = math.exp((-0.5)* mu_x * mu_transpose)
    #print phi
    
    #print "phi: " 
    #print phi.shape
    #print "phi_transpose: %f" % phi_transpose.shape()
    #phi_transpose = phi.transpose()
    
    #print "phi: %f" % len(phi)
    #print "len of phi[1,:]:  %d" %len(phi[1,:]) 
    
    #print "phi_transpose: %f" % len(phi_transpose)
    #print "len of phi_transpose[1,:]:  %d" %len(phi_transpose[1,:])
    return phi
    
def weight_ML(data_set, phi):
    temp = np.linalg.pinv(phi)
    
    wML = np.dot(temp, target_train_set)
   
    #print "target:"
    #print target_train_set.shape
    #print "wML: %f" 
    #print wML.shape
    
    #print "wML: %f" % len(wML)
    #print "len of wML[1,:]:  %d" %len(wML[1,:]) 
    
    return wML
    
def eRMS(data_set, wML, phi, M):
 #   print "xxx inside erms xxx"

    wML_transpose = wML.transpose()
    sum = 0.0
    
    for i in range(len(data_set)):
 #       print "data_set[i]:"
  #      print data_set[i]
   #     print "mult:"
        temp3 = data_set[i] - np.dot(wML_transpose, phi[i,:])
        sum = sum + math.pow(temp3, 2)
  #  print "Sum:"
  #  print sum
    E = (0.5) * sum
    
    Erms = math.sqrt((2 * E)/len(data_set))
    
   # print "Erms:"
   # print Erms
    return Erms
    
def SGD_w(data_set, phi, M, eta, lamb):
    w = np.ones(M)
    Erms = 0
    count = 0
 #   flag ='false'
   # print "len:"
   # print len(data_set)
   # print "M"
    #print M
    #print "phi"
 #   print phi
  #  print "eta"
   # print eta
    #print "lamb"
    #print lamb
    for i in range(100):
        for j in range(len(data_set)):
            temp = target_train_set[j] - np.dot(w.transpose(), phi[j,:])
            delta_Ed = (-1) * temp * phi[j,:]
            delta_w = (-1) * delta_Ed * eta
            #print "delta_w"
            #print delta_w
            #print "w[j]"
            #print w[j]
            w = (lamb * w) +  delta_w 
        old_Erms = Erms
        Erms = eRMS(data_set, w, phi, M)
        #print "ERMS xxxx"
        #print Erms
        if(Erms == old_Erms):
            count =  count + 1
        else:
            count  = 0
        if(count >= 10):
                break

       # print "Stochastic Gradient Descent:"
    print ("w SGD:")
    print (w)
    print ("Erms SGD:")
    print (Erms)
    return w

def graph_plot(x, y, xlab, ylab):
    #plot1.figure(num = 1, figsize =(15,10), dpi = 72)
    #plot1.subplot(321)
    #plot1.scatter(CS_Score,Res_OH)
    plot1.plot(x, y, 'g^')
    plot1.xlabel(xlab)
    plot1.ylabel(ylab)
    plot1.show()

    
if __name__ == "__main__":
    print ("\n\nUBitName = jruvikam")
    print ("personNumber = 50207613")
    #book = xlrd.open_workbook('Querylevelnorm_X.csv')
    #sheet = book.sheet_by_index(0)
    X_data = np.genfromtxt('Querylevelnorm_X.csv',delimiter = ',', dtype=np.float64)
    target_data = np.genfromtxt('Querylevelnorm_t.csv',delimiter = ',', dtype=np.float64)
    train = int(math.floor(0.8 * len(X_data)))
    validate = int(math.floor(0.1*len(X_data)))
    train_set = X_data[0:train,:]
    validate_set = X_data[(train+1):(train+validate),:]
    test_set = X_data[(train+validate+1):,:]
    
    train = int(math.floor(0.8 * len(target_data)))
    validate = int(math.floor(0.1*len(target_data)))
    target_train_set = target_data[0:train]
    target_validate_set = target_data[(train+1):(train+validate)]
    target_test_set = target_data[(train+validate+1):]
    
    eta = 0.01
    lamb = 1 
    Marr = np.zeros(12)
    Ermsarr = np.zeros(12)
    
    print ("\n\n##################        LETOR        ##################")
    
    print ("\n****************** Training Set ******************")
    for M in range(1,12):
        print("M= %d" %M)
        mu = cluster(M, train_set)
        
      #  print "len train_set: %d" %len(train_set)
      #  print "len train_set[1,:]: %d" %len(train_set[1,:])
      #  print "M: %d" %M
      #  print "len of mu:  %d" %len(mu)
      #  print "len of mu[1,:]:  %d" %len(mu[1,:])
        
        phi = feature_mat(M, train_set, mu)
        
        wML = weight_ML(target_train_set, phi)
      
        
        Erms = eRMS(target_train_set, wML, phi, M)
        
        print ("M: %f" %M)
       # print "mu:"
       # print mu
        print ("lambda: %f" %lamb)
        print "phi"
        print phi
        print ("eta: %f" %eta)
        print ("sigma: Identity matrix")
        print ("wML:")
        print (wML)
        
        print ("ERMS:")
        print (Erms)
        
        Marr[M-1] = M
        Ermsarr[M-1] = Erms
        
        wSGD = SGD_w(target_train_set, phi, M, eta, lamb)
    
    graph_plot(Marr, Ermsarr, "M-LeToR", "Erms-LeToR")

    print ("\n****************** Validation Set ******************")
    M= 10
    #mu = cluster(M, validate_set)
    phi = feature_mat(M, validate_set, mu)
    
 #   print "len:"
 #   print len(y_linear)
 #   print "M"
 #   print M
  #  print "phi"
   # print phi
    #print "eta"
 #   print eta
  #  print "lamb"
 #   print lamb
 #   print "y:"
 #   print y_linear[:]
 #   print "wML:"
 #   print wML
    Erms_linear_validate = eRMS(target_validate_set, wML, phi, M)
    print ("M: %f" %M)
 #   print "mu:"
 #   print mu
    print ("lambda: %f" %lamb)
    print ("eta: %f" %eta)
    print ("sigma: Identity matrix")
    print ("wML:")
    print (wML)
    print ("ERMS:")
    print (Erms_linear_validate)
    Erms_SGD_validate = SGD_w(target_validate_set, phi, M, eta, lamb)
   # print "ERMS SGD main:"
   # print Erms_SGD_validate
    
    print ("\n****************** Testing Set ******************")
    
    #mu = cluster(M, test_set)
    phi = feature_mat(M, test_set, mu)

    Erms_linear_test = eRMS(target_test_set, wML, phi, M)
    print ("M: %f" %M)
 #   print "mu:"
 #   print mu
    print ("lambda: %f" %lamb)
    print ("eta: %f" %eta)
    print ("sigma: Identity matrix")
    print ("wML:")
    print (wML)
    print ("ERMS:")
    print (Erms_linear_test)
    Erms_SGD_test = SGD_w(target_test_set, phi, M, eta, lamb)
  #  print "ERMS SGD main:"
  #  print Erms_SGD_test
    
 #   print "mu_x:" 
 #   print mu_x
 #   print "mu_transpose:" 
 #   print mu_transpose
#    print "phi"
#    print phi
       
           
    #print "phi: %f" % len(phi)
    #print "len of phi[1,:]:  %d" %len(phi[1,:])
    #phi_transpose = phi.transpose()
    
   # phi_x = math.exp((-0.5) * phi * phi_transpose)
    
    print ("\n\n##################        SYNTHETIC DATA        ##################")

    X_data = np.genfromtxt('input.csv',delimiter = ',', dtype=np.float64)
    target_data = np.genfromtxt('output.csv',delimiter = ',', dtype=np.float64)
    train = int(math.floor(0.8 * len(X_data)))
    validate = int(math.floor(0.1*len(X_data)))
    train_set = X_data[0:train,:]
    validate_set = X_data[(train+1):(train+validate),:]
    test_set = X_data[(train+validate+1):,:]
    
    train = int(math.floor(0.8 * len(target_data)))
    validate = int(math.floor(0.1*len(target_data)))
    target_train_set = target_data[0:train]
    target_validate_set = target_data[(train+1):(train+validate)]
    target_test_set = target_data[(train+validate+1):]
    
    print ("\n****************** Training Set ******************")
    
    for M in range(1,12):
        print("M= %d" %M)
        mu = cluster(M, train_set)
        
        phi = feature_mat(M, train_set, mu)
        
        wML = weight_ML(target_train_set, phi)
        
        Erms = eRMS(target_train_set, wML, phi, M)
        print ("M: %f" %M)
      #  print "mu:"
      #  print mu
        print ("lambda: %f" %lamb)
        print ("eta: %f" %eta)
        print ("sigma: Identity matrix")
        print ("wML:")
        print (wML)
        print ("ERMS:")
        print (Erms)
        Marr[M-1] = M
        Ermsarr[M-1] = Erms
        wSGD = SGD_w(target_train_set, phi, M, eta, lamb)
    graph_plot(Marr, Ermsarr, "M-Syn", "Erms-Syn")
    
    print ("\n****************** Validation Set ******************")
    
    M=10
   # mu = cluster(M, validate_set)
    phi = feature_mat(M, validate_set, mu)

    Erms_linear_validate = eRMS(target_validate_set, wML, phi, M)
    print ("M: %f" %M)
   # print "mu:"
   # print mu
    print ("lambda: %f" %lamb)
    print ("eta: %f" %eta)
    print ("sigma: Identity matrix")
    print ("wML:")
    print (wML)
    print ("ERMS:")
    print (Erms_linear_validate)
    Erms_SGD_validate = SGD_w(target_validate_set, phi, M, eta, lamb)
    
    print ("\n****************** Testing Set ******************")
    
   # mu = cluster(M, test_set)
    phi = feature_mat(M, test_set, mu)
    
    Erms_linear_test = eRMS(target_test_set, wML, phi, M)
    print ("M: %f" %M)
 #   print "mu:"
 #   print mu
    print ("lambda: %f" %lamb)
    print ("eta: %f" %eta)
    print ("sigma: Identity matrix")
    print ("wML:")
    print (wML)
    print ("ERMS:")
    print (Erms_linear_test)
    Erms_SGD_test = SGD_w(target_test_set, phi, M, eta, lamb)

    print ("EXIT")
    
