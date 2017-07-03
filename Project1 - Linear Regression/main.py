import numpy
from scipy.stats import norm
#import matplotlib.pyplot as plot1
from math import log1p, pi
import xlrd

def log_like(y, x):
    x0 = 1
    a11 = 0 
    a12 = 0
    a21 = 0
    a22 = 0
    
    for i in range(49):
        a11 = a11 + (x0 * x0)
        a12 = a12 + (x[i] * x0)
        a21 = a21 + (x0 * x[i])
        a22 = a22 + (x[i] * x[i])
    a = ([a11, a12],[a21, a22])
#    print "a:"
#    print a

    y11 = 0
    y21 = 0
    i = 0
    for i in range(49):
        y11 = y11 + (y[i] * x0)
        y21 = y21 + (y[i] * x[i])
    y1 = ([y11],[y21])
#    print "y:"
#    print y1
    
    beta = numpy.linalg.solve(a,y1)
#    print "beta:"
#    print beta
    
    sigma = 0
    i=0
    for i in range(49):
        term1 = beta[0] * x0
        term2 = beta[1] * x[i]
        sigma = sigma + ((term1+term2-y[i])**2)
    
    sigma = sigma/49

 #   print "sigma: %.3f" % sigma    
    
    likelihood = 0
    i = 0
    
    for i in range(49):
        term1 = log1p(2*pi*sigma)
        term2 = (beta[0]*x0)+(beta[1]*x[i])-y[i]
        term2sq = term2**2
        term3 = 2*sigma
        term4 = 1/term3
        likelihood = likelihood + (((-0.5)*term1)-(term4*term2sq))
    
#    print "likelihood: %.3f" % likelihood
    return likelihood

if __name__ == "__main__":
    print "UBitName = jruvikam"
    print "personNumber = 50207613"
    book = xlrd.open_workbook('university data.xlsx')
    sheet = book.sheet_by_index(0)

    CS_Score = numpy.asarray(sheet.col_values(2)[1:50])
    Res_OH = numpy.asarray(sheet.col_values(3)[1:50])
    Admin_Base_Pay = numpy.asarray(sheet.col_values(4)[1:50])
    Tuition_OS = numpy.asarray(sheet.col_values(5)[1:50])
    
    #univ_data = genfromtxt('/Users/Juvi/Desktop/UB/univ data.csv',delimiter = ',', dtype=numpy.float64)
    #mean = numpy.mean(univ_data, axis = 0, dtype = numpy.float64) 
    #mu1 = mean[0]
    #mu2 = mean[1]    
    #mu3 = mean[2]    
    #mu4 = mean[3]
    mu1 = numpy.mean(CS_Score, dtype = numpy.float64)
    mu2 = numpy.mean(Res_OH, dtype = numpy.float64)
    mu3 = numpy.mean(Admin_Base_Pay, dtype = numpy.float64)
    mu4 = numpy.mean(Tuition_OS, dtype = numpy.float64)
    #print "mean:"
    #print mean
    print "mu1 = %.3f" % mu1
    print "mu2 = %.3f" % mu2
    print "mu3 = %.3f" % mu3
    print "mu4 = %.3f" % mu4
    
    #var = numpy.var(univ_data, axis = 0, dtype = numpy.float64)
    var1 = numpy.var(CS_Score, dtype = numpy.float64)
    var1 = (var1*49)/48
    var2 = numpy.var(Res_OH, dtype = numpy.float64)
    var2 = (var2*49)/48
    var3 = numpy.var(Admin_Base_Pay, dtype = numpy.float64)
    var3 = (var3*49)/48
    var4 = numpy.var(Tuition_OS, dtype = numpy.float64)
    var4 = (var4*49)/48
#    var2 = (var[1]*49)/48   
 #   var3 = (var[2]*49)/48   
  #  var4 = (var[3]*49)/48
    print "var1: %.3f" % var1
    print "var2: %.3f" % var2
    print "var3: %.3f" % var3
    print "var4: %.3f" % var4
    
    #std = numpy.std(univ_data, axis = 0, dtype = numpy.float64)
    std1 = numpy.std(CS_Score, dtype = numpy.float64)
    std2 = numpy.std(Res_OH, dtype = numpy.float64)
    std3 = numpy.std(Admin_Base_Pay, dtype = numpy.float64)
    std4 = numpy.std(Tuition_OS, dtype = numpy.float64)
    #print "std:"
    print "sigma1: %.3f" % std1
    print "sigma2: %.3f" % std2
    print "sigma3: %.3f" % std3
    print "sigma4: %.3f" % std4
    
    univ_data = [CS_Score, Res_OH, Admin_Base_Pay, Tuition_OS]
    #print univ_data
    
    cov = numpy.cov(univ_data)
    print "covarianceMat = "
    numpy.set_printoptions(precision=3)
    print cov
    
    cor = numpy.corrcoef(univ_data)
    print "correlationMat = "
    print cor

    #CS_Score = univ_data[:,0]
    #Res_OH = univ_data[:,1]
    #Admin_Base_Pay = univ_data[:,2]
    #Tuition_OS = univ_data[:,3]

 #   plot1.figure(num = 1, figsize =(15,10), dpi = 72)
#    plot1.subplot(321)
#    plot1.scatter(CS_Score,Res_OH)
#    plot1.xlabel("CS Score (US News)")
#    plot1.ylabel("Research Overhead")

#    plot1.subplot(322)    
#    plot1.scatter(CS_Score,Admin_Base_Pay)
#    plot1.xlabel("CS Score (US News)")
#    plot1.ylabel("Admin Base Pay $")

#    plot1.subplot(323)    
#    plot1.scatter(CS_Score,Tuition_OS)
#    plot1.xlabel("CS Score (US News)")
#    plot1.ylabel("Tuition Out-State $")

#    plot1.subplot(324)
#    plot1.scatter(Res_OH,Admin_Base_Pay)
#    plot1.xlabel("Research Overhead")
#    plot1.ylabel("Admin Base Pay $")

#    plot1.subplot(325)
#    plot1.scatter(Res_OH,Tuition_OS)
#    plot1.xlabel("Research Overhead")
#    plot1.ylabel("Tuition Out-State $")

#    plot1.subplot(326)
#    plot1.scatter(Admin_Base_Pay,Tuition_OS)
#    plot1.xlabel("Admin Base Pay $")
#    plot1.ylabel("Tuition Out-State $")

    pdf = norm.logpdf(CS_Score,mu1,std1)
    pdf1 = sum(pdf)
    pdf = norm.logpdf(Res_OH,mu2,std2)
    pdf2 = sum(pdf)
    pdf = norm.logpdf(Admin_Base_Pay,mu3,std3)
    pdf3 = sum(pdf)
    pdf = norm.logpdf(Tuition_OS,mu4,std4)
    pdf4 = sum(pdf)
    
    logLikelihood = pdf1 + pdf2 + pdf3 +pdf4
    print "logLikelihood = %.3f" % logLikelihood
    
    BNGraph = numpy.asmatrix([[0,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,1,0]])
    print "BNGraph = "
    print BNGraph
    
    likelihood1 = log_like(Admin_Base_Pay, Tuition_OS)
    likelihood2 = log_like(Tuition_OS, CS_Score)
    likelihood3 = log_like(Res_OH, CS_Score)
    
    BNlogLikelihood = pdf1 + likelihood1 + likelihood2 + likelihood3
    
    print "BNlogLikelihood = % .3f" % BNlogLikelihood
    


    
