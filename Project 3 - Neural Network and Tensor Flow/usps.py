from scipy import misc
from skimage import color
import numpy as np
import os as os

if __name__ == "__main__":
    path = "USPSdata/Numerals/"
    count = 0
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
                   # if count < 5:
                    img = misc.imread(final_path)
                    gray_img = color.rgb2gray(img)
                    resized_img = misc.imresize(gray_img,(28,28))
                #    print "resized img:"
                 #   print len(resized_img)
                  #  print np.shape(resized_img)
                    flat_img = np.ravel(resized_img)
                    #print "resized img:"
                    #print len(flat_img)
                    #print np.shape(flat_img)
                    count = count + 1
    print "count:"
    print count