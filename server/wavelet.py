import numpy as np
import pywt
import cv2    

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    c=pywt.wavedec2(imArray, mode, level=level)

#     #Process Coefficients
#     coeffs_H=list(coeffs)  
#     coeffs_H[0] *= 0;  

#     # reconstruction
#     imArray_H=pywt.waverec2(coeffs_H, mode);
#     imArray_H *= 255;
#     imArray_H =  np.uint8(imArray_H)

    c[0] /= np.abs(c[0]).max()
    for detail_level in range(level):
        c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
    # show the normalized coefficients
    arr, slices = pywt.coeffs_to_array(c)
    return arr