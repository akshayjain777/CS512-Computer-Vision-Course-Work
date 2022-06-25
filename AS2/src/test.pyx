cimport numpy as np 
import numpy as np 

def compute(int[:,:] image):

    cdef Py_ssize_t x_max = image.shape[0]
    cdef Py_ssize_t y_max = image.shape[1]
    kernel = np.ones((3,3),dtype=np.int)/9
    result = np.zeros((x_max, y_max),dtype = np.int)
    cdef int[:, :] out_Bw_lmg = result

    cdef int[:,:] tmp
    cdef int x, y, n
    cdef int[:,:] img_pad = np.pad(image,1)
    for x in range(x_max-3):
        for y in range(y_max-3):
            tmp = img_pad[x:x+3,y:y+3]
            n = np.sum(tmp*kernel)
            out_Bw_lmg[x-1, y-1] = np.int(n)
            
 
    return result
