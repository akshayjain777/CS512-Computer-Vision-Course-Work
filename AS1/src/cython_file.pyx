#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:00:50 2022

@author: akshay
"""

import numpy as np

dtype = np.double

def compute(int[:,:,:] image):

    cdef Py_ssize_t x_max = image.shape[0]
    cdef Py_ssize_t y_max = image.shape[1]

    result = np.zeros((x_max, y_max),dtype = dtype)
    cdef double[:, :] out_Bw_lmg = result

    cdef int tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):
              Y = image[x, y, 0] * 0.299 + image[x, y, 1] * 0.587 + image[x ,y, 2] * 0.114
              out_Bw_lmg[x, y] = Y
            
 
    return result



