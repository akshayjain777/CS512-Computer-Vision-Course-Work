#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:53:04 2022

@author: akshay
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
      name = "RGB to Gray Scale conversion",
      ext_modules= cythonize('cython_file.pyx') )