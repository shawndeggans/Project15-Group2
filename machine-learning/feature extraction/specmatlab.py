


import numpy as np
import IPython
import IPython.display as ipd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as color
import soundfile
import scipy.signal as sig
import sklearn
import math
import os
from os import listdir
import scipy.signal as sig
from os.path import isfile, join
import random
from sklearn.decomposition import PCA
import sklearn
from sklearn import mixture




X = pd.read_csv('nn_ele_hb_00-24hr_TrainingSet_v2.txt', sep="\t", header=0) 


data=[]

wav_ls=[]

for file_path in listdir('datafiles'): 
    
    
    
    samples,sampling_rate=soundfile.read('datafiles/'+file_path)
    
    num_rows= np.shape(X[X['filename']==file_path])[0]
    ref_table=X[X['filename']==file_path]
    wav_name=ref_table['filename'].unique()[0].strip('.wav')
    
    t_start=ref_table[ref_table.columns[2]][0:num_rows+1]
    t_end=ref_table[ref_table.columns[2]][0:num_rows+1]+ref_table[ref_table.columns[4]][0:num_rows+1]
    
   
    window_size, hop_size = int(0.5*sampling_rate), 256 #need to change

    l=[]
    
    wav_ls.append(wav_name)
    
    for i,j,k in zip(t_start.values.tolist(),t_end.values.tolist(),range(num_rows)):
         
        
        l.append(samples[math.floor(i)*sampling_rate:math.ceil(j)*sampling_rate])
    data.append(l)
           





