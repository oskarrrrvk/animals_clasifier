from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import cv2
import csv

class Animals_clisfier:
    def __init__(self,f_train,f_test):
        with open(f_train,'r') as f:
            csv_r = csv.reader(f)
            trn_r = [i for i in csv_r if len(i) > 0]
        self._trainX, self._trainY = [cv2.imread(i[0]) for i in trn_r],[i[1] for i in trn_r]
        with open(f_test,'r') as f:
            tst_r = f.readlines()
        self._trainY = [cv2.imread(i[0]) for i in tst_r]
        print(self._trainY[0])

a_cls = Animals_clisfier("train.csv","test.txt")


