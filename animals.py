from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import cv2
import csv

class Animals_clisfier:
    def __init__(self,f_name):
        with open(f_name,'r') as f:
            csv_r = csv.reader(f)
            paths = [i for i in csv_r if len(i) > 0]

a_cls = Animals_clisfier("train.csv")


