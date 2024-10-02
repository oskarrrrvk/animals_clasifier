from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

class Animals_clisfier:
    def __init__(self,f_train):
        f = pd.read_csv(f_train)

        x = f.drop('tags',axis=1)
        y = f['tags']

        self._train_X,self._test_X,self._train_Y,self._test_Y= train_test_split(x,y,test_size=0.2,random_state=42)
        print(f"train set's size: {self._train_X.shape[0]}")
        print(f"test set's size: {self._test_X.shape[0]}")
        print(f"train: {type(self._train_X)}")


    def create_model(self):
        input = Input(shape=(128,128,1))
        x = Conv2D(16,(3,3),activation='ReLU')(input)
        x = MaxPooling2D((2,2))(x)
        x = Flatten()(x)
        output = Dense(10,activation='softmax')(x)
        self._model = Model(input,output)

a_cls = Animals_clisfier("dataset.csv")
