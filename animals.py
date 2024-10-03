from matplotlib import pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Animals_clisfier:
    def __init__(self,f_train):
        
        f = pd.read_csv(f_train)

        x = f.drop('encode_tags',axis=1)
        y = f['encode_tags']

        train_X,test_X,train_Y,test_Y = train_test_split(x['image'],y,test_size=0.2,random_state=42)
        
        self._train_Y = to_categorical(np.array([train_Y.iloc[i] for i in range(len(train_Y))]))
        self._test_Y = to_categorical(np.array([test_Y.iloc[i] for i in range(len(test_Y))]))
        
        self._train_X = self._open_img(train_X)
        self._test_X = self._open_img(test_X)
        
        print(f"train length: {len(self._train_X),self._train_Y.shape}")
        print(f"test lenght: {len(self._test_X),self._test_Y.shape}")

    def _open_img(self,x):
        result = []
        for i in range(len(x)):
            img = Image.open(x.iloc[i])
            img = img.resize((128,128))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img).astype(np.float16) / 255.0
            result.append(img)
        return np.array(result)

    def create_model(self):
        input = Input(shape=(128,128,3))
        x = Conv2D(16,(3,3),activation='ReLU')(input)
        x = MaxPooling2D((2,2))(x)
        x = Conv2D(32,(3,3),activation='ReLU')(x)
        x = MaxPooling2D((2,2))(x)
        x = Conv2D(64,(3,3),activation='ReLU')(x)
        x = MaxPooling2D((2,2))(x)
        x = Flatten()(x)
        x = Dense(8,activation='softmax')(x)
        output = Dense(10,activation='softmax')(x)
        self._model = Model(input,output)
    
    def execute_model(self):
        self._model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        self._history = self._model.fit(self._train_X,self._train_Y,validation_split=0.2,batch_size=64,epochs=50,validation_data=(self._test_X,self._test_Y))
    
    def show_plot_losses(self):
        plt.rcParams['figure.figsize'] = [20, 5]
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        ax1.set_title('Losses')
        ax1.set_xlabel('epoch')
        ax1.legend(loc="upper right")
        ax1.grid()
        ax1.plot(self._history.history['loss'], label='Training loss')
        ax1.plot(self._history.history['val_loss'], label='Validation loss')
        ax1.legend()

        ax2.set_title('Accuracy')
        ax2.set_xlabel('epoch')
        ax2.legend(loc="upper right")
        ax2.grid()
        ax2.plot(self._history.history['accuracy'], label='Training accuracy')
        ax2.plot(self._history.history['val_accuracy'], label='Validation accuracy')
        ax2.legend()

        plt.show()

    def save_net(self,name):
        self._model.save(name)

a_cls = Animals_clisfier("dataset.csv")
a_cls.create_model()
a_cls.execute_model()
a_cls.show_plot_losses()
