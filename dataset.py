import os 
import pandas as pd
from PIL import Image
import numpy as np

def take_images(directory ='.'):
    paths,images = [],[]
    for root,_,files in os.walk(directory):
        for file in files:
            completed_path = os.path.join(root,file)
            paths.append(completed_path)
            img = Image.open(completed_path)
            img = img.resize((128,128))
            images.append(np.array(img))
    return paths,images

def take_tags(paths):
    tags = []
    for i in paths:
        aux = i.split('\\')
        tags.append(aux[3])
    return tags

def save_dataset(train:list,tags:list):
    f = pd.DataFrame({'image':train,'tags':tags})
    f.to_csv('dataset.csv',index=False)

def save_test(elements:list,name:str):
    with open(name,'w') as f:
        f.writelines(elements)

path_train,images = take_images('.\\mg-animal-prediction-24-25\\train_images')
path_test,_ = take_images('.\\mg-animal-prediction-24-25\\test_images')
tags = take_tags(path_train)

save_dataset(images,tags)
save_test(path_test,"test.txt")
