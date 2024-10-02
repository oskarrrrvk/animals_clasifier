import os 
import pandas as pd


def take_images(directory ='.'):
    paths = []
    for root,_,files in os.walk(directory):
        for file in files:
            completed_path = os.path.join(root,file)
            paths.append(completed_path)
    return paths

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

path_train = take_images('.\\mg-animal-prediction-24-25\\train_images')
path_test= take_images('.\\mg-animal-prediction-24-25\\test_images')
tags = take_tags(path_train)

save_dataset(path_train,tags)
save_test(path_test,"test.txt")
