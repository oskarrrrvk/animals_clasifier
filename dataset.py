import os 
import csv

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

def mix_data(data_1,data_2):
    result =[]
    if len(data_1) == len(data_2):
        length = len(data_1)
        for i in range(length):
            result.append([data_1[i],data_2[i]])
    return result


def save_file(elements:list,name:str):
    with open(name,'w')as f:
        if '.csv' in name:
            writer = csv.writer(f)
            writer.writerows(elements)
        else:
            for element in elements:
                f.write(element+'\n')

path_train = take_images('.\\mg-animal-prediction-24-25\\train_images')
path_test = take_images('.\\mg-animal-prediction-24-25\\test_images')
tags = take_tags(path_train)

mix = mix_data(path_train,tags)

save_file(mix,"train.csv")
save_file(path_test,"test.txt")
