import glob
from random import sample
import os

train_ratio = 0.7

train_list = []
valid_list = []

data_dic = {'crazing': [], 'inclusion': [], 'patches': [], 'pitted_surface': [], 'rolled-in_scale': [], 'scratches': []}

for file in glob.glob('./IMAGES/*.jpg'):
    file_name = file.split('/')[-1].split('.')[0]
    for k in data_dic:
        if k in file_name:
            data_dic[k].append(file)
            break

# sample
index_list = list(range(300))

for k in data_dic:
    train_index = sample(index_list, int(300 * train_ratio))
    for i in range(300):
        if i in train_index:
            train_list.append(data_dic[k][i])
        else:
            valid_list.append(data_dic[k][i])

label_files = glob.glob('./LABELS/*.txt')

for i, img in enumerate(train_list):
    os.system('cp ' + img + ' ./data/train/images/')
    txt = './LABELS/' + img.split('/')[-1].replace('jpg', 'txt')
    os.system('cp ' + txt + ' ./data/train/labels/')

for i, img in enumerate(valid_list):
    os.system('cp ' + img + ' ./data/valid/images/')
    txt = './LABELS/' + img.split('/')[-1].replace('jpg', 'txt')
    os.system('cp ' + txt + ' ./data/valid/labels/')

print('Success!')
