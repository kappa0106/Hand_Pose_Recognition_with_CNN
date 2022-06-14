# 410721242 資工四 何昀軒
import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self,  file, transform=None):
        self.datas = pd.read_csv(file)
        self.transform = transform

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        path = self.datas.iloc[index, 0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        lbl = torch.tensor(int(self.datas.iloc[index, 1]))

        if self.transform:
            img = self.transform(img)

        return img, lbl


def enumerate_files(dirs, path='All_gray_1_32_32/', n_poses=3, n_samples=20):
    filenames, targets = [], []
    for p in dirs:
        for n in range(n_poses):
            for j in range(3):
                dir_name = path+p+'/000'+str(n*3+j)+'/'
                for s in range(n_samples):
                    d = dir_name+'%04d/'%s 
                    for f in os.listdir(d):
                        if f.endswith('jpg'):
                            filenames += [d+f] 
                            targets.append(n)
    return filenames, targets

def read_datasets(datasets, file):
    files, labels = enumerate_files(datasets)
    data = {"filename": files,
                 "label": labels}
    data = pd.DataFrame(data)
    data.to_csv(file,index=False)


if __name__ == "__main__":
    train_sets = ['Set1', 'Set2', 'Set3']
    test_sets = ['Set4', 'Set5']
    read_datasets(train_sets, file="train.csv")
    read_datasets(test_sets, file="test.csv")
