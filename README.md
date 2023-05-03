# Programming Assignment 1: Hand Pose Recognition with CNN

![GitLab (self-managed)](https://img.shields.io/github/license/kappa0106/Hand_Pose_Recognition_with_CNN?style=for-the-badge)
![GitHub top language](https://img.shields.io/github/languages/top/kappa0106/Hand_Pose_Recognition_with_CNN?style=for-the-badge)
    
    用CNN辨識三種手勢
    
    the data set from the following link:
    http://web.csie.ndhu.edu.tw/ccchiang/Data/All_gray_1_32_32.rar
___

## Ⅰ. Method descriptions

1. 首先建立一個簡單的基本模型 **Model 1**，包含兩層 convolution layer、pooling layer 和一個 activation function，最後連接一個 fully connected layer。
2. 利用Model 1嘗試不同的 **Epoch**, **Optimizer**，分別有 20、25、30、50 及 SGD、RMSprop、Adagrad、Adam。
3. 再以Model 1為基礎，建立不同的模型 **Model 2**, **Model 3**，並分別使用不同 **Learning Rate** (0.02, 0.002, 0.0002) 進行訓練。<br>
    **Model 2** 中增加了一層 convolution layer 以及在 activation function 後及 fully connected layer 前加入 Dropout 。<br>
    **Model 3** 則是在 activation function 前加入 batch normalization layer。<br>
    
5. 依照上面的結果選定的模型與基本參數，再進行 Hyper Parameters 的調整，調整的部分有 convolution layer 的 out_channels、kernel_size 以及 pooling layer 的 kernel_size。
6. 最後綜合上述實驗結果挑選適合的參數再次訓練模型得出結果。
___
#### Model 1

利用Conv2d、MaxPool2d以及Linear建立模型，Activation Function使用ReLU()完成建構卷積神經網路

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             832
         MaxPool2d-2             [-1, 32, 8, 8]               0
            Conv2d-3             [-1, 64, 8, 8]          51,264
              ReLU-4             [-1, 64, 8, 8]               0
         MaxPool2d-5             [-1, 64, 2, 2]               0
            Linear-6                    [-1, 3]             771
================================================================
Total params: 52,867
Trainable params: 52,867
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.33
Params size (MB): 0.20
Estimated Total Size (MB): 0.54
----------------------------------------------------------------
```

___
#### Model 2

以Model 1為基礎，增加一些Conv2d及Dropout
```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param # 
================================================================
            Conv2d-1           [-1, 32, 32, 32]             832 
         MaxPool2d-2             [-1, 32, 8, 8]               0 
            Conv2d-3             [-1, 64, 8, 8]          51,264 
              ReLU-4             [-1, 64, 8, 8]               0 
           Dropout-5             [-1, 64, 8, 8]               0 
            Conv2d-6            [-1, 128, 8, 8]         204,928 
           Dropout-7            [-1, 128, 8, 8]               0
            Linear-8                    [-1, 3]          24,579
================================================================
Total params: 281,603
Trainable params: 281,603
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.48
Params size (MB): 1.07
Estimated Total Size (MB): 1.56
----------------------------------------------------------------
```
___
#### Model 3
以Model 1為基礎，增加Batch Normalization Layer
```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param # 
================================================================
            Conv2d-1           [-1, 32, 32, 32]             832 
       BatchNorm2d-2           [-1, 32, 32, 32]              64 
              ReLU-3           [-1, 32, 32, 32]               0 
         MaxPool2d-4             [-1, 32, 8, 8]               0 
            Conv2d-5             [-1, 64, 8, 8]          51,264 
       BatchNorm2d-6             [-1, 64, 8, 8]             128 
              ReLU-7             [-1, 64, 8, 8]               0 
         MaxPool2d-8             [-1, 64, 2, 2]               0 
            Linear-9                    [-1, 3]             771 
================================================================
Total params: 53,059
Trainable params: 53,059
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.86
Params size (MB): 0.20
Estimated Total Size (MB): 1.07
----------------------------------------------------------------
```


___
## Ⅱ. Source code explanations

### Dataset split
將資料讀入並分隔成 train sets 以及 test sets，再建立 PyTorch Dataset。<br>
主要架構是利用教授作業說明中附上的程式碼進行修改，將圖片的路徑及 label 分別存入 train.csv 及 test.csv，建立自己的 Dataset 並在當中進行讀取圖片的動作。

```python
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

```
___
### DataLoader

import 需要的 module 並設定訓練需要的參數。<br>
利用自己建的 Dataset 將 data 讀入，並將 train data 分出 30% 做為 validation data。<br>
再分別用 DataLoader 以 batch 的方式載入資料進行訓練。<br>

```python
# import module
import torch
import torch.nn as nn
from Dataset import myDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Parameters
num_classes = 3
num_epochs = 30
BATCH_SIZE = 64

# Load train and test data
trn_data = myDataset(file="train.csv", transform=transforms.ToTensor())
# split 70% data for train, 30% for validation
train_size = int(0.7 * len(trn_data))
valid_size = len(trn_data) - train_size
trn_data, valid_data = torch.utils.data.random_split(trn_data, [train_size, valid_size])
tst_data = myDataset(file="test.csv", transform=transforms.ToTensor())

trn_loader = DataLoader(dataset=trn_data, batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE,shuffle=True)
tst_loader = DataLoader(dataset=tst_data, batch_size=BATCH_SIZE)
```
___
### Building a Convolutional Neural Network
建立 CNN Model，並設定 Optimizer、Loss function。
```python
# Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        
        self.cnn1 = nn.Conv2d(                                      # Convolution 1 , input_shape = (1, 32, 32)
                        in_channels=1,
                        out_channels=32,
                        kernel_size=5,
                        stride=1,
                        padding=2)                                  # output_shape = (32, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)                               # Batch normalization 1
        self.relu1 = nn.ReLU()                                      # activation  
        self.maxpool1 = nn.MaxPool2d(kernel_size=4)                 # Max pool 1 , output_shape = (32, 8, 8)
        self.cnn2 = nn.Conv2d(                                      # Convolution 2
                        in_channels=32,
                        out_channels=64,
                        kernel_size=5,
                        stride=1,
                        padding=2)                                  # output_shape = (64, 8, 8)
        self.bn2 = nn.BatchNorm2d(64)                               # Batch normalization 2
        self.relu2 = nn.ReLU()                                      # activation
        self.maxpool2 = nn.MaxPool2d(kernel_size=4)                 # Max pool 2 , output_shape = (64, 2, 2)  
        self.fc1 = nn.Linear(64*2*2, num_classes)                   # Fully connected 1 , input_shape = (64 * 2 * 2)
    
    def forward(self, x):
        
        x = self.cnn1(x)
        x = self.relu1(self.bn1(x))
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu2(self.bn2(x))                                 
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)                                   # flatten the output
        output = self.fc1(x)                                        # Linear function (readout)
        return output, x                                            # return x for visualization
```
```python
model = CNN_Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)         # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                                   # the target label is not one-hotted
```
___
### Train the model
設定訓練模型需要的步驟，forward -> backward -> gradient -> evaluate，<br>
並在每個 Epoch 輸出 Train loss、Train accuracy、Validation loss、Validation accuracy 方便觀察訓練過程。
```python
def fit_model(model, loss_func, optimizer, num_epochs, train_loader, test_loader):
    # Traning the Model
    training_loss = []                                              # list for store loss & acc value
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        # training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            
            # Forward propagation
            outputs = model(images)[0]                              # CNN model output
            train_loss = loss_func(outputs, labels)                 # calculate cross entropy loss

            # Backward
            optimizer.zero_grad()                                   # clear gradients
            train_loss.backward()                                   # backpropagation, calculate gradients

            # Gradient Step
            optimizer.step()                                        # apply gradients and update parameters

            predicted = torch.max(outputs.data, 1)[1]               # get predictions from the maximum value
            
            total_train += len(labels)                              # total number of train data labels
            correct_train += (predicted == labels).float().sum()    # total correct predictions of train data
        
        train_accuracy = 100 * correct_train / float(total_train)   # calculate & store train_acc / epoch
        training_accuracy.append(train_accuracy)
        training_loss.append(train_loss.data)                       # store loss / epoch
        # evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        for images, labels in test_loader:

            # Forward propagation
            outputs = model(images)[0] 
            val_loss = loss_func(outputs, labels)                   # Calculate softmax and cross entropy loss
            predicted = torch.max(outputs.data, 1)[1]               # get predictions from the maximum value

            
            total_test += len(labels)                               # total number of test data labels
            correct_test += (predicted == labels).float().sum()     # total correct predictions of test data

        val_accuracy = 100 * correct_test / float(total_test)       # calculate & store val_acc / epoch
        validation_accuracy.append(val_accuracy)
        validation_loss.append(val_loss.data)                       # store val_loss / epoch

        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    
    return training_loss, training_accuracy, validation_loss, validation_accuracy

```

```python
training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, num_epochs, trn_loader, valid_loader)
```

___
### Test the model
在訓練結束後利用 Test Dataset 進行測試，並輸出 Test accuracy。

```python
# test the model
correct_test = 0
total_test = 0
model.eval()
with torch.no_grad():
    for img, lbl in tst_loader:
        predict = torch.max(model(img)[0], 1)[1]
        total_test += predict.size()[0]
        correct_test += (predict == lbl).float().sum()
        test_accuracy = 100 * correct_test / float(total_test)
    print('Test_accuracy: {:.6f}%'.format(test_accuracy))
```
___
### Visulizartion

將訓練過程圖像化，輸出 Train 和 Validation 的 loss、accuracy。

```python
# visualization train & validation loss
plt.plot(range(num_epochs), training_loss, label='Training_loss')
plt.plot(range(num_epochs), validation_loss, label='validation_loss')
plt.title('Training & Validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

```python
# visualization train & validation accuracy
plt.plot(range(num_epochs), training_accuracy, label='Training_accuracy')
plt.plot(range(num_epochs), validation_accuracy, label='Validation_accuracy')
plt.title('Training & Validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
___
## Ⅲ. Experimental results



### 訓練結果

#### 最佳結果及參數
```
Epoch:  30
batch size: 64
optimizer: Adam
loss function: CrossEntropyLoss
out channels: 32, kernel size: 5
Pooling Size: 4
Training Accuracy:  100.0 %
Validation Accuracy: 98.15%
Test Accuracy: 97.50%
```

#### Loss
![](https://i.imgur.com/zUFr6lm.png)




#### Accuracy
![](https://i.imgur.com/q2Skfqa.png)



#### 訓練過程
```!
Train Epoch: 1/30 Traing_Loss: 0.8525720238685608 Traing_acc: 48.677250% Val_Loss: 0.9149513840675354 Val_accuracy: 56.790123%
Train Epoch: 2/30 Traing_Loss: 0.6855289936065674 Traing_acc: 72.751320% Val_Loss: 0.701896071434021 Val_accuracy: 79.629631%
Train Epoch: 3/30 Traing_Loss: 0.5556931495666504 Traing_acc: 85.449738% Val_Loss: 0.5618810653686523 Val_accuracy: 85.802467%
Train Epoch: 4/30 Traing_Loss: 0.49033257365226746 Traing_acc: 89.682541% Val_Loss: 0.5506885051727295 Val_accuracy: 88.271606%
Train Epoch: 5/30 Traing_Loss: 0.41329658031463623 Traing_acc: 91.269844% Val_Loss: 0.4608863294124603 Val_accuracy: 87.654320%
Train Epoch: 6/30 Traing_Loss: 0.3491668403148651 Traing_acc: 94.708992% Val_Loss: 0.3903864026069641 Val_accuracy: 91.358025%
Train Epoch: 7/30 Traing_Loss: 0.3104687035083771 Traing_acc: 96.031746% Val_Loss: 0.32426974177360535 Val_accuracy: 92.592590%
Train Epoch: 8/30 Traing_Loss: 0.293700635433197 Traing_acc: 96.296295% Val_Loss: 0.3107680082321167 Val_accuracy: 93.827164%
Train Epoch: 9/30 Traing_Loss: 0.24981679022312164 Traing_acc: 97.089951% Val_Loss: 0.3222508132457733 Val_accuracy: 95.061729%
Train Epoch: 10/30 Traing_Loss: 0.2231176197528839 Traing_acc: 96.825394% Val_Loss: 0.3172222077846527 Val_accuracy: 93.209877%
Train Epoch: 11/30 Traing_Loss: 0.2121712863445282 Traing_acc: 97.354500% Val_Loss: 0.2603428363800049 Val_accuracy: 95.061729%
Train Epoch: 12/30 Traing_Loss: 0.25488272309303284 Traing_acc: 98.148148% Val_Loss: 0.22303934395313263 Val_accuracy: 96.913582%
Train Epoch: 13/30 Traing_Loss: 0.15888121724128723 Traing_acc: 98.148148% Val_Loss: 0.14633703231811523 Val_accuracy: 93.827164%
Train Epoch: 14/30 Traing_Loss: 0.17644819617271423 Traing_acc: 98.412697% Val_Loss: 0.17273028194904327 Val_accuracy: 96.913582%
Train Epoch: 15/30 Traing_Loss: 0.15930718183517456 Traing_acc: 99.206352% Val_Loss: 0.19012679159641266 Val_accuracy: 96.296295%
Train Epoch: 16/30 Traing_Loss: 0.11723208427429199 Traing_acc: 99.206352% Val_Loss: 0.19922706484794617 Val_accuracy: 96.913582%
Train Epoch: 17/30 Traing_Loss: 0.10865329951047897 Traing_acc: 99.206352% Val_Loss: 0.24739408493041992 Val_accuracy: 96.913582%
Train Epoch: 18/30 Traing_Loss: 0.12337872385978699 Traing_acc: 99.735451% Val_Loss: 0.14290352165699005 Val_accuracy: 96.913582%
Train Epoch: 19/30 Traing_Loss: 0.08938257396221161 Traing_acc: 100.000000% Val_Loss: 0.11644510924816132 Val_accuracy: 97.530861%
Train Epoch: 20/30 Traing_Loss: 0.09202084690332413 Traing_acc: 99.470901% Val_Loss: 0.11838647723197937 Val_accuracy: 97.530861%
Train Epoch: 21/30 Traing_Loss: 0.09029420465230942 Traing_acc: 100.000000% Val_Loss: 0.1676664501428604 Val_accuracy: 98.148148%
Train Epoch: 22/30 Traing_Loss: 0.08726762235164642 Traing_acc: 100.000000% Val_Loss: 0.1299387663602829 Val_accuracy: 98.148148%
Train Epoch: 23/30 Traing_Loss: 0.06575678288936615 Traing_acc: 100.000000% Val_Loss: 0.15636061131954193 Val_accuracy: 98.148148%
Train Epoch: 24/30 Traing_Loss: 0.07987303286790848 Traing_acc: 100.000000% Val_Loss: 0.0764530822634697 Val_accuracy: 97.530861%
Train Epoch: 25/30 Traing_Loss: 0.06634972989559174 Traing_acc: 100.000000% Val_Loss: 0.09978482127189636 Val_accuracy: 98.148148%
Train Epoch: 26/30 Traing_Loss: 0.059077825397253036 Traing_acc: 100.000000% Val_Loss: 0.1123526394367218 Val_accuracy: 97.530861%
Train Epoch: 27/30 Traing_Loss: 0.06327088177204132 Traing_acc: 100.000000% Val_Loss: 0.1267332136631012 Val_accuracy: 98.765434%
Train Epoch: 28/30 Traing_Loss: 0.044915858656167984 Traing_acc: 100.000000% Val_Loss: 0.06862301379442215 Val_accuracy: 98.765434%
Train Epoch: 29/30 Traing_Loss: 0.04328121989965439 Traing_acc: 100.000000% Val_Loss: 0.09603068232536316 Val_accuracy: 98.148148%
Train Epoch: 30/30 Traing_Loss: 0.04585913568735123 Traing_acc: 100.000000% Val_Loss: 0.11292654275894165 Val_accuracy: 98.148148%


Test_accuracy: 97.500000%
```



---
### 調整 Epoch, Optimizer

use Model 1

#### Epoch 20

| Optimizer | Train Accuracy | Test Accuracy |
|:---------:|:--------------:|:-------------:|
|    SGD    |     38.10%     |    34.17%     |
|  RMSprop  |     86.24%     |    60.28%     |
|  Adagrad  |     84.92%     |    76.94%     |
|   Adam    |     98.67%     |    86.11%     |

#### Epoch 25

| Optimizer | Train Accuracy | Test Accuracy |
|:---------:|:--------------:|:-------------:|
|    SGD    |     37.03%     |    35.28%     |
|  RMSprop  |     86.68%     |    75.28%     |
|  Adagrad  |     88.89%     |    81.67%     |
|   Adam    |     98.41%     |    87.50%     |
        

#### Epoch 30

| Optimizer | Train Accuracy | Test Accuracy |
|:---------:|:--------------:|:-------------:|
|    SGD    |     57.40%     |    46.94%     |
|  RMSprop  |     93.82%     |    78.05%     |
|  Adagrad  |     89.15%     |    75.55%     |
| **Adam**  |  **100.00%**   |  **90.83%**   |

        
#### Epoch 50

| Optimizer | Train Accuracy | Test Accuracy |
|:---------:|:--------------:|:-------------:|
|    SGD    |     56.79%     |    44.16%     |
|  RMSprop  |     97.35%     |    77.77%     |
|  Adagrad  |     89.68%     |    84.72%     |
|   Adam    |    100.00%     |    71.44%     |

---
### 使用不同模型架構並調整 learning_rate

Optimizer: Adam<br>
Epoch: 50

#### Learning Rate 0.02
|  Model  | Train Accuracy | Test Accuracy |
|:-------:|:--------------:|:-------------:|
| Model 1 |     96.29%     |    75.83%     |
| Model 3 |     35.74%     |    33.33%     |
| Model 3 |     97.53%     |    91.11%     |


#### Learning Rate 0.002
|  Model  | Train Accuracy | Test Accuracy |
|:-------:|:--------------:|:-------------:|
| Model 1 |     89.94%     |    78.05%     |
| Model 3 |     92.32%     |    83.33%     |
| Model 3 |     98.76%     |    94.44%     |


#### Learning Rate 0.0002
|    Model    | Train Accuracy | Test Accuracy |
|:-----------:|:--------------:|:-------------:|
|   Model 1   |     96.29%     |    90.39%     |
|   Model 2   |     98.15%     |    92.78%     |
| **Model 3** |  **100.00%**   |  **95.27%**   |




___
### 調整模型內的參數

use Model 3<br>
Optimizer: Adam<br>
Epoch: 50<br>
Learning rate: 0.002<br>

#### convolution layer 的 out_channels
| out_channels | Train Accuracy | Test Accuracy |
|:------------:|:--------------:|:-------------:|
|      8       |     74.44%     |    63.06%     |
|      16      |     86.67%     |    77.78%     |
|    **32**    |   **90.56%**   |  **85.83%**   |
|      64      |     96.11%     |    91.39%     |
|     128      |    100.00%     |    93.06%     |
|     256      |     98.15%     |    91.11%     |


#### convolution layer 的 kernel_size
| kernel_size | Train Accuracy | Test Accuracy |
|:-----------:|:--------------:|:-------------:|
|      1      |     86.29%     |    53.89%     |
|      3      |     97.78%     |    81.94%     |
|    **5**    |   **99.44%**   |  **92.78%**   |

#### pooling layer 的 kernel_size
| kernel_size | Train Accuracy | Test Accuracy |
|:-----------:|:--------------:|:-------------:|
|      2      |     99.43%     |    91.94%     |
|      3      |     98.52%     |    91.39%     |
|    **4**    |   **98.33%**   |  **94.72%**   |


___
## Ⅳ. Discussions on the results


### Epoch & Optimizer
1. 相同 Epoch 下，Adam 的訓練效果較佳。
2. Epoch 越大訓練效果越好，但 50 的結果卻差於 30，可能發生 Overfitting。<br>
   因模型是使用結構較簡單的 Model 1，推測 Overfitting 的原因可能是資料集不夠大量。

### Model & Learning rate
1. 當其餘參數皆無更動時，Learning rate 越小訓練效果越好。<br>
   但沒有嘗試更小的學習率，無法確定是否當學習率再更小時，訓練效果是否會更好。
2. 基於前一次的試驗結果，嘗試增加 Dropout 改善 Epoch 50 時的 Overfitting。<br>
   Learning rate 太大時，Dropout 沒效果，反而照成學習不佳。<br>
   Learning rate 調小時，Dropout 有效果，但進步幅度不大。
3. 改用正規化 (在 activation function 前加入 batch normalization layer) 來改善 Overfitting ，訓練效果進步明顯，在 Learning rate = 0.02 時，Test accuracy 也有 91.11%
   

### Hyper parameters
1. 雖然增加 convolution layer 的 out_channels 的訓練效果會變好，但相對的需要更多的訓練時間及硬體，所以選擇適合自己電腦硬體且效果不差的channel數很重要。
2. 經過試驗 convolution layer 的 kernel_size 和 pooling layer 的 kernel_size 分別在 5 和 4 時表現最佳

___
## Ⅴ. Concluding remarks

之前接觸機器學習時是使用 Tensorflow，這學期是我第一次嘗試 Pytorch，語法以及一些細節不太一樣，剛開始寫時一直有一些 error 讓我很挫折，還好 Stack Overflow 上有好多可以參考的資料，而為了建構更好的 Model 我也更認真的去了解 CNN 中各個 Layer 的詳細用途及效果，最後查了好多資料終於順利完成這次作業。

