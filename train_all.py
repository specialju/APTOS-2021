#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
from pandas import read_csv
import torchvision
from torcheval.metrics import BinaryAUROC
from torcheval.metrics.functional import r2_score
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# In[17]:


source_root=r"E:\dataset\APTOS-2021"
img_stage1_train_path=os.path.join(source_root,"img_stage1_train")
img_stage2_train_path=os.path.join(source_root,"img_stage2_train","train")
train_anno_stage1_path=r"E:\dataset\APTOS-2021\train_anno_stage1.csv"
train_anno_pic_stage2_path=r"train_anno_pic_stage2.csv"
train_anno_case_stage2_path=r"train_anno_case_stage2.csv"

train_pic_stage1_path='train_pic_stage1.csv'
train_pic_stage2_path='train_pic_stage2.csv'
train_case_stage1_path='train_case_stage1.csv'
train_case_stage2_path='train_case_stage2.csv'

pro_pic_path='pro_pic.csv'
pro_case_path='pro_case.csv'

model_root_path='model/all'
model_log_path='model/all/log'
model_load_path='model/all/ImageModule/img_0.0001_98.pth'
image_model_name='ImageModel'

EPOCHS=60

pic_cla_label=['IRF','SRF','PED','HRF']

pic_order=['patient ID','image name','injection','VA','CST','IRF','SRF','PED','HRF','path']
case_order=['patient ID','gender','age','diagnosis','anti-VEGF','continue injection',
            'preVA','VA','preCST','CST','preIRF','IRF','preSRF','SRF','prePED','PED','preHRF','HRF']

case_label_input=['patient ID','gender','age','diagnosis','anti-VEGF',
            'preVA','VA','preCST','CST','preIRF','IRF','preSRF','SRF','prePED','PED','preHRF','HRF']
case_label_output=['continue injection']


# In[4]:


PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True
pd.options.mode.copy_on_write = True
pil_to_tensor=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mse_loss_fn = nn.MSELoss()
bce_log_loss_fn = nn.BCEWithLogitsLoss()


# In[5]:


class BasicConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(stride, stride),
                      padding=padding,
                      dilation=(dilation, dilation)),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ImageModule(nn.Module):
    def __init__(self, input_shape=(3, 400, 650), out_features=64, classify_sum: int = 4):
        super(ImageModule, self).__init__()
        self.ms_cam = MS_CAM(channels=64) # MS_CAM 模块
        self.layers = nn.Sequential(
            BasicConv2d(input_shape[0], 8),
            BasicConv2d(8, 8, stride=2),
            BasicConv2d(8, 8, stride=2),
            BasicConv2d(8, 16, stride=2),
            BasicConv2d(16, 16, stride=2),
            BasicConv2d(16, 32, stride=2),
            BasicConv2d(32, 32, stride=2),
            BasicConv2d(32, 64, stride=2),
            BasicConv2d(64, out_features, stride=2)
        )
        conv2d_count = 8
        H = int(input_shape[1] // (2 ** conv2d_count)) + 1
        W = int(input_shape[2] // (2 ** conv2d_count)) + 1
        in_features = out_features * H * W
        
        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, int(in_features / 2)),
            nn.ReLU(),
            nn.Linear(int(in_features / 2), int(in_features / 4)),
            nn.ReLU(),
            nn.Linear(int(in_features / 4), 1)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, int(in_features / 2)),
            nn.ReLU(),
            nn.Linear(int(in_features / 2), int(in_features / 4)),
            nn.ReLU(),
            nn.Linear(int(in_features / 4), classify_sum),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x) # x has 64 channels
        x = self.ms_cam(x)
        y_regression = self.regression(x)
        y_classify = self.classify(x)
        return y_regression, y_classify


class CSVModule(nn.Module):
    def __init__(self, in_features: int, classify_sum: int = 2):
        super(CSVModule, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )

        self.regression = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        self.classify = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, classify_sum),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        y_regression = self.regression(x)
        y_classify = self.classify(x)
        return y_regression, y_classify


class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


# In[6]:


class CaseDataset(Dataset):
    def __init__(self,pd):
        self.pd=pd
    def __getitem__(self,index):
        label_in=self.pd[case_label_input]
        label_out=self.pd[case_label_output]
    def __len__(self):
        return len(self.pd)

class ImageDataset(Dataset):
    def __init__(self,pd):
        self.pd=pd
    def __getitem__(self,index):
        #:param index:
        #:return: training=True: image, (regression_label, classify_label);
        x=self.pd.iloc[index]
        image=Image.open(x['path'])
        image=image.crop((550,50,1200,450))
        regression_label=x[['CST']]/100.
        classify_label=x[['IRF','SRF','PED','HRF']]
        return pil_to_tensor(image).float(),(torch.from_numpy(regression_label.values.astype(float)),torch.from_numpy(classify_label.values.astype(float)))
    def __len__(self):
        return len(self.pd)


# In[7]:


def ImageLoss(inputs,targets):
    y_regression, y_classify = inputs
    regression_labels, classify_labels = targets
    
    mse_loss = mse_loss_fn(y_regression.float(), regression_labels.float())
    bce_log_loss = bce_log_loss_fn(y_classify.float(), classify_labels.float())
    return mse_loss + bce_log_loss * 4
    
def cal_auroc(y_preds, y_true, batch_size):
    metric = BinaryAUROC(num_tasks=batch_size)
    metric.update(y_preds, y_true)
    vals = metric.compute()
    return vals
    
def cal_reg(y_preds, y_true):
    val = r2_score(y_preds, y_true)
    return val

def getBinaryTensor(imgTensor, boundary = 0.5):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero).bool()

def cal_P(x):
    x=getBinaryTensor(x)
    return (x>0.5).float()
    
def cal_correct_P(x,y):
    x=getBinaryTensor(x)
    y=getBinaryTensor(y)
    return (x&y).float()

def plot_log(title, xlabel, ylabel, data1, label1, data2=None, label2=None, log_path=None):
    if not os.path.exists(log_path):
        return
    x = [i for i in range(len(data1))]
    plt.plot(x, data1, c='r', lw=1, label=label1)
    if data2:
        plt.plot(x, data2, c='b', lw=1, label=label2)
        plt.legend(prop={'size': 10})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(os.path.join(log_path, f'{title}.png'))
    plt.close()


# In[18]:


model=torch.load(model_load_path)


# In[19]:


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pro_pic=read_csv(pro_pic_path)
img_data=ImageDataset(pro_pic)
#indices=range(5200)
#img_data=torch.utils.data.Subset(img_data, indices)
train_size=int(0.8*len(img_data))
test_size=len(img_data)-train_size
train_data,test_data=torch.utils.data.random_split(img_data, [train_size, test_size])

train_loader=DataLoader(train_data,batch_size=64,drop_last=True)
test_loader=DataLoader(test_data,batch_size=64,drop_last=True)


# In[21]:


optim = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)


# In[ ]:


model.train()
model = model.to(device)

auroc_epochs = []
acc_epochs = []
loss_epochs=[]

test_auroc_epochs = []
test_acc_epochs = []
test_loss_epochs=[]

recall_epochs=[]
precision_epochs=[]
for epoch in range(EPOCHS):
    loss_batchs = []
    auroc_batchs = []
    acc_batchs = []

    test_loss_batchs = []
    test_auroc_batchs = []
    test_acc_batchs = []

    for images, labels in tqdm(train_loader):
        optim.zero_grad()
        y = model(images.to(device))
        labels[0] = labels[0].to(device)
        labels[1] = labels[1].to(device)
        loss = ImageLoss(y, labels)
            
        reg_acc = cal_reg(y[0].cpu().detach(),labels[0].cpu().detach())
        auroc = cal_auroc(y[1].cpu().detach(), labels[1].cpu().detach(), len(y[0])).mean()
            
        loss.backward() 
        optim.step()
        with torch.no_grad():
            loss_batch = loss.detach().cpu().numpy()
            loss_batchs.append(loss_batch)
            acc_batch = reg_acc.detach().cpu().numpy()
            acc_batchs.append(acc_batch)
            auroc_batch = auroc.detach().cpu().numpy()
            auroc_batchs.append(auroc_batch)

    loss_epochs.append(np.mean(loss_batchs))
    acc_epochs.append(np.mean(acc_batchs))
    auroc_epochs.append(np.mean(auroc_batchs))
    learning_rate=optim.state_dict()['param_groups'][0]['lr']
    print('Epoch: {}/{}'.format(epoch + 1, EPOCHS))
    print('learning_rata:{}'.format(learning_rate))
    print('TrainLoss: {:.3f}, TrainReg ACC: {:.2f}%, TrainCls AUROC: {:.2f}%'.format(np.mean(loss_batchs), np.mean(acc_batchs)*100, np.mean(auroc_batchs)*100))
    
    correct_P=[]
    predicts_P=[]
    targets_P=[]
    print('start test:')
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            y = model(images.to(device))
            labels[0] = labels[0].to(device)
            labels[1] = labels[1].to(device)
            loss = ImageLoss(y, labels)

            reg_acc = cal_reg(y[0].cpu().detach(),labels[0].cpu().detach())
            auroc = cal_auroc(y[1].cpu().detach(), labels[1].cpu().detach(), len(y[0])).mean()

            loss_batch = loss.detach().cpu().numpy()
            test_loss_batchs.append(loss_batch)
            acc_batch = reg_acc.detach().cpu().numpy()
            test_acc_batchs.append(acc_batch)
            auroc_batch = auroc.detach().cpu().numpy()
            test_auroc_batchs.append(auroc_batch)

            correct_P.append(cal_correct_P(y[1].cpu().detach(),labels[1].cpu().detach()))
            predicts_P.append(cal_P(y[1].cpu().detach()))
            targets_P.append(cal_P(labels[1].cpu().detach()))

    correct_P=sum(correct_P)
    correct_P=correct_P.sum(dim=0)
    predicts_P=sum(predicts_P)
    predicts_P=predicts_P.sum(dim=0)
    targets_P=sum(targets_P)
    targets_P=targets_P.sum(dim=0)

    recall=correct_P/targets_P
    recall_epochs.append(recall)
    precision=correct_P/predicts_P
    precision_epochs.append(precision)

    test_loss_epochs.append(np.mean(test_loss_batchs))
    test_acc_epochs.append(np.mean(test_acc_batchs))
    test_auroc_epochs.append(np.mean(test_auroc_batchs))

    scheduler.step()
    print('TestLoss: {:.3f}, TestReg ACC: {:.2f}%, TestCls AUROC: {:.2f}%'.format(np.mean(test_loss_batchs), np.mean(test_acc_batchs)*100, np.mean(test_auroc_batchs)*100))
    print('recall:{},precision:{}'.format(recall,precision))
    torch.save(model,'model/all/ImageModule/img_{}_{}.pth'.format(optim.state_dict()['param_groups'][0]['lr'],epoch))
    print('模型已保存')
    
log_name = f'{image_model_name}-loss-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log(log_name, 'epoch', 'loss', loss_epochs, 'train_loss',test_loss_epochs, 'test_loss',log_path=model_log_path)
log_name = f'{image_model_name}-auroc-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log(log_name, 'epoch', 'auroc',auroc_epochs,'train_auroc',test_auroc_epochs,'test_auroc',log_path=model_log_path)
log_name = f'{image_model_name}-acc-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log(log_name, 'epoch', 'acc',acc_epochs,'train_acc',test_acc_epochs,'test_acc',log_path=model_log_path)

torch.save(precision_epochs,os.path.join(model_root_path,'log/precision.pth'))
torch.save(recall_epochs,os.path.join(model_root_path,'log/recall.pth'))
torch.save(test_loss_epochs,os.path.join(model_root_path,'log/test_loss_epochs.pth'))
torch.save(test_acc_epochs,os.path.join(model_root_path,'log/test_acc_epochs.pth'))
torch.save(test_auroc_epochs,os.path.join(model_root_path,'log/test_auroc_epochs.pth'))
torch.save(loss_epochs,os.path.join(model_root_path,'log/train_loss_epochs.pth'))
torch.save(acc_epochs,os.path.join(model_root_path,'log/train_acc_epochs.pth'))
torch.save(auroc_epochs,os.path.join(model_root_path,'log/train_auroc_epochs.pth'))


# In[25]:


torch.save(precision_epochs,os.path.join(model_root_path,'log/precision.pth'))
torch.save(recall_epochs,os.path.join(model_root_path,'log/recall.pth'))


# In[26]:


torch.save(test_loss_epochs,os.path.join(model_root_path,'log/test_loss_epochs.pth'))
torch.save(test_acc_epochs,os.path.join(model_root_path,'log/test_acc_epochs.pth'))
torch.save(test_auroc_epochs,os.path.join(model_root_path,'log/test_auroc_epochs.pth'))
torch.save(loss_epochs,os.path.join(model_root_path,'log/train_loss_epochs.pth'))
torch.save(acc_epochs,os.path.join(model_root_path,'log/train_acc_epochs.pth'))
torch.save(auroc_epochs,os.path.join(model_root_path,'log/train_auroc_epochs.pth'))


# In[ ]:


log_name = f'{image_model_name}-loss-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log(log_name, 'epoch', 'loss', loss_epochs, 'train_loss',test_loss_epochs, 'test_loss',log_path=model_log_path)
log_name = f'{image_model_name}-auroc-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log(log_name, 'epoch', 'auroc',auroc_epochs,'train_auroc',test_auroc_epochs,'test_auroc',log_path=model_log_path)


# In[114]:


log_name = f'{image_model_name}-acc-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log(log_name, 'epoch', 'acc',acc_epochs,'train_acc',test_acc_epochs,'test_acc',log_path=model_log_path)


# In[28]:


log_name = f'{image_model_name}-acc-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log(log_name, 'epoch', 'acc',acc_epochs,'train_auroc',test_acc_epochs,'test_acc',log_path=model_log_path)


# In[51]:


IRF_precision=[]
SRF_precision=[]
PED_precision=[]
HRF_precision=[]

IRF_recall=[]
SRF_recall=[]
PED_recall=[]
HRF_recall=[]
for x in precision_epochs:
    IRF_precision.append(x[0].item())
    SRF_precision.append(x[1].item())
    PED_precision.append(x[2].item())
    HRF_precision.append(x[3].item())

for x in recall_epochs:
    IRF_recall.append(x[0].item())
    SRF_recall.append(x[1].item())
    PED_recall.append(x[2].item())
    HRF_recall.append(x[3].item())


# In[52]:


recall_list=[IRF_recall,SRF_recall,PED_recall,HRF_recall]
precision_list=[IRF_precision,SRF_precision,PED_precision,HRF_precision]


# In[53]:


log_name = f'{image_model_name}-precisioin-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log_mul(log_name, 'epoch', 'precision',precision_list,['IRF','SRF','PED','HRF'],log_path=model_log_path)
log_name = f'{image_model_name}-recall-epoch({epoch})-eta({learning_rate})-{time.strftime("%Y%m%d%H%M")}'
plot_log_mul(log_name, 'epoch', 'recall',recall_list,['IRF','SRF','PED','HRF'],log_path=model_log_path)


# In[49]:


def plot_log_mul(title, xlabel, ylabel,data_epochs,label_list, log_path=None):
    if not os.path.exists(log_path):
        return
    color=['r','b','g','y']

    x = [i for i in range(len(data_epochs))]
    for epoch in range(len(label_list)):
        
        plt.plot(x, data_list[index], c=color[index], lw=1, label=label_list[index])
        plt.legend(prop={'size': 10})

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(os.path.join(log_path, f'{title}.png'))
    plt.close()


# In[41]:


a=[1,2,3]
b=[4,5,6]
c=[a,b]
for y in c:
    print('{}'.format(y))


# In[57]:


x = [i for i in range(len(recall_epochs))]
plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.plot(x,recall_epochs,label=['1','2','3','4'])


# In[56]:


plt.close()


# In[ ]:




