import os
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image
import pandas as pd
from pandas import read_csv
pd.options.mode.copy_on_write = True

source_root=r"../autodl-tmp/APTOS-2021"
img_stage1_train_path=os.path.join(source_root,"img_stage1_train")
img_stage2_train_path=os.path.join(source_root,"img_stage2_train","train")
train_anno_stage1_path=r"train_anno_stage1.csv"
train_anno_pic_stage2_path=r"train_anno_pic_stage2.csv"
train_anno_case_stage2_path=r"train_anno_case_stage2.csv"

train_pic_stage1_path='train_pic_stage1.csv'
train_pic_stage2_path='train_pic_stage2.csv'

pic_order=['patient ID','image name','injection','VA','CST','IRF','SRF','PED','HRF','path']

def search_image1(root,lst):
    files=os.listdir(root)
    for file in files:
        path=os.path.join(root,file)
        if os.path.isdir(path):
            search_image1(path,lst)
        if path.endswith('.jpg'):
            image_name=file.replace('.jpg','')
            split=image_name.split('_')
            patient_ID=split[0]
            injection=('Pre injection' if split[1][0]=='1' else 'Post injection')
            lst.append({'patient ID':patient_ID,'image name':image_name,'path':path,'injection':injection})
            
train_anno_stage1=read_csv(train_anno_stage1_path)
lst=[]
search_image1(img_stage1_train_path,lst)
train_pic_stage1=pd.DataFrame(lst)
#train_pic_stage1

train_pic_stage1['IRF']=None
train_pic_stage1['SRF']=None
train_pic_stage1['PED']=None
train_pic_stage1['HRF']=None
train_pic_stage1['VA']=None
train_pic_stage1['CST']=None
for index,x in train_pic_stage1.iterrows():
    y=train_anno_stage1[train_anno_stage1['patient ID']==x['patient ID']]
    if len(y)<1:
        continue
    if x['injection']=='Pre injection':
        train_pic_stage1.loc[index,'IRF']=y.iloc[0]['preIRF'].copy()
        train_pic_stage1.loc[index,'SRF']=y.iloc[0]['preSRF'].copy()
        train_pic_stage1.loc[index,'PED']=y.iloc[0]['prePED'].copy()
        train_pic_stage1.loc[index,'HRF']=y.iloc[0]['preHRF'].copy()
        train_pic_stage1.loc[index,'VA']=y.iloc[0]['preVA'].copy()
        train_pic_stage1.loc[index,'CST']=y.iloc[0]['preCST'].copy()
    else:
        train_pic_stage1.loc[index,'IRF']=y.iloc[0]['IRF'].copy()
        train_pic_stage1.loc[index,'SRF']=y.iloc[0]['SRF'].copy()
        train_pic_stage1.loc[index,'PED']=y.iloc[0]['PED'].copy()
        train_pic_stage1.loc[index,'HRF']=y.iloc[0]['HRF'].copy()
        train_pic_stage1.loc[index,'VA']=y.iloc[0]['VA'].copy()
        train_pic_stage1.loc[index,'CST']=y.iloc[0]['CST'].copy()

train_pic_stage1=train_pic_stage1[pic_order]
train_pic_stage1.sort_values(by='patient ID',axis=0,inplace=True)
#train_pic_stage1.reset_index(drop=True,inplace=True)
train_pic_stage1.to_csv(train_pic_stage1_path,index=False)
#train_anno_stage1.sort_values(by='patient ID',axis=0,inplace=True)
#train_pic_stage1.reset_index(drop=True,inplace=True)
#train_anno_stage1.to_csv(train_anno_stage1_path)

train_anno_pic_stage2=read_csv(train_anno_pic_stage2_path)
train_anno_case_stage2=read_csv(train_anno_case_stage2_path)
train_case_stage2=train_anno_case_stage2
train_pic_stage2=train_anno_pic_stage2

#train_pic_stage2['injection']=train_pic_stage2['injection'].map({'Pre injection':1,'Post injection':2})
train_pic_stage2['path']=None 
train_pic_stage2['VA']=None 
train_pic_stage2['CST']=None 
for index,x in train_pic_stage2.iterrows(): 
    path=os.path.join(img_stage2_train_path,x['patient ID'], 
                      'Pre Injection OCT Images' if x['injection']=='Pre injection' else 'Post Injection OCT Images',
                      x['image name']+'.jpg') 
    df=train_case_stage2[train_case_stage2['patient ID']==x['patient ID']].copy()
    if x['injection']=='Pre injection': 
        df=df[['preVA','preCST']]
        df.rename(columns={'preVA':'VA','preCST':'CST'},inplace=True)
        #df=df[['VA','CST']] 
    else: 
        df=df[['VA','CST']] 
    if os.path.exists(path) and len(df): 
        train_pic_stage2.loc[index,'path']=path 
        train_pic_stage2.loc[index,'VA']=df.iloc[0]['VA']
        train_pic_stage2.loc[index,'CST']=df.iloc[0]['CST']

train_pic_stage2=train_pic_stage2[pic_order]
train_pic_stage2.to_csv(train_pic_stage2_path,index=False)