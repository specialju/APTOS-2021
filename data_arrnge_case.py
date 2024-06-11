import os
import pandas as pd
from pandas import read_csv
pd.options.mode.copy_on_write = True

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

case_order=['patient ID','gender','age','diagnosis','anti-VEGF','continue injection',
            'preVA','VA','preCST','CST','preIRF','IRF','preSRF','SRF','prePED','PED','preHRF','HRF']

train_anno_stage1=read_csv(train_anno_stage1_path)
train_case_stage1=train_anno_stage1[case_order]
train_case_stage1.to_csv(train_case_stage1_path,index=False)

train_pic_stage2=read_csv(train_pic_stage2_path)
before_pic=train_pic_stage2[train_pic_stage2['injection']=='Pre injection']
after_pic=train_pic_stage2[train_pic_stage2['injection']=='Post injection']
after_pic.rename(columns={'IRF':'preIRF','SRF':'preSRF','PED':'prePED','HRF':'preHRF'},inplace=True)
before_grouped=before_pic[['patient ID','IRF','SRF','PED','HRF']].groupby('patient ID').max()
after_grouped=after_pic[['patient ID','preIRF','preSRF','prePED','preHRF']].groupby('patient ID').max()

grouped=pd.merge(before_grouped,after_grouped,how='outer',on='patient ID')

train_anno_case_stage2=read_csv(train_anno_case_stage2_path)
train_case_stage2=pd.merge(train_anno_case_stage2,grouped,how='inner',on='patient ID')
train_case_stage2=train_case_stage2[case_order]
train_case_stage2

train_case_stage2.to_csv(train_case_stage2_path,index=False)