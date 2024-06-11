#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from pandas import read_csv
pd.options.mode.copy_on_write = True


# In[2]:


source_root=r"../autodl-tmp/APTOS-2021"
img_stage1_train_path=os.path.join(source_root,"img_stage1_train")
img_stage2_train_path=os.path.join(source_root,"img_stage2_train","train")
train_anno_stage1_path=r"train_anno_stage1.csv"
train_anno_pic_stage2_path=r"train_anno_pic_stage2.csv"
train_anno_case_stage2_path=r"train_anno_case_stage2.csv"

train_pic_stage1_path='train_pic_stage1.csv'
train_pic_stage2_path='train_pic_stage2.csv'
train_case_stage1_path='train_case_stage1.csv'
train_case_stage2_path='train_case_stage2.csv'

pro_pic_path='pro_pic.csv'
pro_case_path='pro_case.csv'

pic_order=['patient ID','image name','injection','VA','CST','IRF','SRF','PED','HRF','path']


# In[3]:


train_pic_stage1=read_csv(train_pic_stage1_path)
train_pic_stage2=read_csv(train_pic_stage2_path)


# In[4]:


pro_pic=pd.concat([train_pic_stage1,train_pic_stage2],axis=0)


# In[5]:


pro_pic=pro_pic[pro_pic['VA'].notna()&pro_pic['CST'].notna()&
    pro_pic['IRF'].notna()&pro_pic['SRF'].notna()&pro_pic['PED'].notna()&pro_pic['HRF'].notna()]


# In[6]:


pro_pic.to_csv(pro_pic_path,index=False)


# In[7]:


train_case_stage1=read_csv(train_case_stage1_path)
train_case_stage2=read_csv(train_case_stage2_path)


# In[8]:


pro_case=pd.concat([train_case_stage1,train_case_stage2],axis=0)


# In[9]:


pro_case=pro_case[pro_case['preVA'].notna()&pro_case['VA'].notna()&pro_case['preCST'].notna()&pro_case['CST'].notna()
    &pro_case['preIRF'].notna()&pro_case['IRF'].notna()&pro_case['preSRF'].notna()&pro_case['SRF'].notna()
    &pro_case['prePED'].notna()&pro_case['PED'].notna()&pro_case['preHRF'].notna()&pro_case['HRF'].notna()]


# In[10]:


pro_case['gender']=pro_case['gender'].replace({'Male':1,'Female':2})
pro_case['diagnosis']=pro_case['diagnosis'].replace({'DME':4,'PCV':2,'CNVM':1})
pro_case['anti-VEGF']=pro_case['anti-VEGF'].replace({'Avastin':1,'Accentrix':2,'Razumab':2,'Eylea':3,'Tricort':5,'Ozurdex':9,'Ozrudex':9,'Pagenax':9})


# In[11]:


pro_case.to_csv(pro_case_path,index=False)


# In[ ]:




