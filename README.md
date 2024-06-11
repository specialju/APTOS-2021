# APTOS-2021

数据处理:
data_arrange_pic.py用于整理图片数据，将图片的标签和病例、图片地址建立连接。
train_pic_stage1.csv,train_pic_stage2.csv为处理结果

data_arrange_case.py用于整理病例数据，主要将复赛中每个病例的图片合并到病例信息中。
train_case_stage1.csv,train_case_stage2.csv为处理结果

data_fact_pic.py对图片进行预处理，删除有缺失项的数据，结果保存在pro_pic.csv和pro_case中

训练：
train_all.py是训练文件，以8:2划分了训练集和测试集，测试的指标为分类任务(IRF,SRF,PED,HRF)的AUROC均值，回归任务(CST)的R2_SCORE,
以及分类任务四个子项的准确率、召回率

训练结果:
