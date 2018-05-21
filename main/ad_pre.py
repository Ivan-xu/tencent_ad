# -*- coding: utf-8 -*-
# @author:ivan
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import h5py
#####
#from basic_fun.sample import mprint
##20180513
from sample import mem_usage
from sample import mprint
from  sample import mail
from sample import ftp_upload
from sample import sysmode,readmode,params_flag
from datetime import datetime
#gc.set_debug(gc.DEBUG_COLLECTABLE)

import numpy as np

now = datetime.now()
now_begin = datetime.now()
def timespent(msg=''):
    global now
    now_end = datetime.now()
    delta = now_end-now
#    delta2 = now_end - now_begin
    if msg =='':
        mprint ('last code spent-times:%s'%str(delta))
#        print ('the whole program spent-times:%s'%str(delta2))
    else:
        mprint (str(msg) +'\t spent-times:%s'%str(delta))
#        print ('the whole program spent-times:%s'%str(delta2))
    now = datetime.now()
##### mode windows




if sysmode == 'ubuntu':
####    PATH
    path_user_feature='/root/workspace/data/userFeature.csv'
    path_ad_feature ='/root/workspace/data/adFeature.csv'
    path_train_csv='/root/workspace/data/train.csv'
    path_test1_csv ='/root/workspace/data/test1.csv'
    path_userFeaturedata ='/root/workspace/data/userFeature.data'
    path_submit='/root/workspace/data/submission.csv'
    def_path_log_path  ='/root/workspace/data/log/ad_'
    path_newuser_feature ='/root/workspace/data/newuserFeature.csv'
    path_nullsubmit_data='/root/workspace/data/nullsubmission.csv'
    path_data_dtypes = '/root/workspace/data/data_dtypes.txt'
    path_data_hdf5='/root/workspace/data/data_prepared_2.hdf5'
    path_data_csv='/root/workspace/data/data_prepared.csv'
    path_user_feature_dtypes='/root/workspace/data/userFeature_dtypes.txt'

    ## 用户特征读取数量
    ##正负样本聚类   
    path_data_negative_cluster ='/root/workspace/data/data_negative_cluster.csv'
    path_data_postive_cluster ='/root/workspace/data/data_postive_cluster.csv'
    path_train_cluster_class ='/root/workspace/data/train_cluster_class.csv'
    path_train_cluster ='/root/workspace/data/train_cluster.csv'
    stpcnt=int(5000*10000)
    chunk=int(50*10000)
    if readmode =='part':
        path_user_feature ='/root/workspace/data/userFeature_test.csv'
        stpcnt =500000
        chunk=100000
else:
####    PATH
    path_user_feature='E:/MLfile/preliminary_contest_data/data/userFeature.csv'
    path_ad_feature ='E:/MLfile/preliminary_contest_data/data/adFeature.csv'
    path_train_csv='E:/MLfile/preliminary_contest_data/data/train.csv'
    path_test1_csv ='E:/MLfile/preliminary_contest_data/data/test1.csv'
    path_userFeaturedata ='C:/Users/persp/workspace/GitHub/data/ad/userFeature.data'    
    path_submit='E:/MLfile/preliminary_contest_data/data/submission.csv'
    def_path_log_path  ='E:/MLfile/preliminary_contest_data/log/ad_'
    path_newuser_feature ='E:/MLfile/preliminary_contest_data/data/newuserFeature.csv'
    path_nullsubmit_data='E:/MLfile/preliminary_contest_data/data/nullsubmission.csv'
    path_data_dtypes = 'E:/MLfile/preliminary_contest_data/data/data_dtypes.txt'
    path_data_hdf5='E:/MLfile/preliminary_contest_data/data/data_prepared_2.hdf5'
    path_data_csv='E:/MLfile/preliminary_contest_data/data/data_prepared.csv'
    path_user_feature_dtypes='E:/MLfile/preliminary_contest_data/data/userFeature_dtypes.txt'


    ## 用户特征读取数量
    
    ##正负样本聚类
    path_data_negative_cluster ='E:/MLfile/preliminary_contest_data/data/data_negative_cluster.csv'
    path_data_postive_cluster ='E:/MLfile/preliminary_contest_data/data/data_postive_cluster.csv'
    path_train_cluster_class ='E:/MLfile/preliminary_contest_data/data/train_cluster_class.csv'
    path_train_cluster ='E:/MLfile/preliminary_contest_data/data/train_cluster.csv'
    stpcnt=200000
    chunk =100000
## 训练/测试数据跑批
if readmode =='part':
    Chunksize =250000
    readnum = 100000    
else:
    Chunksize =500000
    # readnum = 100000   
##  PATH SELECTION IS END!
mprint('PROGRAM IS STARTTING!')    
## 直接读取 MERGED后的数据 尚未ONEHOT AND COUNTVECTOR
data_pre_flag = False
if os.path.exists(path_data_dtypes) and os.path.exists(path_data_csv):
#    try:       
        with open(path_data_dtypes,"r") as f:
            dtypesread =f.read()
        column_types=eval(dtypesread)
        mprint(column_types,'column_types read')
        #读取
        data=pd.read_csv(path_data_csv,dtype=column_types)
        timespent('data read finished')   
        data_pre_flag = True
#    except :
#        data_pre_flag =False

elif (data_pre_flag == False):
    if os.path.exists(path_user_feature):
        
        raw_user_feature=pd.read_csv(path_user_feature)
        timespent('userFeature read') 
        mprint(hex(id(raw_user_feature)),'raw_user_feature') 
        mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')   

    else:
        #第一次读取全量原始数据写入磁盘
        userFeature_data = []
        headerflag=True
        cnt =0
        cnt_i=0
        with open(path_userFeaturedata, 'r') as f:
            for i, line in enumerate(f):
                if i==stpcnt:
                    break
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)        
                if (i+1) % chunk == 0:
                    cnt=cnt+1
                    print (i+1)
                    if stpcnt-(i+1)<chunk:
                        continue
                    else:
                        raw_user_feature = pd.DataFrame(userFeature_data) 
                       
                        userFeature_data=[]
                        raw_user_feature.to_csv(path_user_feature,index=False, header=headerflag,mode='a')   
                        headerflag =False
                        print('chunk %d done.' %cnt)   

        #剩下的处理
            mprint('Last chunk done!')    
            raw_user_feature = pd.DataFrame(userFeature_data)   
            raw_user_feature.to_csv(path_user_feature, header=False,index=False,mode='a')
            timespent('userFeature read')   
            
        # mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')
        
        # raw_user_feature[raw_user_feature.select_dtypes(['object']).columns] = raw_user_feature.select_dtypes(['object']).apply(lambda x: x.astype('category').cat.add_categories(['-1']))
        # mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')
        
        # raw_user_feature[raw_user_feature.select_dtypes(['float']).columns] = raw_user_feature.select_dtypes(['float']).apply(pd.to_numeric,downcast='float32')
        
        # mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')
        #第一次跑时，从这里读取
        del raw_user_feature
        raw_user_feature=pd.read_csv(path_user_feature)
        timespent('userFeature') 
        timespent('read raw_user_feature finished') 
        mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)')   
        mprint(raw_user_feature.dtypes,'raw_user_feature.dtypes')
    ##正式开始数据转换
    user_feature = pd.DataFrame()
    ##start to opt the memory
    for col in raw_user_feature.columns:
        dtype = raw_user_feature[col].dtypes
        mprint(dtype,'feature %s dtype'%(col))
        if  dtype== np.dtype('int64'):
            try:
                user_feature.loc[:,col] = raw_user_feature[col].apply(pd.to_numeric,downcast='int32')
                mprint('%s feature downcast as int'%(col))
            except:
                user_feature.loc[:,col] = raw_user_feature[col]
        if  dtype== np.dtype('float64'):
            try:
                user_feature.loc[:,col] = raw_user_feature[col].apply(pd.to_numeric,downcast='float')
                mprint('%s feature downcast as float'%(col))
            except:
                user_feature.loc[:,col] = raw_user_feature[col]
        if dtype== np.dtype('object'):
            num_unique_values = len(raw_user_feature[col].unique())
            num_total_values = len(raw_user_feature[col])
#            if num_unique_values / num_total_values < 0.5:
            try:
                user_feature.loc[:,col] = raw_user_feature[col].astype('category').cat.add_categories(['-1']).fillna('-1')
                mprint('%s feature downcast as category'%(col))
            except:
                user_feature.loc[:,col] = raw_user_feature[col]
#            else:
#                user_feature.loc[:,col] = raw_user_feature[col]
        else:
            user_feature.loc[:,col] = raw_user_feature[col]
        ##drop the column
        user_feature.fillna('-1')
        raw_user_feature=raw_user_feature.drop(col,axis=1)
        
        #mprint (mem_usage(raw_user_feature),'mem_usage(raw_user_feature)_after')   
        #mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
    mprint('user_feature casttype is done!')               
    mprint (mem_usage(user_feature),'mem_usage(user_feature)')   
    mprint(user_feature.dtypes,'user_feature.dtypes')

    ##write user_feature_dtypesdtypes
    try:
        dtypes = user_feature.dtypes
        dtypes_col = dtypes.index
        dtypes_type = [i.name for i in dtypes.values]

        column_types = dict(zip(dtypes_col, dtypes_type))

        with open(path_user_feature_dtypes,"w") as f:
                f.write(str(column_types))
        mprint('write user_feature_dtypes done')

    except :
        mprint('write user_feature_dtypes error')
    ##对AD_FEATURE数据类型转换            

    raw_ad_feature=pd.read_csv(path_ad_feature)
    ad_feature = pd.DataFrame()
    mprint (mem_usage(raw_ad_feature),'mem_usage(raw_ad_feature)')   
    mprint(ad_feature.dtypes,'ad_featured.dtypes')

    for col in raw_ad_feature.columns:
        dtype = raw_ad_feature[col].dtypes
        mprint(dtype,'feature %s dtype'%(col))
        if  dtype== np.dtype('int64'):
            try:
                ad_feature.loc[:,col] = raw_ad_feature[col].apply(pd.to_numeric,downcast='int32')
                mprint('%s feature downcast as int'%(col))
                mprint('%s feature casttype failed,so keep old'%(col))

            except:
                ad_feature.loc[:,col] = raw_ad_feature[col]
        if  dtype== np.dtype('float64'):
            try:
                ad_feature.loc[:,col] = raw_ad_feature[col].apply(pd.to_numeric,downcast='float32')
                mprint('%s feature downcast as float'%(col))
            except:
                ad_feature.loc[:,col] = raw_ad_feature[col]
                mprint('%s feature casttype failed,so keep old'%(col))

        if dtype== np.dtype('object'):
            num_unique_values = len(raw_ad_feature[col].unique())
            num_total_values = len(raw_ad_feature[col])
            # if num_unique_values / num_total_values < 0.5:
            try:
                ad_feature.loc[:,col] = raw_ad_feature[col].astype('category').cat.add_categories(['-1']).fillna('-1')
                mprint('%s feature downcast as category'%(col))
            except:
                ad_feature.loc[:,col] = raw_ad_feature[col]
                mprint('%s feature casttype failed,so keep old'%(col))

        else:
            ad_feature.loc[:,col] = raw_ad_feature[col]
            mprint('%s feature casttype failed,so keep old'%(col))

        ##drop the column
        ad_feature.fillna('-1')
        raw_ad_feature=raw_ad_feature.drop(col,axis=1)
        #mprint (mem_usage(raw_ad_feature),'mem_usage(raw_ad_feature)_after')   
        #mprint (mem_usage(ad_feature),'mem_usage(ad_feature)')   
                
    mprint('ad_feature casttype is done!') 
    mprint (mem_usage(ad_feature),'mem_usage(ad_feature)')   
    mprint(ad_feature.dtypes,'ad_featured.dtypes')


          
    ##开始数据合并

    train_data=pd.DataFrame()
    predict_data=pd.DataFrame()
    if readmode =='part':
        ##  raad train_data    
        cnt=0
        for df_train in pd.read_csv(open(path_train_csv,'r'),
                                    chunksize =Chunksize,nrows=readnum):
            df_train.loc[df_train['label']==-1,'label']=0
            df_data = pd.merge(df_train,ad_feature,on='aid',how='left')
            df_data =pd.merge(df_data,user_feature,on='uid',how='left')
            if cnt==0:
                train_data = df_data
                mprint(hex(id(train_data)),'train_data')
            else:
                train_data = pd.concat([train_data,df_data])
                mprint(hex(id(train_data)),'train_data')

            cnt=cnt+1
            mprint('chunk %d done.' %cnt)       
        timespent('train_data read finished')
    
           

    else:
        ##  raad train_data   
    ##0515 修改为一次性读取
        cnt=0
        for df_train in pd.read_csv(open(path_train_csv,'r'),
                                    chunksize =Chunksize):
            df_train.loc[df_train['label']==-1,'label']=0
            df_data = pd.merge(df_train,ad_feature,on='aid',how='left')
            df_data =pd.merge(df_data,user_feature,on='uid',how='left')
            if cnt==0:
                train_data = df_data
            else:
                train_data = pd.concat([train_data,df_data])
            cnt=cnt+1
            del df_data
            gc.collect()
            mprint('chunk %d done.' %cnt)       
            
        timespent('train_data read finished')

        
    del user_feature
    del ad_feature
    gc.collect()
    mprint (mem_usage(train_data),'mem_usage(train_data)')          
    mprint(train_data.dtypes,'train_data.dtypes')
    len_train_data= len(train_data)
    mprint('len_train_data %d'%(len_train_data))


    len_train_data_postive= len(train_data[train_data['label']==1])
    len_train_data_negative= len(train_data[train_data['label']==0])
    mprint(len_train_data_postive,'len_train_data_postive')
    mprint(len_train_data_negative,'len_train_data_negative')
    N_P_ratio =float(len_train_data_negative/len_train_data_postive)
    mprint('N/P ratio: %s'%(str(N_P_ratio)))
    train_data.fillna('-1',inplace= True)
    # predict_data.fillna('-1',inplace =True)
    data= train_data
    mail('data fillna and merged!')  
    del train_data
    del predict_data    
    mprint (mem_usage(data),'mem_usage(data)')          

    mprint(data.dtypes,'data dtypes')

    mprint('start gc.collect')
    gc.collect()
    mprint('stop gc.collect')

    one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
           'adCategoryId', 'productId', 'productType']
    vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
    for feature in one_hot_feature:
        try:
            mprint('%s LabelEncoder apply int  '%(feature))
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))

        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
            mprint('%s LabelEncoder failed !'%(feature))
    mprint('LabelEncoder finished!')

    ## DATA DTYPES SAVES

    dtypes = data.dtypes
    dtypes_col = dtypes.index
    dtypes_type = [i.name for i in dtypes.values]

    column_types = dict(zip(dtypes_col, dtypes_type))

    with open(path_data_dtypes,"w") as f:
            f.write(str(column_types))


    try:
        data.to_csv(path_data_csv,index=False)
        timespent('data_to_csv finished!')
    except:
        mprint('data_to_csv failed!')
    #try:
    #    
    #    h5_file = h5py.File(path_data_hdf5,'w')
    #    for col in data.columns:
    #        dtype = data[col].dtypes
    #        mprint(dtype)
    #        dset =h5_file.create_dataset(name =col,data = data[col],dtype= dtype)
    #        mprint(' h5py  create_dataset %s sucessed'%(col))
    #        mprint('h5_file sucessedd')
    #except:
    #    mprint('h5_file failed')
else:
    pass 
##负样本聚类
##负样本按正负比抽样 N/P平衡
##正样本不变
one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']

#one_hot_feature=['LBS']
#vector_feature=['kw1']

####    开始ONEHOT 编码和稀疏向量化
data_negative = data.loc[data['label']==0]

data_negative_cluster = data_negative.loc[:,['uid','aid','label']]
#data_negative_cluster = data_negative
data_postive= data.loc[data['label']==1]
#data_postive_cluster =data_postive
data_postive_cluster = data_postive.loc[:,['uid','aid','label']]
len_data_postive_cluster =len (data_postive_cluster)
mprint('data N/P split is done!')
N_P_ratio = float(len(data_negative)/len(data_postive))
del data_postive
del data
gc.collect()
data_negative_x=data_negative[['creativeSize']]

##负样本稀疏处理
mprint('onehot_trans begin')
for feature in one_hot_feature:
#for feature in one_hot_feature:

    enc = OneHotEncoder()
    tmp_enc = enc.fit_transform(data_negative[feature].values.reshape(-1, 1))
#    enc.fit(data_negative[feature].values.reshape(-1, 1))
    mprint(enc.n_values_,'feature:%s enc.n_values_'%(feature))
#    tmp_enc=enc.transform(data_negative[feature].values.reshape(-1, 1))
    data_negative_x= sparse.hstack((data_negative_x, tmp_enc))
    
    # tmp_enc=enc.transform(valid[feature].values.reshape(-1, 1))
    # valid_x= sparse.hstack((valid_x, tmp_enc))  

    # tmp_enc=enc.transform(test[feature].values.reshape(-1, 1))
    # test_x= sparse.hstack((test_x, tmp_enc))

    # tmp_enc=enc.transform(test_off[feature].values.reshape(-1, 1))
    # test_off_x= sparse.hstack((test_off_x, tmp_enc))
    del tmp_enc
    data_negative=data_negative.drop(feature,axis=1)
    # valid=valid.drop(feature,axis=1)
    # test=test.drop(feature,axis=1)
    # test_off=test_off.drop(feature,axis=1)
    gc.collect()
    mprint (mem_usage(data_negative),'mem_usage(data_negative) after onehot_trans %s'%(feature))
    # mprint (mem_usage(data_negative),'mem_usage(data_negative) after onehot_trans %s'%(feature))
    # mprint (mem_usage(valid),'mem_usage(valid) after %s'%(feature))
    # mprint (mem_usage(test),'mem_usage(test) after %s'%(feature))
    # mprint (mem_usage(test_off),'mem_usage(test_off) after %s'%(feature))

    mprint('feature:%s one-hot finished!'%(feature))

mprint('onehot_trans prepared !')

mprint('countvec_trans begin')

for feature in vector_feature:
    cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
#    cv.fit(data_negative[feature])
    
    tmp_enc=cv.fit_transform(data_negative[feature])
    data_negative_x= sparse.hstack((data_negative_x, tmp_enc))

    # tmp_enc=cv.transform(valid[feature])
    # valid_x= sparse.hstack((valid_x, tmp_enc))  

    # tmp_enc=cv.transform(test[feature])
    # test_x= sparse.hstack((test_x, tmp_enc))

    # tmp_enc=cv.transform(test_off[feature])
    # test_off_x= sparse.hstack((test_off_x, tmp_enc))
    gc.collect()
    del tmp_enc

    data_negative=data_negative.drop(feature,axis=1)
    # valid=valid.drop(feature,axis=1)
    # test=test.drop(feature,axis=1)
    # test_off=test_off.drop(feature,axis=1)

    mprint (mem_usage(data_negative),'mem_usage(data_negative) after countvec_trans %s'%(feature))
    # mprint (mem_usage(data_negative),'mem_usage(data_negative) after countvec_trans %s'%(feature))
    # mprint (mem_usage(valid),'mem_usage(valid) after %s'%(feature))
    # mprint (mem_usage(test),'mem_usage(test) after %s'%(feature))
    # mprint (mem_usage(test_off),'mem_usage(test_off) after %s'%(feature))

    mprint('feature:%s CountVectorizer finished!'%(feature))
try:
    
    len_data_negative_x =data_negative_x.getnnz()
    
except:
    len_data_negative_x=0
mprint('countvec_trans prepared !')
mprint(type(data_negative_x),'data_negative_x')
mprint(data_negative_x.shape,'data_negative_x.shape')
mprint(data_negative_x.row,'data_negative_x.row')
mprint(data_negative_x.col,'data_negative_x.col')

timespent('countvec_trans prepared ')

data_negative_x_tocsr =data_negative_x.tocsr()
mprint(type(data_negative_x_tocsr),'data_negative_x_tocsr')


mprint('begin minibatchmeans')
nrows =data_negative_x_tocsr.shape[0]
km_batchsize = 100000
n_clusters= 1000
classes_data_negative_x =[]

from sklearn.cluster import MiniBatchKMeans
#mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=500, reassignment_ratio=10**-4) 

mbk = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, batch_size=100,
  verbose=0, compute_labels=True, random_state=2018,tol=0.0, max_no_improvement=10, init_size=3*n_clusters, n_init=3, reassignment_ratio=0.01)
#batch 100
#nrows =201
#chunk=1
#ct= 2
#left_data 1
circle_times = int(nrows/km_batchsize)
left_count = int(nrows%km_batchsize)
for i  in range (circle_times):
    bgn_index = i*km_batchsize
    end_index = (i+1)*km_batchsize
    data_negative_x_tocsr_i =data_negative_x_tocsr[bgn_index:end_index,:]
    mbk.partial_fit(data_negative_x_tocsr_i)
    mprint (data_negative_x_tocsr_i.shape)
    classes_data_negative_x = np.append(classes_data_negative_x, mbk.labels_)
    timespent('chunk %s bgn_index:%s end_index:%s'%(str(i+1),str(bgn_index),str(end_index)))
    
bgn_index =circle_times*km_batchsize
end_index =circle_times*km_batchsize+left_count
mprint('last chunk %s bgn_index:%s end_index:%s'%(str(circle_times+1),str(bgn_index),str(end_index)))
data_negative_x_tocsr_end = data_negative_x_tocsr[bgn_index:end_index,:]
mbk.partial_fit(data_negative_x_tocsr_end)
mprint (data_negative_x_tocsr_end)
classes_data_negative_x = np.append(classes_data_negative_x, mbk.labels_)
## 尝试KMEANS聚
mprint (classes_data_negative_x.shape,'classes_data_negative_x.shape')

len_classes_data_negative_x= len(classes_data_negative_x)

mprint(len_data_postive_cluster,'len_data_postive_cluster')
mprint(len_classes_data_negative_x,'len_classes_data_negative_x')
n_clusters = int(mbk.labels_.max())

## 负样本类别添加
## 正样本类别置0
data_negative_cluster['class']=classes_data_negative_x.astype(int)+1
data_postive_cluster['class']=0
del classes_data_negative_x
timespent('data_cluster classfied finished')
data_negative_cluster.to_csv(path_data_negative_cluster)
data_postive_cluster.to_csv(path_data_postive_cluster)

##  采样
frac_ratio = 1.0/N_P_ratio
mprint('负样本采样率')
data_cluster = data_postive_cluster
classes_null =[]
for i in range(1,n_clusters+1,1):
    try:
        data_negative_class_i = data_negative_cluster.loc[data_negative_cluster['class']==i]
        data_negative_class_i = data_negative_class_i.sample(frac = frac_ratio)
        data_cluster = pd.concat([data_cluster, data_negative_class_i])
    except:
        classes_null.append(i)
        continue
mprint(classes_null,'classes_null')
timespent('data_negative sapmle done! frac_ratio %s'%(str(1/frac_ratio)))

data_cluster.to_csv(path_train_cluster_class)
timespent("train_cluster_class has been wirted as csv.")    
data_cluster = data_cluster.drop('class',axis=1)
data_cluster.to_csv(path_train_cluster,index = False)
timespent("train_cluster  has been wirted as csv.")    


## 重新利用
