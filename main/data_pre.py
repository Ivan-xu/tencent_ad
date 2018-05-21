

##负样本聚类
##负样本按正负比抽样 N/P平衡
##正样本不变
##
####    开始ONEHOT 编码和稀疏向量化
data_negative = data.loc[data['label']==0]
# data_negative_y = data_negative.pop('label',axis=1)
# data_negative_cluster = data_negative['uid','aid','label']
data_negative_cluster = data_negative.loc[:,['uid','aid','label']]

data_postive_cluster = data.loc[[data['label']==1],['uid','aid','label']]
del data
gc.collect()
data_negative_x=data_negative[['creativeSize']]

##负样本稀疏处理
mprint('onehot_trans begin')
for feature in one_hot_feature:
    enc = OneHotEncoder()
    enc.fit(data_negative[feature].values.reshape(-1, 1))
    mprint(enc.n_values_,'feature:%s enc.n_values_'%(feature))
    tmp_enc=enc.transform(data_negative[feature].values.reshape(-1, 1))
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
    cv.fit(data_negative[feature])

    tmp_enc=cv.transform(data_negative[feature])
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
len_data_negative_x =len(data_negative_x)
mprint('countvec_trans prepared !')

##负样本抽样
## 尝试KMEANS聚类
from sklearn.cluster import MiniBatchKMeans


classes_data_negative_x =[]
n_clusters= 500
mbk1 = MiniBatchKMeans(n_clusters=n_clusters, init=’k-means++’, max_iter=100, batch_size=100,\
 verbose=0, compute_labels=True, random_state=None,tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
timespent('begin_MiniBatchKMeans')
mbk.fit(data_negative_x)
timespent('finished_MiniBatchKMeans')
classes_data_negative_x = np.append(classes_data_negative_x, mbk.labels_)
len_classes_data_negative_x= len(classes_data_negative_x)

mprint(len_data_negative_x,'len_data_negative_x')
mprint(len_classes_data_negative_x,'len_classes_data_negative_x')
mprint (len_data_negative_x==len_classes_data_negative_x,'聚类前后长度')

## 类别添加
data_negative_cluster['class']=classes_data_negative_x.astype(int)+1
data_postive_cluster['class']=0
timespent('data_cluster finished')
##  采样
frac_ratio =1/19
data_cluster = data_negative_cluster
for i in range(1,n_clusters+1,1):
    data_negative_class_i = data_negative_cluster[data_negative_cluster['class'] == i]
    data_negative_class_i = data_negative_class_i.sample(frac = frac_ratio)
    data_cluster = pd.concat([data_cluster, data_negative_class_i])

timespent('data_negative sapmle done! frac_ratio %s'%(str(1/frac)))

print("training subset uic_label keys is selected.")    

