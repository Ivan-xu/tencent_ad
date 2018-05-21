
##训练集包含正负样本
## 线上测试集
test=data[data.label==-1]
test=test.drop('label',axis=1)
res=test[['aid','uid']]
mprint('data set test split finished')
##  训练集
train=data[data.label!=-1]
train_y=train.pop('label')
mprint('data set train split finished')
mprint('del data and free m')
gc.collect()

##删除完整集
#del data
# 训练集、线下测试集
train, test_off, train_y, test_off_y = train_test_split(train,train_y,test_size=0.15, random_state=2018)
mprint ('data set offline split finished')
##训练集、验证集
train, valid, train_y, valid_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
mprint('data set valid split finished')
mem_usage_data_ori =(mem_usage(data))
mem_usage_train_ori =(mem_usage(train))

mem_usage_valid_ori =(mem_usage(valid))

mem_usage_test_ori =(mem_usage(test))
mem_usage_test_off_ori =(mem_usage(test_off))


####    开始ONEHOT 编码和稀疏向量化

train_x=train[['creativeSize']]
valid_x=valid[['creativeSize']]
test_x=test[['creativeSize']]
test_off_x=test_off[['creativeSize']]
mprint('onehot_trans begin')
for feature in one_hot_feature:
    enc = OneHotEncoder()
    enc.fit(data[feature].values.reshape(-1, 1))
    mprint(enc.n_values_,'feature:%s enc.n_values_'%(feature))
    tmp_enc=enc.transform(train[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, tmp_enc))
    
    tmp_enc=enc.transform(valid[feature].values.reshape(-1, 1))
    valid_x= sparse.hstack((valid_x, tmp_enc))  

    tmp_enc=enc.transform(test[feature].values.reshape(-1, 1))
    test_x= sparse.hstack((test_x, tmp_enc))

    tmp_enc=enc.transform(test_off[feature].values.reshape(-1, 1))
    test_off_x= sparse.hstack((test_off_x, tmp_enc))
    del tmp_enc

    data=data.drop(feature,axis=1)
    train=train.drop(feature,axis=1)
    valid=valid.drop(feature,axis=1)
    test=test.drop(feature,axis=1)
    test_off=test_off.drop(feature,axis=1)
    gc.collect()
    mprint (mem_usage(data),'mem_usage(data) after %s'%(feature))
    mprint (mem_usage(train),'mem_usage(train) after %s'%(feature))
    mprint (mem_usage(valid),'mem_usage(valid) after %s'%(feature))
    mprint (mem_usage(test),'mem_usage(test) after %s'%(feature))
    mprint (mem_usage(test_off),'mem_usage(test_off) after %s'%(feature))

    mprint('feature:%s one-hot finished!'%(feature))

mprint('onehot_trans prepared !')

mprint('countvec_trans begin')

for feature in vector_feature:
    cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    cv.fit(data[feature])

    tmp_enc=cv.transform(train[feature])
    train_x= sparse.hstack((train_x, tmp_enc))

    tmp_enc=cv.transform(valid[feature])
    valid_x= sparse.hstack((valid_x, tmp_enc))  

    tmp_enc=cv.transform(test[feature])
    test_x= sparse.hstack((test_x, tmp_enc))

    tmp_enc=cv.transform(test_off[feature])
    test_off_x= sparse.hstack((test_off_x, tmp_enc))
    gc.collect()
    del tmp_enc

    data=data.drop(feature,axis=1)
    train=train.drop(feature,axis=1)
    valid=valid.drop(feature,axis=1)
    test=test.drop(feature,axis=1)
    test_off=test_off.drop(feature,axis=1)

    mprint (mem_usage(data),'mem_usage(data) after %s'%(feature))
    mprint (mem_usage(train),'mem_usage(train) after %s'%(feature))
    mprint (mem_usage(valid),'mem_usage(valid) after %s'%(feature))
    mprint (mem_usage(test),'mem_usage(test) after %s'%(feature))
    mprint (mem_usage(test_off),'mem_usage(test_off) after %s'%(feature))

    mprint('feature:%s CountVectorizer finished!'%(feature))

mprint('countvec_trans prepared !')

mprint((mem_usage_data_ori),'mem_usage(data) ori ')

mprint((mem_usage_train_ori),'mem_usage(train) ori ')
mprint ((mem_usage_train_ori),'mem_usage(valid) ori ')
mprint ((mem_usage_test_ori),'mem_usage(test) ori ')
mprint ((mem_usage_test_off_ori),'mem_usage(test_off) ori ')