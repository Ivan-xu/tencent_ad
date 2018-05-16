# -*- coding: utf-8 -*
from ignore import my_sender,my_pass,my_user

import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from sklearn import preprocessing
import numpy as np
import pandas as pd
def_path_log_path  ='log_ad_'
mail_key = ['on','off'][0]

def mail(msg=''):
    if mail_key =='off':
        mprint('邮件通知关闭',msg)
        return True
    ret=True
    try:
        if msg=='':
            msg=MIMEText('填写邮件内容','plain','utf-8')
        else:
            msg=MIMEText(msg,'plain','utf-8')
        msg['From']=formataddr(["我的139邮箱",my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To']=formataddr(["我的QQ邮箱",my_user])              # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject']="ML_AD_算法"                # 邮件的主题，也可以说是标题
 
        server=smtplib.SMTP_SSL("smtp.139.com", 465)  # 发件人邮箱中的SMTP服务器，端口是465
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender,[my_user,],msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()# 关闭连接
        mprint("邮件发送成功",msg)
    except Exception:# 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret=False
        print("邮件发送失败")
    return ret
 
#ret=mail('1008611')
#if ret:
#    print("邮件发送成功")
#else:
#    print("邮件发送失败")
    
def select_best(arr1,arr2,lamda =-10,fator =2):
    if len(arr1) == len(arr2):
        min_max_scaler = preprocessing.MinMaxScaler()
        arr2_scale = min_max_scaler.fit_transform(np.transpose([arr2]))
        score =[]
        for i in range(len(arr1)):
            score_i = arr1[i]+lamda*arr2_scale[i][0]**fator
            score.append(score_i)
            print([i,arr1[i],arr2[i],arr2_scale[i][0],score[i]])
            
        print (max(score))
        result = (score.index(max(score)))
        return['True',result,score]
    else:
        return ['False','DATA ERROR']
# -*- coding: utf-8 -*
    
'''
@author: PY131
'''
def wrlog(file,data,msg=''):
    with open(file,"a") as f:
#        print (type(msg))
        if type(data) ==type([]):
            for i in range(len(data)):
                f.write(str(data[i])+'\n')
        else :
            f.write(str(data)+'\n')
#        if msg=='':
#            f.write('msg is logged!\n')
#        else :
#            f.write(str(msg) +' is logged!\n')
#wrlog('test.txt',1,'msg11')
from datetime import datetime
date = datetime.now().strftime('%Y%m%d-%H')
#time=datetime.now().strftime('%H:%M:%S')
def_path_log = def_path_log_path+str(date)+'.txt'
def myprint(data,file=def_path_log,mode='w'):
    print (data)
    if mode =='w':
        with open(file,"a") as f:
            f.write(str(data)+'\n')
def mprint (data,msg=''):
    time=datetime.now().strftime('%H:%M:%S')    
    print (data)
    with open(def_path_log,"a") as f:
        f.write(str(time)+': '+msg+'\n'+str(data)+'\n\n')
        

# We're going to be calculating memory usage a lot,
# so we'll create a function to save us some time!

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)        
#import os
#import sys
#import timeit
#
#start_time = timeit.default_timer()
#
#end_time = timeit.default_timer()
#
#print(('The code for file ' + os.path.split(__file__)[1] +
#       ' ran for %.2fm' % ((end_time - start_time) / 60.)), file = sys.stderr)