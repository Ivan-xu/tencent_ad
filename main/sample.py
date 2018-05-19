# -*- coding: utf-8 -*
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from sklearn import preprocessing
import numpy as np
import pandas as pd
##my privacy 
from ignore import my_sender,my_pass,my_user
from ignore import ftp_host,ftp_username,ftp_password
##mode windows
sysmode =['windows','ubuntu'][1]
readmode =['part','whole'][1]
params_flag =False
testmode =False

## mode ubuntu 
#sysmode =['windows','ubuntu'][1]
#readmode =['part','whole'][1]
#params_flag =False

if sysmode =='ubuntu':
    def_path_log_path='/root/workspace/log/ad_'
else :
    def_path_log_path ='../log/ad_'
    
    
mail_key = ['on','off'][1]

def mail(orimsg=''):
    if mail_key =='off':
        mprint('邮件通知关闭',orimsg)
        return True
    ret=True
    try:
        if orimsg=='':
            msg=MIMEText('填写邮件内容','plain','utf-8')
        else:
            msg=MIMEText(orimsg,'plain','utf-8')
        msg['From']=formataddr(["我的139邮箱",my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To']=formataddr(["我的QQ邮箱",my_user])              # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject']="ML_AD_算法"                # 邮件的主题，也可以说是标题
 
        server=smtplib.SMTP_SSL("smtp.139.com", 465)  # 发件人邮箱中的SMTP服务器，端口是465
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender,[my_user,],msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()# 关闭连接
        mprint("邮件发送成功",orimsg)
    except Exception:# 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret=False
        print("邮件发送失败")
    return ret
 

    
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
date = datetime.now().strftime('%Y%m%d')
#time=datetime.now().strftime('%H:%M:%S')
def_path_log = def_path_log_path+str(date)+'.txt'

def mprint (data,msg=''):
    time=datetime.now().strftime('%H:%M:%S')    
    print (msg)
    print (data)
    print('\n')
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
  
import ftplib
def ftp_upload(file_remote,file_local):
    mprint('ftp ing...')
    f = ftplib.FTP() 
    f.set_debuglevel(2)    
    f.connect(ftp_host, 21)
    f.login(ftp_username, ftp_password)
    mprint (f.getwelcome())  # 获得欢迎信息   
    cwdpath ='/anonymous/upload/'
    f.cwd(cwdpath)    # 设置FTP路径  
    listf = f.nlst()       # 获得目录列表  
    for name in listf:  
        mprint(name)      
    '''以二进制形式上传文件'''
#    file_remote = 'ftp_upload.txt'
#    file_local = 'D:\\test_data\\ftp_upload.txt'
    bufsize = 1024  # 设置缓冲器大小
    fp = open(file_local, 'rb')
    file_remote =cwdpath+file_remote
    f.storbinary('STOR ' + file_remote, fp, bufsize)
    fp.close()
    mprint ('ftp done!')

if testmode ==True:
    try:
        mail('test mail ')
    except:
        mprint('test mail send failed!')
    try:
        file_remote ='test1.csv'
        file_local ='/root/workspace/data/test1.csv'
        ftp_upload(file_remote,file_local)
        mprint('test  ftp upload sucess')

    except:
        mprint('test ftp upload failed')

else:
    pass
#file_remote ='test1.csv'
#file_local ='/root/workspace/data/test1.csv'
#ftp_upload(file_remote,file_local)
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