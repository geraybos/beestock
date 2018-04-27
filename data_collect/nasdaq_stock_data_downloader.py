# -*- coding:utf8 -*-
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

import os

def download_one_dayily(filename,symbol=None,ts=None):
    # ts=TimeSeries(key='YPBHUAIW5R00BCBV',output_format='pandas')
    # data,metadata=ts.get_daily(symbol=symbol,outputsize='full')
    data, metadata =ts.get_intraday(symbol=symbol,interval='30min',outputsize='full')
    data.to_csv(filename)

def download_daily(path,security_list=[],file_number=0):
    ts = TimeSeries(key='YPBHUAIW5R00BCBV', output_format='pandas')
    i=file_number
    for security in security_list:
        security=security.strip()
        filename=path+'/'+security+'.csv'
        download_one_dayily(filename,symbol=security,ts=ts)
        print('download %s completed at %d'%(filename,i))
        i+=1
def filter_company_list():
    security_list = pd.read_csv('../data/resource/companylist.csv')
    security_list=security_list[(security_list.LastSale.str.replace('.','',1).str.isdigit())]
    security_list = security_list[(security_list.MarketCap!='n/a')]
    security_list.to_csv('../data/resource/companylist_1.csv')

if __name__=='__main__':
    save_path='../data/nasdaq_m30/all'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path+'/model')
        os.makedirs(save_path + '/test')
        os.makedirs(save_path + '/validation')
    security_list=pd.read_csv('../data/resource/companylist_1.csv')

    file_count=len(os.listdir(save_path))-3

    security_list=security_list[file_count:]
    security_list=security_list['Symbol']
    print('start at %d-th file named %s'%(file_count,security_list[file_count]))#
    download_daily(path=save_path,security_list=security_list,file_number=file_count)
    # filter_company_list()