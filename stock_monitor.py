# -*- coding:utf8 -*-
import tushare as ts
from apscheduler.schedulers.blocking import BlockingScheduler
import pandas as pd

def monitor_on_tick():
    scheduler=BlockingScheduler()
    scheduler.add_job(show_real_data,'interval',seconds=3,id='monitor on every second',args=[['600808','600230']])

    try:
        scheduler.start()
    except(KeyboardInterrupt,SystemExit):
        scheduler.remove_job('monitor on every second')

def show_real_data(security='000001'):
    data=ts.get_realtime_quotes(security)
    i=0
    print(data['time'][0])
    for code in security:
        print(code,data['price'][i])
        i+=1
    return data

def handle_data(security='000001'):
    pass

if __name__=='__main__':
    monitor_on_tick()
    # all_stock = ts.get_industry_classified(standard='sw')#
    # all_stock.to_csv('data/resource/all_industry_stock_from_sw.csv',encoding='utf8')
    # all_stock=pd.read_csv('data/resource/all_industry_stock_from_sw.csv',encoding='utf8')
    # all_stock.d
    # industry= set(all_stock['c_name'].tolist())
    # for ind in industry:
    #     print(ind)
    # home_stock=all_stock[all_stock['code']=='002230']
    #
    # print home_stock
    # print ts.get_realtime_quotes(['000001','600230'])

