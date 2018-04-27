# -*- coding:utf8 -*-
import mysql.connector
from datetime import datetime,timedelta
import stock_model_sys_v2
import pandas as pd
import tushare as ts
from k_pattern import stock_pattern_by_main


def db_find(cursor,sql='select count(*) from mysql_table'):
    cursor.execute(sql)
    return cursor.fetchall()

def db_update(db,cursor,sql='update mysql_table set column=value'):
    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()

def select_best_stock_after_open(stocks,start_date='', industry_name='jisuanji', max_file_count=None, seq_dim=5, input_dim=5, out_dim=2):
    # stocks = pd.read_csv('data/result/hope_stock_'+datetime.now().strftime('%Y-%m-%d')+'.csv', encoding='utf8')

    seq_dim_=seq_dim+1
    for index,stock in stocks.iterrows():
        if stock['delta']<0:
            break
        code=stock['code']

        raw_data = ts.get_k_data(code, start=start_date)
        today_open=ts.get_realtime_quotes(code)

        if today_open is None:
            stocks.drop([index], inplace=True)
            continue
        today_open=float(today_open.ix[today_open.index[-1],'open'])
        raw_data=raw_data.append(pd.DataFrame([[today_open,today_open,today_open,today_open,today_open]],index=[-1],columns=['open','close','high','low','volume']))

        if len(raw_data) >= seq_dim_:
            raw_data=raw_data[-seq_dim_:]
            input_data=stock_model_sys_v2.format_time_series_data(raw_data)
            if stock_pattern_by_main.pattern_analysis(input_data)<0:# or code.startswith('300'):#and stock_pattern_by_main.previous_h_line(input_data)>0
                stocks.drop([index],inplace=True)
            # else:
                #print(stock)
        else:
            stocks.drop([index], inplace=True)
    stocks.to_csv('data/result/hope_stock_'+datetime.now().strftime('%Y-%m-%d')+'_final.csv')

def select_best_stock():
    source_dir = 'data/industry_sw/'
    industry = 'all'
    max_file_count = 1000
    seq_dim = 20
    input_dim = 5
    out_dim = 8

    start = datetime.today() - timedelta(days=45)
    start = start.strftime('%Y-%m-%d')
    stock_model_sys_v2.select_best_stock_at_yestoday(start_date=start, industry_name=industry, max_file_count=max_file_count, seq_dim=seq_dim,
              input_dim=input_dim, out_dim=out_dim)

    # stock_model_sys_v2.select_best_stock_after_open(start_date=start, industry_name=industry, max_file_count=max_file_count,
    #                              seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
def run():
    if datetime.today().weekday()<5:#exclude Saturday,Sunday
        source_dir = 'data/industry_sw/'
        industry = 'all'
        max_file_count = 1000
        seq_dim = 20
        input_dim = 5
        out_dim = 8
        start = datetime.today() - timedelta(days=45)
        start = start.strftime('%Y-%m-%d')

        db=mysql.connector.connect(user='root',password='yangxh',database='quant_bee',use_unicode=True)
        cursor=db.cursor()
        sql='select value from leo_env_config where groupId=1 and name="run_status"'
        rs=db_find(cursor,sql)
        rs_set=rs[0][0].split('_')
        if len(rs_set)==2:
            run_date=datetime.strptime(rs_set[0],'%Y-%m-%d')
            if datetime.today().day==run_date.day and datetime.today().month==run_date.month:
                if rs_set[1]=='2':
                    print('it has been run on '+datetime.today().strftime('%y-%m-%d'))
                elif rs_set[1]=='1':
                    stock_model_sys_v2.select_best_stock_after_open(start_date=start, industry_name=industry, max_file_count=max_file_count,
                                     seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
            else:
                select_best_stock()
                print('run successfully on ' + datetime.today().strftime('%y-%m-%d'))
        run_status=datetime.today().strftime('%Y-%m-%d')+"_2"
        sql='update leo_env_config set value="'+run_status+'" where groupId=1 and name="run_status"'
        db_update(db,cursor,sql)
        db.close()

def check_by_classify(classify=''):
    ts.get_concept_classified()

if __name__=='__main__':
    run()
    # stocks = pd.read_csv('data/result/hope_stock_' + datetime.now().strftime('%Y-%m-%d') + '.csv', encoding='utf8')
    # stocks.set_value(stocks.shape[0]-1,'code','str')
    # stocks.to_csv('data/result/hope_stock_' + datetime.now().strftime('%Y-%m-%d') + '.csv', encoding='utf8')
    # print 'h'