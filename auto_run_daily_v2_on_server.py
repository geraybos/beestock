# -*- coding:utf8 -*-
import stock_model_sys_v2
from apscheduler.schedulers.background import BackgroundScheduler,BlockingScheduler
from apscheduler.executors.pool import ProcessPoolExecutor
from datetime import datetime,timedelta

def run():
    executors={
        'default': {'type': 'threadpool', 'max_workers': 10},
        'processpool': ProcessPoolExecutor(max_workers=5)
    }
    scheduler=BlockingScheduler()
    scheduler.configure(executors=executors)

    start = datetime.today() - timedelta(days=45)
    start = start.strftime('%Y-%m-%d')
    # source_dir = 'data/industry_sw/'
    industry = 'all'
    max_file_count = 1000
    seq_dim = 20
    input_dim = 5
    out_dim = 8
    #add job for computing trendency of all stock
    scheduler.add_job(stock_model_sys_v2.select_best_stock_at_yestoday,'cron',day_of_week='mon-fri',hour=3,minute=0
                      ,args=[start,industry,max_file_count,seq_dim,input_dim,out_dim])
    #select best stock when matket has opened
    scheduler.add_job(stock_model_sys_v2.select_best_stock_after_open,'cron',day_of_week='mon-fri',hour=9,minute=26,
                      args=[start,industry,max_file_count,seq_dim,input_dim,out_dim])

    try:
        scheduler.start()
    except(KeyboardInterrupt, SystemExit):
        scheduler.remove_all_jobs()

if __name__=='__main__':
    run()