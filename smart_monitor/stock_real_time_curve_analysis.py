# -*- coding: utf-8 -*-
import tushare as ts
import json
import logging.config
import os

def setup_logging(default_path = "logging.json",default_level = logging.INFO,env_key = "LOG_CFG"):
    path = default_path
    value = os.getenv(env_key,None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path,"r") as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level = default_level)

class stock_real_time_curve(object):
    def __init__(self,code='000001'):
        setup_logging(default_path='../data/resource/config/logging.json')
        self.code=code
        self.first_high=None
        self.second_high=None
        self.third_high=None
        self.previous_price=None
        self.previous_low=100000    #no stock price can up to 10000 until now

    def juge_tendency(self,real_data):
        self.first_high=self.second_high
        self.second_high=self.third_high
        self.third_high=real_data['price'].iat[0]

    def __juge_high(self,current_price):
        if current_price<self.previous_low:
            self.previous_low=current_price

        if self.previous_price is not None:
            if self.previous_price>current_price:
                pass
        else:#init
            self.previous_price=current_price
    def __stop_loss(self,real_data):
        """
        设置低于开盘价3个点就止损
        :param real_data:
        :return:
        """
        if real_data['open'].iat[0]/real_data['price'].iat[0]>1.03:
            return True
        return False


from jqdatasdk import *
# auth('聚宽账号','登录密码')
auth('18961841823','leonardoyang5715')
df = get_price('000001.XSHE', start_date='2015-01-01', end_date='2015-01-31 23:00:00', frequency='minute', fields=['open', 'close'])
print df