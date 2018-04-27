# -*- coding:utf8 -*-
from apscheduler.schedulers.blocking import BlockingScheduler
import tushare as ts
from pydub import AudioSegment
import pygame

class eagle_eye_bot(object):
    def __init__(self):
        self._scheduler=BlockingScheduler()

    def start(self):
        try:
            self._scheduler.start()
            print('start to monitor which stock should buy')
        except(KeyboardInterrupt, SystemExit):
            self._scheduler.remove_all_jobs()

    def add_buy_stock(self,stock_code,buy_price=-1):
        self._scheduler.add_job(self.__stock_monitor_on_second,'interval',seconds=3,id='monitor_'+str(stock_code),args=[stock_code,buy_price])

    def add_sell_stock(self,stock_code):
        """

        :param stock_code:
        :param yestoday_close:
        :return:
        """
        self._scheduler.add_job(self.__stock_sell_on_second, 'interval', seconds=1, id='monitor_sell_' + str(stock_code),
                                args=[stock_code])

    def __stock_monitor_on_second(self,stock_code,buy_price):
        data=ts.get_realtime_quotes(stock_code)
        price=float(data['price'][0])
        # print('==================================================')
        # print('stock:%f', stock_code)
        # print('current_price:%f', price)
        # print('buy_price:%f', buy_price)
        rate=price/buy_price
        # print('rate:%f',rate)
        if rate<1.005:
            print('++++++++++++++++++++++++++++++++++++++++++++++++')
            print('stock:', stock_code)
            print('current_price:', price)
            print('buy_price:', buy_price)

    def __stock_sell_on_second(self,stock_code):
        data=ts.get_realtime_quotes(stock_code)
        # pre_close=data['pre_close'][0]
        price=data['price'][0]
        high=data['high'][0]
        open=data['open'][0]
        if price/open<0.8:
            print('--------------------')
            print('sell stock:', stock_code)
            print('sell price:', price)
        elif high/price>1.1:
            print('--------------------+=')
            print('sell stock:',stock_code)
            print('sell price:',price)

def troggle_buy_audio():
    file='I_wanted_you.mp3'
    import time
    # file = '/home/leonardo/Music/CloudMusic/Ina - I wanted you.mp3'
    # buy_sound=AudioSegment.from_mp3('../data/resource/audio/I_wanted_you.mp3')

    from pygame import mixer
    mixer.init()
    mixer.music.load(file)
    mixer.music.play()
    time.sleep(5)

if __name__=='__main__':
    troggle_buy_audio()