# -*- coding: utf-8 -*-
import math
class bee_stock_system(object):
    """
    realtime analysys
    """

    def __init__(self, pool_size=60, block_unit=60, freqency=60, handle_funcs=[]):
        self.pool_size = pool_size
        self.freqency = freqency
        self.handle_funcs = handle_funcs
        self.merge_flag_number = 0

    def add_handle_first(self, handle_func=None):
        self.handle_funcs.insert(0, handle_func)

    def add_handle_last(self, handle_func=None):
        self.handle_funcs.append(handle_func)

    def add_handle_at(self, handle_func=None, index=0):
        self.handle_funcs.insert(index, handle_func)

    def handle_security_unit_data(self, security_unit_data):
        self.bar_cache.add_data(security_unit_data)

    def run(self, security_unit_data):
        """
        run this method when a new data coming
        :param security_unit_data:{open,close,high,low,volume;money,factor,high_limit,low_limit,avg,pre_close,paused}
        :return:int32,if positive then buy,else sell
        """
        self.handle_security_unit_data(security_unit_data)
        sum = 0
        for handle_func in self.handle_funcs:
            sum += handle_func(security_unit_data)
        return sum
class wall_e_ai(object):

    def __init__(self,stocks=[]):
        self.hope_code=None#已买入的股票代码，全村人的希望
        self.__brothers={}#希望候选人列表

    def put_stock(self,code='000001.XSHE',cache_pool=None):
        self.__brothers[code]=cache_pool


    def eval_stock_data(self,code='000001.XSHE',bar_unit=None):
        self.__brothers[code].add_data(bar_unit)

    def check(self):
        """

        :return: buy_code,sell_code
        """
        if self.__brothers[self.hope].check()<0:#sell this stock,and find another hope

            return self.hope_code,-1
        else:
            pass

def tanh(x):
    ex=math.exp(x)
    e_x=math.exp(-x)
    return (ex-e_x)/(ex+e_x)

def macd_trend(macds):
    sum_trend=0
    for i in range(len(macds)-1):
        diff=macds[i+1]-macds[i]
        if diff<0:
            return 0
        if macds[i+1]<=0:
            trend=2.0*diff/abs(macds[i+1]+macds[i])
        else:
            trend = diff / abs(macds[i + 1] + macds[i])
        sum_trend+=trend
    return sum_trend

class bee_bar(object):
    def __init__(self, pool_size=60):
        self.open = 0
        self.close = 0
        self.high = 0
        self.low = 0
        self.avg = 0
        self.volume = 0
        self.money = 0
        self.paused = 0


class cache_pool(object):
    def __init__(self, block_unit=240, size=60, real_bar_unit=240,
                 isMA=False, MA_Params={'m_short': 5, 'm_long': 10, 'analysis_field': 'close'},
                 isMACD=False, MACD_Params={'alpha': 2, 'longperiod': 26, 'shortperiod': 12, 'signalperiod': 9,
                                            'is_macd_diff': False, 'analysis_field': 'close', 'macd_max_base': 0.0,'trend_threshold':6,'trend_bar_threshold':4},
                 isKDJ=False, KDJ_Params={'period': 9},
                 isRSI=False, RSI_Params={'N1': 6, 'N2': 12, 'N3': 24},
                 isBOLL=False,
                 isHighLow=False,HighLow_Params={'period':30}
                 ):
        """

        :param block_unit:
        :param size:
        :param pool_type: bar,macd,kdj
        """
        self.block_unit = block_unit
        self.size = size
        self.merge_size = block_unit / real_bar_unit
        self.merge_index = 0
        self.isMA = isMA
        self.MA_Params = MA_Params
        self.isMACD = isMACD
        self.isKDJ = isKDJ
        self.isRSI = isRSI
        self.isBOLL = isBOLL
        self.isHighLow=isHighLow
        self.current_index = 0
        self.previous_index = 0
        self.bar_pool = [None] * size
        self.bar_unit = {}
        if self.isMA:
            self.ma_pool = [None] * size
        if self.isMACD:
            self.macd_pool = [None] * size
            self.MACD_Params = MACD_Params
            self.MACD_Params['alpha_long'] = self.MACD_Params['alpha'] / (
                self.MACD_Params['longperiod'] + self.MACD_Params['alpha'] - 1.0)
            self.MACD_Params['_alpha_long'] = 1.0 - self.MACD_Params['alpha_long']
            self.MACD_Params['alpha_short'] = self.MACD_Params['alpha'] / (
                self.MACD_Params['shortperiod'] + self.MACD_Params['alpha'] - 1.0)
            self.MACD_Params['_alpha_short'] = 1.0 - self.MACD_Params['alpha_short']
            self.MACD_Params['alpha_signal'] = self.MACD_Params['alpha'] / (
                self.MACD_Params['signalperiod'] + self.MACD_Params['alpha'] - 1.0)
            self.MACD_Params['_alpha_signal'] = 1.0 - self.MACD_Params['alpha_signal']
            self.MACD_Params['analysis_field'] = 'close'
            self.MACD_Params['trend_bar_unit'] =self.MACD_Params['trend_threshold']-self.MACD_Params['trend_bar_threshold']

        if self.isKDJ:
            self.kdj_pool = [None] * size
            self.KDJ_Params = KDJ_Params
        if self.isRSI:
            self.rsi_pool = [None] * size
            self.RSI_Params = RSI_Params
        if self.isBOLL:
            self.boll_pool = [None] * size
        if self.isHighLow:
            self.highlow_pool=[None]*size
            self.HighLow_Params=HighLow_Params


    def add_data(self, bar_unit=None):
        """
        not thread safely.do not run it in multithread
        :param bar_unit:
        :return:
        """
        if self.merge_index >= self.merge_size or self.merge_index == 0:  # add new data
            self.bar_unit={}
            self.bar_unit['close'] = bar_unit.close
            self.bar_unit['open'] = bar_unit.open
            self.bar_unit['high'] = bar_unit.high
            self.bar_unit['low'] = bar_unit.low
            self.bar_unit['volume'] = bar_unit.volume
            self.bar_unit['money'] = bar_unit.money
            self.bar_unit['avg'] = bar_unit.avg
            self.bar_unit['paused'] = bar_unit.paused
            self.merge_index = 0
            self.__handle_data(bar_unit)
        else:
            self.__merge_bar(bar_unit)
            self.__handle_data(bar_unit, True)
        self.merge_index += 1

    def __handle_data(self, bar_unit=None, is_merge=False):
        if not is_merge:
            self.current_index += 1
        if bar_unit is not None:
            self.previous_index = self.current_index - 1
            if self.current_index >= self.size:
                self.current_index = 0

            self.bar_pool[self.current_index] = self.bar_unit
            if self.isMA:
                self.__set_ma()
            # if self.isMACD:
                # self.__set_macd()
            if self.isKDJ:
                self.__set_kdj()
            if self.isRSI:
                self.__set_rsi()
            if self.isBOLL:
                self.__set_boll()
            if self.isHighLow:
                self.__set_high_low()

    def __merge_bar(self, bar_unit=None):
        if bar_unit is not None:
            self.bar_unit['close'] = bar_unit.close
            if self.bar_unit['high'] < bar_unit.high:
                self.bar_unit['high'] = bar_unit.high
            if self.bar_unit['low'] > bar_unit.low:
                self.bar_unit['low'] = bar_unit.low
            self.bar_unit['volume'] += bar_unit.volume
            self.bar_unit['money'] += bar_unit.money
            if self.bar_unit['money']==0:
                self.bar_unit['avg']=0
            else:
                self.bar_unit['avg'] = self.bar_unit['money'] / self.bar_unit['volume']
            self.bar_unit['paused'] = bar_unit.paused

    def check(self):
        """
        positive:buy;negative:sell;zero:nothing
        :return:
        """
        if self.bar_pool[self.previous_index] is None:  # first run
            return 0
        sum = 0
        if self.isMA:
            sum += self.check_ma()
        if self.isMACD:
            if self.MACD_Params['is_macd_diff']:
                sum += self.check_macd_diff()
            else:
                sum += self.check_macd()

        if self.isKDJ:
            sum += self.check_kdj()
        if self.isRSI:
            sum += self.check_rsi()
        if self.isBOLL:
            sum += self.check_boll()

        return sum

    def history_k_bar(self,datas):
        """

        :param datas:
        :return:
        """
        index=self.current_index
        index-=len(datas)
        index+=1

        for i in range(datas.shape[0]):
            bar_unit=datas.iloc[i]
            index%=self.size
            self.bar_unit={}
            self.bar_unit['close'] = bar_unit.close
            self.bar_unit['high'] = bar_unit.high
            self.bar_unit['low'] = bar_unit.low
            self.bar_unit['volume'] = bar_unit.volume
            self.bar_unit['money'] = bar_unit.money
            self.bar_unit['avg'] = bar_unit.avg
            self.bar_unit['paused'] = bar_unit.paused

            self.bar_pool[index]=self.bar_unit
            index+=1
    def __set_ma(self):
        if self.ma_pool[self.previous_index] is None:
            self.ma_pool[self.current_index] = {}
            self.ma_pool[self.current_index]['m_short'] = self.bar_pool[self.current_index][
                self.MA_Params['analysis_field']]
            self.ma_pool[self.current_index]['m_long'] = self.bar_pool[self.current_index][
                self.MA_Params['analysis_field']]
            self.ma_pool[self.current_index]['argsum'] = self.bar_pool[self.current_index][
                self.MA_Params['analysis_field']]
            self.ma_pool[self.current_index]['dea'] = self.ma_pool[self.current_index]['m_short'] - \
                                                      self.ma_pool[self.current_index]['m_long']
            # init previous m_long size argsum
            # index = self.current_index
            # tmp = self.bar_pool[self.current_index]['close']
            # index-=1
            # argsum = 0
            # for _ in range(self.MA_Params['m_long']):
            #     index %= self.size
            #     self.ma_pool[index] = {}
            #     self.ma_pool[index]['argsum'] = argsum
            #     argsum -= tmp
            #     index -= 1
        else:
            self.ma_pool[self.current_index] = {}
            m_short_index = (self.current_index - self.MA_Params['m_short']) % self.size
            m_long_index = (self.current_index - self.MA_Params['m_long']) % self.size
            self.ma_pool[self.current_index]['argsum'] = self.ma_pool[self.previous_index]['argsum'] + \
                                                         self.bar_pool[self.current_index][
                                                             self.MA_Params['analysis_field']]
            self.ma_pool[self.current_index]['m_short'] = (self.ma_pool[self.current_index]['argsum'] -
                                                           self.ma_pool[m_short_index]['argsum']) / self.MA_Params[
                                                              'm_short']
            self.ma_pool[self.current_index]['m_long'] = (self.ma_pool[self.current_index]['argsum'] -
                                                          self.ma_pool[m_long_index]['argsum']) / self.MA_Params[
                                                             'm_long']
            self.ma_pool[self.current_index]['dea'] = self.ma_pool[self.current_index]['m_short'] - \
                                                      self.ma_pool[self.current_index]['m_long']

    def history_ma(self, data):
        # init previous m_long size argsum,it must run before add_data()
        index = self.current_index

        # index -= 1
        argsum = 0
        for i in range(self.MA_Params['m_long']):
            index %= self.size
            self.ma_pool[index] = {}
            self.ma_pool[index]['argsum'] = argsum
            argsum -= data[self.MA_Params['analysis_field']][-(i + 1)]
            index -= 1

    def check_ma(self):
        if self.ma_pool[self.previous_index]['dea'] < 0:
            if self.ma_pool[self.current_index]['dea'] >= 0:  # golden to buy
                return 1
        elif self.ma_pool[self.previous_index]['dea'] > 0:
            if self.ma_pool[self.current_index]['dea'] <= 0:  # dead to sell
                return -1

        return 0  # nothing

    def history_macd(self, longema=0, shortema=0, dif=0, dea=0, macd=0):
        """it must run before add_data()"""
        self.macd_pool[self.current_index] = {}
        self.macd_pool[self.current_index]['longema'] = longema
        self.macd_pool[self.current_index]['shortema'] = shortema
        self.macd_pool[self.current_index]['dif'] = dif
        self.macd_pool[self.current_index]['dea'] = dea
        self.macd_pool[self.current_index]['macd'] = macd
        self.MACD_Params['macd_max_base'] = macd
        self.macd_pool[self.current_index]['macd_trending'] = 0
        self.macd_pool[self.current_index]['macd_diff'] = 0

    def __set_macd(self):
        if self.macd_pool[self.previous_index] is None:  # first run ,init todo
            self.macd_pool[self.current_index] = {}
            self.macd_pool[self.current_index]['longema'] = self.bar_pool[self.current_index][
                self.MACD_Params['analysis_field']]
            self.macd_pool[self.current_index]['shortema'] = self.bar_pool[self.current_index][
                self.MACD_Params['analysis_field']]
            self.macd_pool[self.current_index]['dif'] = self.macd_pool[self.current_index]['shortema'] - \
                                                        self.macd_pool[self.current_index]['longema']
            self.macd_pool[self.current_index]['dea'] = self.macd_pool[self.current_index]['dif']
            self.macd_pool[self.current_index]['macd'] = self.macd_pool[self.current_index]['dif'] - \
                                                         self.macd_pool[self.current_index]['dea']
            self.macd_pool[self.current_index]['macd_trending'] = 0
            self.macd_pool[self.current_index]['macd_diff'] = 0
        else:
            self.macd_pool[self.current_index] = {}
            self.__ema(alpha=self.MACD_Params['alpha_long'], _alpha=self.MACD_Params['_alpha_long'], ema_name='longema')
            self.__ema(alpha=self.MACD_Params['alpha_short'], _alpha=self.MACD_Params['_alpha_short'],
                       ema_name='shortema')
            self.macd_pool[self.current_index]['dif'] = self.macd_pool[self.current_index]['shortema'] - \
                                                        self.macd_pool[self.current_index]['longema']
            self.macd_pool[self.current_index]['dea'] = self.MACD_Params['alpha_signal'] * \
                                                        self.macd_pool[self.current_index]['dif'] + self.MACD_Params[
                                                                                                        '_alpha_signal'] * \
                                                                                                    self.macd_pool[
                                                                                                        self.previous_index][
                                                                                                        'dea']
            self.macd_pool[self.current_index]['macd'] = 2 * (self.macd_pool[self.current_index]['dif'] - \
                                                              self.macd_pool[self.current_index]['dea'])

            if self.macd_pool[self.current_index]['macd']>self.MACD_Params['macd_max_base']:
                self.MACD_Params['macd_max_base']=self.macd_pool[self.current_index]['macd']
            if self.macd_pool[self.current_index]['macd'] >= 0:
                if self.macd_pool[self.current_index]['macd'] < self.macd_pool[self.previous_index]['macd']:
                    self.macd_pool[self.current_index]['macd_trending'] = self.macd_pool[self.previous_index][
                                                                              'macd_trending'] - 1  # trend to death
                    if self.bar_pool[self.current_index]['open']>self.bar_pool[self.current_index]['close']:#down bar
                        self.macd_pool[self.current_index]['macd_trending']-=self.MACD_Params['trend_bar_unit']
                    if self.bar_unit[self.current_index]['close']<self.bar_unit[self.previous_index]['close']:
                        self.macd_pool[self.current_index]['macd_trending']-=1
                else:
                    self.macd_pool[self.current_index]['macd_trending'] = 0
            else:
                if self.macd_pool[self.current_index]['macd'] > self.macd_pool[self.previous_index]['macd']:
                    self.macd_pool[self.current_index]['macd_trending'] = self.macd_pool[self.previous_index][
                                                                              'macd_trending'] + 1  # trend to golden
                    if self.bar_pool[self.current_index]['open']<self.bar_pool[self.current_index]['close']:#up bar
                        self.macd_pool[self.current_index]['macd_trending']+=self.MACD_Params['trend_bar_unit']
                    if self.bar_unit[self.current_index]['close']<self.bar_unit[self.previous_index]['close']:
                        self.macd_pool[self.current_index]['macd_trending']+=1
                else:
                    self.macd_pool[self.current_index]['macd_trending'] = 0
            self.macd_pool[self.current_index]['macd_diff'] = self.macd_pool[self.current_index]['macd'] - \
                                                              self.macd_pool[self.previous_index]['macd']

    def set_macd(self, dif=0, dea=0, macd=0):
        self.macd_pool[self.current_index] = {}
        self.macd_pool[self.current_index]['dif'] = dif
        self.macd_pool[self.current_index]['dea'] = dea
        self.macd_pool[self.current_index]['macd'] = macd
        if self.macd_pool[self.current_index]['macd'] > self.MACD_Params['macd_max_base']:
            self.MACD_Params['macd_max_base'] = self.macd_pool[self.current_index]['macd']
        if self.macd_pool[self.current_index]['macd'] >= 0:
            if self.macd_pool[self.current_index]['macd'] < self.macd_pool[self.previous_index]['macd']:
                self.macd_pool[self.current_index]['macd_trending'] = self.macd_pool[self.previous_index][
                                                                          'macd_trending'] - 1  # trend to death
                if self.bar_pool[self.current_index]['open'] > self.bar_pool[self.current_index]['close']:  # down bar
                    self.macd_pool[self.current_index]['macd_trending'] -= self.MACD_Params['trend_bar_unit']
                if self.bar_unit[self.current_index]['close'] < self.bar_unit[self.previous_index]['close']:
                    self.macd_pool[self.current_index]['macd_trending'] -= 1
            else:
                self.macd_pool[self.current_index]['macd_trending'] = 0
        else:
            if self.macd_pool[self.current_index]['macd'] > self.macd_pool[self.previous_index]['macd']:
                self.macd_pool[self.current_index]['macd_trending'] = self.macd_pool[self.previous_index][
                                                                          'macd_trending'] + 1  # trend to golden
                if self.bar_pool[self.current_index]['open'] < self.bar_pool[self.current_index]['close']:  # up bar
                    self.macd_pool[self.current_index]['macd_trending'] += self.MACD_Params['trend_bar_unit']
                if self.bar_unit[self.current_index]['close'] < self.bar_unit[self.previous_index]['close']:
                    self.macd_pool[self.current_index]['macd_trending'] += 1
                else:
                    self.macd_pool[self.current_index]['macd_trending'] = 0
        self.macd_pool[self.current_index]['macd_diff'] = self.macd_pool[self.current_index]['macd'] - \
                                                          self.macd_pool[self.previous_index]['macd']

    def __ema(self, alpha=0.2, _alpha=0.8, ema_name='longema'):
        self.macd_pool[self.current_index][ema_name] = alpha * self.bar_pool[self.current_index][
            self.MACD_Params['analysis_field']] + _alpha * self.macd_pool[self.previous_index][ema_name]

    def check_macd_diff(self):
        if self.macd_pool[self.current_index]['macd_trending'] != 0:
            trending = self.macd_pool[self.current_index]['macd_trending']
            if trending > 0:
                macd_diff = self.macd_pool[self.current_index]['macd'] - self.macd_pool[self.current_index - trending][
                    'macd']
                sum_rate=abs(macd_diff/self.MACD_Params['macd_max_base'])
                #sum_rate = abs(macd_diff)
                if sum_rate > 0.15:
                    return 1
                if sum_rate / trending > 0.1:
                    return 1
            else:
                macd_diff = self.macd_pool[self.current_index]['macd'] - self.macd_pool[self.current_index + trending][
                    'macd']
                sum_rate = abs(macd_diff / self.MACD_Params['macd_max_base'])
                # sum_rate = abs(macd_diff)
                if sum_rate > 0.15:
                    return -1
                if abs(sum_rate / trending) > 0.1:
                    return -1
        return 0

    def check_macd(self):
        # todo:the trending relationship between dea and K-bar

        if self.macd_pool[self.previous_index]['macd'] < 0:
            if self.macd_pool[self.current_index]['macd'] >= 0:  # golden to buy
                return 1
        elif self.macd_pool[self.previous_index]['macd'] > 0:
            if self.macd_pool[self.current_index]['macd'] <= 0:  # dead to sell
                return -1

        if self.macd_pool[self.current_index]['macd_trending'] > self.MACD_Params['trend_threshold']:  # trending to golden
            return 1
        elif self.macd_pool[self.current_index]['macd_trending'] < -self.MACD_Params['trend_threshold']:  # trending to death
            return -1

        return 0  # nothing

    def macd_trending(self):
        """
        通过趋势强弱选择股票
        :return:
        """
        t1=(self.highlow_pool[self.current_index]['max_high']-self.bar_pool[self.current_index]['close'])/(self.highlow_pool[self.current_index]['max_high']-self.highlow_pool[self.current_index]['min_low'])
        t2=tanh(self.macd_pool[self.current_index]['macd_trending'])
        return t1*t2

    def __set_kdj(self):
        """
        rsv(n)=(Cn-Low)/(High-Low)*100
        K(n)=2/3*K(n-1)+1/3*rsv(n)
        D(n)=2/3*D(n-1)+1/3*K(n)
        J(n)=3*K(n)-2*D(n)
        :return:
        """
        self.kdj_pool[self.current_index] = {}
        self.__cal_high_low()
        rsv_diff = self.kdj_pool[self.current_index]['rsv_high'] - self.kdj_pool[self.current_index]['rsv_low']
        if rsv_diff == 0:
            rsv = 0
        else:
            rsv = 100.0 * (
                self.bar_pool[self.current_index]['close'] - self.kdj_pool[self.current_index]['rsv_low']) / (
                      self.kdj_pool[self.current_index]['rsv_high'] - self.kdj_pool[self.current_index]['rsv_low'])
        if self.kdj_pool[self.previous_index] is None:
            k = 50 * 2.0 / 3 + rsv / 3.0
            self.kdj_pool[self.current_index]['k'] = k  # for writing simply
            d = 50 * 2.0 / 3 + k / 3.0
            self.kdj_pool[self.current_index]['d'] = d
            self.kdj_pool[self.current_index]['j'] = 3 * k - 2 * d
        else:
            k = self.kdj_pool[self.previous_index]['k'] * 2.0 / 3 + rsv / 3.0
            self.kdj_pool[self.current_index]['k'] = k  # for writing simply
            d = self.kdj_pool[self.previous_index]['d'] * 2.0 / 3 + k / 3.0
            self.kdj_pool[self.current_index]['d'] = d
            self.kdj_pool[self.current_index]['j'] = 3 * k - 2 * d

    def __cal_high_low(self):
        """
        calculate max and min in N bars
        :return:
        """
        Cnow = self.bar_pool[self.current_index]['close']
        # if self.kdj_pool[self.previous_index]['rsv_low'] > Cnow:
        #     self.kdj_pool[self.current_index]['rsv_low'] = Cnow
        #
        #     index = self.current_index - self.KDJ_Params['period'] + 1
        #     index %= self.size
        #     if self.kdj_pool[index]['rsv_high'] < self.kdj_pool[self.previous_index]['rsv_high']:
        #         self.kdj_pool[self.current_index]['rsv_high'] = self.kdj_pool[self.previous_index]['rsv_high']
        #     else:
        #         self.kdj_pool[self.current_index]['rsv_high'] = self.kdj_pool[index]['rsv_high']
        #         index = self.__increase_index(index)
        #         i = 1
        #         while i < self.KDJ_Params['period']:
        #             if self.kdj_pool[index]['rsv_high'] > self.kdj_pool[self.current_index]['rsv_high']:
        #                 self.kdj_pool[self.current_index]['rsv_high'] = self.kdj_pool[index]['rsv_high']
        #             index = self.__increase_index(index)
        #             i += 1
        # elif self.kdj_pool[self.previous_index]['rsv_high'] < Cnow:
        #     self.kdj_pool[self.current_index]['rsv_high'] = Cnow
        #     index = self.current_index - self.KDJ_Params['period'] + 1
        #     index = self.__wraper_index(index)
        #     if self.kdj_pool[index]['rsv_low'] > self.kdj_pool[self.previous_index]['rsv_low']:
        #         self.kdj_pool[self.current_index]['rsv_low'] = self.kdj_pool[self.previous_index]['rsv_low']
        #     else:
        #         self.kdj_pool[self.current_index]['rsv_low'] = self.kdj_pool[index]['rsv_low']
        #         index = self.__increase_index(index)
        #         i = 1
        #         while i < self.KDJ_Params['period']:
        #             if self.kdj_pool[index]['rsv_low'] < self.kdj_pool[self.current_index]['rsv_low']:
        #                 self.kdj_pool[self.current_index]['rsv_low'] = self.kdj_pool[index]['rsv_low']
        #             index = self.__increase_index(index)
        #             i += 1
        # else:  # this section is more general.the code in the top is for running more efficiently
        index = self.current_index - self.KDJ_Params['period'] + 1
        index %= self.size
        self.kdj_pool[self.current_index]['rsv_low'] = Cnow
        self.kdj_pool[self.current_index]['rsv_high'] = Cnow
        i = 0

        while i < self.KDJ_Params['period']:
            if self.kdj_pool[index] is not None and self.kdj_pool[index].has_key('rsv_low'):
                if self.kdj_pool[index]['rsv_low'] < Cnow:
                    self.kdj_pool[self.current_index]['rsv_low'] = self.kdj_pool[index]['rsv_low']
                if self.kdj_pool[index]['rsv_high'] > Cnow:
                    self.kdj_pool[self.current_index]['rsv_high'] = self.kdj_pool[index]['rsv_high']
            index = self.__increase_index(index)
            i += 1

    def check_kdj(self):
        if self.kdj_pool[self.current_index]['k'] > 80 and self.kdj_pool[self.current_index]['d'] > 80:  # much more buy
            return -1
        elif self.kdj_pool[self.current_index]['k'] < 20 and self.kdj_pool[self.current_index][
            'd'] < 20:  # much more sell
            if self.kdj_pool[self.current_index]['j'] > 20:
                return 1
        else:
            if self.kdj_pool[self.previous_index]['k'] < self.kdj_pool[self.previous_index]['d']:
                if self.kdj_pool[self.current_index]['k'] >= self.kdj_pool[self.current_index]['d']:  # golden to buy
                    return 1
            if self.kdj_pool[self.previous_index]['k'] > self.kdj_pool[self.previous_index]['d']:
                if self.kdj_pool[self.current_index]['k'] <= self.kdj_pool[self.current_index]['d']:  # dead to sell
                    return -1
        return 0

    def __set_rsi(self):
        self.rsi_pool[self.current_index] = {}
        if self.bar_pool[self.previous_index] is not None:
            self.rsi_pool[self.current_index]['close_diff'] = self.bar_pool[self.current_index]['close'] - \
                                                              self.bar_pool[self.previous_index]['close']
        else:
            self.rsi_pool[self.current_index]['close_diff'] = 0
        n1_up, n1_down = self.__rsi_up_down(self.current_index - self.RSI_Params['N1'] + 1, self.current_index + 1)
        n2_up, n2_down = self.__rsi_up_down(self.current_index - self.RSI_Params['N2'] + 1,
                                            self.current_index - self.RSI_Params['N1'] + 1)
        n3_up, n3_down = self.__rsi_up_down(self.current_index - self.RSI_Params['N3'] + 1,
                                            self.current_index - self.RSI_Params['N2'] + 1)
        n1_sum = n1_up + n1_down
        if n1_sum == 0:
            self.rsi_pool[self.current_index]['rsi_n1'] = 0
        else:
            self.rsi_pool[self.current_index]['rsi_n1'] = 100.0 * n1_up / n1_sum
        n2_sum = n2_up + n2_down + n1_sum
        n2_up += n1_up
        if n2_sum == 0:
            self.rsi_pool[self.current_index]['rsi_n2'] = 0
        else:
            self.rsi_pool[self.current_index]['rsi_n2'] = 100.0 * n2_up / n2_sum
        n3_sum = n3_up + n3_down + n2_sum
        n3_up += n2_up
        if n3_sum == 0:
            self.rsi_pool[self.current_index]['rsi_n3'] = 0
        else:
            self.rsi_pool[self.current_index]['rsi_n3'] = 100.0 * n3_up / n3_sum

    def __rsi_up_down(self, start_index, end_index):
        up = 0
        down = 0
        while start_index < end_index:
            index = self.__wraper_index(start_index)
            if self.rsi_pool[index] is not None:
                if self.rsi_pool[index]['close_diff'] >= 0:
                    up += self.rsi_pool[index]['close_diff']
                else:
                    down -= self.rsi_pool[index]['close_diff']
            start_index += 1
        return up, down

    def check_rsi(self):
        # todo
        return 0

    def __set_boll(self):
        # todo
        pass

    def check_boll(self):
        # todo
        return 0

    def __set_high_low(self):
        """

        :return:
        """
        self.highlow_pool[self.current_index] = {}
        if self.highlow_pool[self.previous_index] is None:
            self.__cal_max_high_low(self.current_index)
        else:
            self.__set_low_field(bar_field='close', high_low_field='close_low', index_dest=self.current_index)
            self.__set_high_field(bar_field='close',high_low_field='close_high',index_dest=self.current_index)
            self.__set_high_field(bar_field='high', high_low_field='max_high', index_dest=self.current_index)
            self.__set_low_field(bar_field='low', high_low_field='min_low', index_dest=self.current_index)


    def __cal_max_high_low(self,index_dest):
        index = index_dest - self.HighLow_Params['period']+1
        index %= self.size
        max_high = self.bar_pool[index]['high']
        min_low = self.bar_pool[index]['low']
        close_high = self.bar_pool[index]['close']
        close_low = self.bar_pool[index]['close']
        index += 1
        for _ in range(self.HighLow_Params['period']-1):
            index %= self.size
            if close_high < self.bar_pool[index]['close']:
                close_high = self.bar_pool[index]['close']
            elif close_low > self.bar_pool[index]['close']:
                close_low = self.bar_pool[index]['close']

            if max_high < self.bar_pool[index]['high']:
                max_high = self.bar_pool[index]['high']
            if min_low > self.bar_pool[index]['low']:
                min_low = self.bar_pool[index]['low']
            index+=1
        # self.highlow_pool[index_dest] = {}
        self.highlow_pool[index_dest]['max_high'] = max_high
        self.highlow_pool[index_dest]['min_low'] = min_low
        self.highlow_pool[index_dest]['close_high'] = close_high
        self.highlow_pool[index_dest]['close_low'] = close_low
    def __cal_max_field(self,field='close',index_dest=0):
        index = index_dest - self.HighLow_Params['period']+1
        index %= self.size
        max = self.bar_pool[index][field]

        index += 1
        for _ in range(self.HighLow_Params['period']-1):
            index %= self.size
            if max<self.bar_pool[index][field]:
                max=self.bar_pool[index][field]
            index+=1
        return max

    def __cal_min_field(self, field='close', index_dest=0):
        index = index_dest - self.HighLow_Params['period']+1
        index %= self.size
        min = self.bar_pool[index][field]

        index += 1
        for _ in range(self.HighLow_Params['period']-1):
            index %= self.size
            if min > self.bar_pool[index][field]:
                min = self.bar_pool[index][field]
            index+=1
        return min

    def __set_high_field(self,bar_field='close',high_low_field='close_high',index_dest=0):
        start_index = index_dest - self.HighLow_Params['period']
        start_index%=self.size
        previous_index=(index_dest-1)%self.size
        if self.bar_pool[index_dest][bar_field] >= self.highlow_pool[previous_index][high_low_field]:
            self.highlow_pool[index_dest][high_low_field] = self.bar_pool[index_dest][bar_field]
        else:
            self.highlow_pool[index_dest][high_low_field] = self.__cal_max_field(field=bar_field,index_dest=index_dest)

    def __set_low_field(self,bar_field='close',high_low_field='close_high',index_dest=0):
        start_index = index_dest - self.HighLow_Params['period']
        start_index%=self.size
        previous_index=(index_dest-1)%self.size
        if self.bar_pool[index_dest][bar_field] <= self.highlow_pool[previous_index][high_low_field]:
            self.highlow_pool[index_dest][high_low_field] = self.bar_pool[index_dest][bar_field]
        else:
            self.highlow_pool[index_dest][high_low_field] = self.__cal_min_field(field=bar_field,index_dest=index_dest)

    def __increase_index(self, index):
        index += 1
        if index >= self.size:
            index = 0
        return index

    def __decrease_index(self, index):
        index -= 1
        if index < 0:
            index = self.size - 1
        return index

    def __wraper_index(self, index):
        index %= self.size
        return index


def sim():
    pool = cache_pool(isMACD=False, isKDJ=False, isRSI=False,isHighLow=True)
    import pandas as pd
    import numpy as np
    pool.history_k_bar(pd.DataFrame(np.random.random((30,7)),columns=['close','high','low','volume','money','paused','avg']))
    # pool2 = cache_pool(real_bar_unit=1, isMA=True, isMACD=False, isKDJ=False, isRSI=False)
    for i in range(1, 1000):
        d = get_unit_data()
        pool.add_data(d)
        # pool2.add_data(d)
        print i, pool.highlow_pool[pool.current_index]['max_high'], pool.highlow_pool[pool.current_index]['min_low']


import random


def get_unit_data():
    # {open,close,high,low,volume;money,factor,high_limit,low_limit,avg,pre_close,paused}
    ret = bee_bar()
    ret.open = random.randint(10, 20) * 1.0
    ret.close = random.randint(10, 20) * 1.0
    ret.low = min(ret.open, ret.close) - 1
    ret.high = max(ret.open, ret.close) + 1.5
    ret.volume = random.randint(10, 200)
    ret.avg = (ret.low + ret.high) / 2
    ret.money = ret.avg * ret.volume
    ret.paused = False
    return ret


if __name__ == "__main__":
    sim()
