#-*- coding: utf-8 -*-
import pandas as pd

class K_Bar(object):
    """
    red means up and green means down
    """
    def __init__(self,bar_series):
        self._data=bar_series

    def check(self):
        """

        :return: [-1,1]区间,正数表示阳线,负数表示阴线
        """
        if self._data['high'] == self._data['low']:
            return 0
        l = self._data['high'] - self._data['low']
        l2 = self._data['close'] - self._data['open']
        return l2 / l
def check_k_bar(bar):
    """

    :param bar: 单条柱形数据，series格式的数据
    :return: [-1,1]区间,正数表示阳线,负数表示阴线,数值大小代表趋势强弱
    """
    if bar['high'] == bar['low']:
        return 0
    l = bar['high'] - bar['low']
    l2 = bar['close'] - bar['open']
    return l2 / l

def cdl2_crows(data):
    """
    mode:rgg三日K线模式，第一天长阳，第二天高开收阴，第三天再次高开继续收阴， 收盘比前一日收盘价低，预示股价下跌
    :param data: len(data)>=3
    :return:
    """
    pass

def cdl2_mode_check(data):
    pass