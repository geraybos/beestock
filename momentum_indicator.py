#-*- coding: utf-8 -*-
import numpy as np

def moving_avg(data, period=5):
    """
    MA(N)=(C1+C2+...+Cn)/N
    :param data:
    :param period:
    :return:
    """
    ret = []
    if len(data) < period:
        ret.append(np.average(data))
    else:
        i = 0

        while i + period <= len(data):
            ret.append(np.average(data[i:i + period]))
            i += 1
    return ret


def moving_avg_last_weight(data, period=5):
    """
    MA(N)=(C1+C2+...+Cn*2)/(N+1)
    :param data:
    :param period:
    :return:list
    """
    ret = []
    if len(data) < period:
        data1 = np.append(data, data[-1])
        ret.append(np.average(data1))
    else:
        i = 0
        N = period + 1.0
        argsum = np.sum(data[i:i + period])
        ret.append((argsum + data[i + period - 1]) / N)
        i += 1
        while i + period <= len(data):
            argsum -= data[i - 1]
            argsum += data[i + period - 1]
            ret.append((argsum + data[i + period - 1]) / N)
            i += 1
    return ret


def moving_avg_linear_weight(data, period=5):
    """
    MA(N)=(C1*1+C2*2+...+Cn*n)/(1+2+...+n)
    :param data:
    :param period:
    :return:
    """
    ret = []
    if len(data) < period:
        ret.append(np.average(data, weights=np.arange(1, len(data) + 1)))
    else:
        i = 0
        weights = np.arange(1, period + 1)
        while i + period <= len(data):
            ret.append(np.average(data[i:i + period], weights=weights))
            i += 1
    return ret


def moving_avg_trapezium_weight(data, period=5):
    """
    MA(N)=[(C1+C2)*1+(C2+C3)*2+(C3+C4)*3+...(Cn-+Cn)*(n-1)]/(2*(1+2+...+(n-1)))
    :param data:
    :param period:
    :return:
    """
    ret = []
    if len(data) < period:
        ret.append(np.average(data, weights=np.arange(1, len(data) + 1)))  # todo
    else:
        i = 0
        N = period * (period - 1)
        while i + period <= len(data):
            sum = 0.0
            j = 0
            while j < period - 1:
                sum += (data[i + j] + data[i + j + 1]) * (j + 1)
                j += 1
            ret.append(sum / N)
            i += 1
    return ret


def moving_avg_square_weight(data, period=5):
    """
    MA(N)=(C1*1*1+C2*2*2+...+Cn*n*n)/(seq(1)+seq(2)+...+seq(n))
    :param data:
    :param period:
    :return:
    """
    ret = []
    if len(data) < period:
        ret.append(np.average(data, weights=np.arange(1, len(data) + 1)))  # todo
    else:
        weights = []
        for i in range(1, period + 1):
            weights.append(pow(i, 2))
        i = 0
        while i + period <= len(data):
            ret.append(np.average(data[i:i + period], weights=weights))
            i += 1
    return ret


def EMA2(data, period=5, alpha=2):
    """
    alpha/=period
    EMA(N)=(p1+(1-alpha)*p2+...+((1-alpha)^(N-1))*pn)/(1+(1-alpha)+(1-alpha)^2+...+(1-alpha)^(N-1))
    :param data:
    :param period:
    :param alpha:
    :return:
    """
    ret = []
    alpha /= (alpha - 1.0 + period)
    _alpha = 1 - alpha
    data_pre = np.array([data[0]] * (period - 1))
    data = np.concatenate((data_pre, data))

    weights = []
    for i in range(0, period):
        weights.append(pow(_alpha, i))
    weights = weights[::-1]
    i = 0
    while i + period <= len(data):
        ret.append(np.average(data[i:i + period], weights=weights))
        i += 1
    return ret


def EMA(data, period=5, alpha=2):
    """
    alpha/=period
    EMA(N)=alpha*Cn+(1-alpha)*EMA(N-1)
    EMA(1)=C1 or EMA(1)=avg(C1+...+Cn);where we used the first one
    :param data:
    :param period:
    :param alpha:
    :return:
    """
    ema = data[0]
    ret = []
    ret.append(ema)

    i = 1
    alpha /= (alpha - 1.0 + period)
    _alpha = 1 - alpha
    while i < len(data):
        ema = data[i] * alpha + _alpha * ema
        ret.append(ema)
        i += 1
    return ret


def macd(data, longperiod=26, shortperiod=12, signalperiod=9, alpha=2):
    """

    :param data:
    :param longperiod:
    :param shortperiod:
    :param signalperiod:
    :param alpha:
    :return: dif,dea,macd.dif is the fast line,dea is the slow line and macd is the bar
    """
    ema_long = np.frombuffer(EMA(data, period=longperiod, alpha=alpha), type=np.double)
    ema_short = np.frombuffer(EMA(data, period=shortperiod, alpha=alpha), type=np.double)
    dif = ema_short - ema_long
    dea = np.frombuffer(EMA(dif, period=signalperiod, alpha=alpha), type=np.double)
    #macd = (dif - dea)*2
    macd = dif - dea #macd is just for judging symbol,we don`t need to times 2
    return dif, dea, macd
