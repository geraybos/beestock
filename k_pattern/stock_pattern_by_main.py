# -*- encoding:utf8 -*-


def pattern_analysis(data):
    if data[-1,0]==0 or data[-2,0]==0:
        return -1
    descrease = data[-2, 1] / data[-1, 0]
    yesterday_k=data[-2, 1] / data[-2, 0]
    if descrease > 1.02:  # 排除低开严重的
        return -1
    elif descrease > 1.01:
        if yesterday_k>1.065:#大阳线
            return -1
        elif yesterday_k<1:#阴线低开，空头行情
            return -1
    elif descrease>1:
        if yesterday_k < 1:  # 阴线低开，空头行情
            return -1
    elif descrease<0.985:#
        if yesterday_k>1.05:#高开诱多
            return -1
    return 1

def three_black_crows(data):
    if data[-2,0]>data[-2,1] and data[-3,1]>data[-2,1]:
        if data[-3, 0] > data[-3, 1] and data[-4,1]>data[-3,1]:
            if data[-4, 0] > data[-4, 1]:
                return -1
    return 1

def yesterday_red_today_low(data):
    """昨日大阳线,次日低开,主力诱惑散户接手[更多细节可追溯前面几天,一般来说低开价大于前几日,即主力抛售]"""
    descrease=data[-2, 2] / data[-1, 0]
    if descrease>1.05:#排除低开严重的
        return -1
    elif descrease>1.02:
        y_k=data[-2, 1] / data[-2, 0]
        if y_k> 1.065:#大阳线
            if data[-1,0]>data[-3,2] or data[-1,0]>data[-4,2]:
                return -1
        elif y_k>1.03:#小阳线,但是低开的高于之前的
            if data[-1,0]>data[-3,2] or data[-1,0]>data[-4,2]:
                return -1

    return 1

def yesterday_green_today_low(data):
    """昨日大阴线,次日低开,空头行情"""
    green_k=data[-2, 1] / data[-2, 0]#阴线
    if green_k< 0.93:#大阴线
        if data[-2, 2] > data[-1, 0]:
            return -1
    else:#小阴线且次日低开
        if data[-2, 2] / data[-1, 0]>1.015:
            return -1
    return 1


def previous_h_line(data):
    """开盘即收盘,新股开售"""
    for i in range(data.shape[0]):
        if abs(data[i, 0] - data[i, 1]) < 1e-3 and abs(data[i, 0] - data[i, 2]) < 1e-3 and abs(
                data[i, 0] - data[i, 3]) < 1e-3:
            return -1
    return 1




