# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime

def test():
    name_list = ['Monday', 'Tuesday', 'Friday', 'Sunday']
    num_list = [1.5, 0.6, 7.8, 6]
    num_list1 = [1, 2, 3, 1]
    x = list(range(len(num_list)))
    total_width, n = 0.8, 4
    width = total_width / n

    plt.bar(x, num_list, width=width, label='boy', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label='girl', tick_label=name_list, fc='r')
    plt.legend()
    plt.draw()
    plt.show()

def sell_rule(data):
    """

    :param data: open,high,remmend,current
    :return:
    """
    if data[3]<data[2]:
        if data[1]*0.97>data[3]:#3%的止损
            return 'r'
    else:
        if data[1]*0.99>data[3]:#1%的下降趋势
            return 'r'
    return 'w'

def show(data,code_list,fig):
    plt.cla()

    n=len(code_list)
    x=list(range(n))
    total_width = 0.8
    width = total_width / (n+2)

    labels=['open','high','recommend','sell','current']
    colors=['k','r','g','b','y','c','m']

    # fig=plt.figure()
    column_size=4
    if n<=column_size:
        column_size=n

    row_size = n // column_size +1
    for i in x:
        yx=range(len(data[i]))

        subplot=fig.add_subplot(row_size,column_size,i+1)
        subplot.set_title('stock code:'+code_list[i])
        bar=plt.bar(yx,data[i],bottom=float(min(data[i])),color=colors)

        color_flag=sell_rule(data[i])
        autolabel(bar,color_flag)
        plt.sca(subplot)


    # for i in x:
    #     start_point=i+width
    #     yx=[]
    #     for j in range(len(data[0])):
    #         yx.append(start_point)
    #         start_point+=width
    #     bar=plt.bar(yx,data[i],width=width,color=colors,label=labels,tick_label=code_list[i])
    #     autolabel(bar)

    # j=0
    # for j in range(len(data[0])-1):
    #     for i in range(n):
    #         x[i]+=width
    #     bar=plt.bar(x,data[:,j],width=width,label=labels[j],fc=colors[j])
    #
    #     autolabel(bar)
    # j+=1
    # for i in range(n):
    #     x[i] += width
    # bar = plt.bar(x, data[:, j], width=width, label=labels[j], fc=colors[j], tick_label=code_list)
    # autolabel(bar)

    # plt.legend()
    # plt.draw()

def monitor(code_list,fig):
    stock_selected=pd.read_csv('../data/result/hope_stock_'+datetime.now().strftime('%Y-%m-%d')+'.csv')
    data=[]
    real_datas=ts.get_realtime_quotes(code_list)
    for code in code_list:
        stock_item=stock_selected[stock_selected['code']==code]
        # real_data=ts.get_realtime_quotes(code)
        # data.append([real_data['high'][0],format(stock_item['today_high'].iat[0],'.2f'),float(real_data['high'][0])*0.99,real_data['price'][0]])
        real_data=real_datas[real_datas['code']==code]
        data.append(
            [float(real_data['open'].iat[0]), float(real_data['high'].iat[0]), float(format(stock_item['today_high'].iat[0], '.2f')), float(real_data['price'].iat[0])])

    show(np.array(data),code_list,fig)

def autolabel(rects,color_flag='w'):
    """

    :param rects:
    :param color_flag: red means to sell,white means to wait
    :return:
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x(), 1.05*height, '%s' % float(height),bbox=dict(facecolor=color_flag, alpha=0.5))
if __name__=='__main__':
    code_list=['600536','000677']#,'300520','000830','603619','603619','300520','000830','603619'
    plt.ion()

    fig=plt.figure()
    plt.subplots_adjust(hspace=0.5)
    while True:
        # print('k')
        monitor(code_list,fig)
        plt.pause(1)
