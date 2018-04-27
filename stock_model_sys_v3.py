# -*- encoding:utf8 -*-
from keras.models import Model
from keras.layers import Dense, GRU, merge, Input, Lambda, Reshape, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import os
import math
import random
import matplotlib.pyplot as plt
from k_pattern import stock_pattern_by_main
from datetime import datetime,timedelta


class stock_model_v3(object):
    def __init__(self, industry_name='industry_name', source_dir=None, max_file_count=None, seq_dim=5, input_dim=5,
                 out_dim=4):
        self.industry_name = industry_name
        self.model_path = source_dir + '/model'
        self.source_dir = source_dir
        self.max_file_count = max_file_count
        self.seq_dim = seq_dim
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.model = None

    def build_model(self):
        input = Input(shape=(self.seq_dim, self.input_dim))
        gru = GRU(self.seq_dim * 2)
        # dropout = Dropout(0.3)
        dense1 = Dense(self.seq_dim * 2)
        dense2 = Dense(self.out_dim*2,activation='sigmoid')
        dense_out = Dense(self.out_dim,activation='softmax')

        z0 = gru(input)
        # z0 = dropout(z0)
        z0 = dense1(z0)
        z0 = dense2(z0)
        y0 = dense_out(z0)  # shape=(1,2) [high,low]


        self.model = Model(inputs=input, outputs=y0)

        self.model.compile(optimizer=RMSprop(0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
        # self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, batch_size=10, epochs=20):
        X, Y = load_data(self.source_dir, self.seq_dim, self.max_file_count,self.out_dim)
        # Y=np.expand_dims(Y,1)
        Y=to_categorical(Y,num_classes=self.out_dim)
        print('load data completed!!!')
        X, Y, _, _ = split_dataset(0.1, X, Y, batch_size)
        self.model.fit(X, Y, batch_size=batch_size, epochs=epochs)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model.save(
            self.model_path + '/model_v3_' + self.industry_name + '_in_' + str(self.input_dim) + '_seq_' + str(
                self.seq_dim) + '_out_' + str(self.out_dim) + '_filecount_' + str(self.max_file_count) + '.h5')

    def load_model(self):
        model_file = self.model_path + '/model_v3_' + self.industry_name + '_in_' + str(self.input_dim) + '_seq_' + str(
            self.seq_dim) + '_out_' + str(self.out_dim) + '_filecount_' + str(self.max_file_count) + '.h5'

        self.model = load_model(model_file)

    def predict(self, X, batch_size=10, verbose=0):
        return self.model.predict(x=X, batch_size=batch_size, verbose=verbose)


def split_end_to_1(x):
    return x[:, -1]


def split_end_from_1(x):
    return x[:, 1:, :]


def split_end_to_1_shape(input_shape):
    input_shape = list(input_shape)
    return tuple(input_shape)


def split_end_from_1_shape(input_shape):
    input_shape = list(input_shape)
    input_shape[-1] -= 1
    return tuple(input_shape)


def load_data(path, seq_len=5, max_size=10,num_classes=40):
    """

    :param path:
    :param seq_len:
    :param max_size: the count of source file that allowed to train
    :return:
    """
    data_x = []
    data_y1 = []
    if os.path.isfile(path):
        raw_data = pd.read_csv(path)

        for i in range(raw_data.shape[0] - seq_len - 1):
            ret = format_time_series_data(raw_data[i:i + seq_len + 1])
            data_x.append(ret[0:-1, :])
            data_y1.append(format_hot_label(ret[-1, 2]/ret[-2,1],num_classes=num_classes))#high
    else:
        dirs = os.listdir(path)
        random.shuffle(dirs)
        file_count = 0
        sum_file_count=len(dirs)
        if max_file_count is not None:
            sum_file_count=max_file_count
        for name in dirs:
            name = path + '/' + name
            if os.path.isfile(name):
                raw_data = pd.read_csv(name)

                for i in range(raw_data.shape[0] - seq_len - 1):
                    ret = format_time_series_data(raw_data[i:i + seq_len + 1])
                    data_x.append(ret[0:-1, :])
                    data_y1.append(format_hot_label(ret[-1, 2]/ret[-2,1],num_classes=num_classes))  # high
                print('load file:%s\n%d/%d' %(name,file_count,sum_file_count))
            file_count += 1
            if max_size is not None and file_count > max_size:
                break
    return data_x, data_y1


def split_dataset(train_rate=0.9, data_x=None, data_y1=None, batch_size=10):
    split_index = int(train_rate * len(data_x))
    split_index //= batch_size
    split_index *= batch_size

    train_x = np.array(data_x[0:split_index])
    train_y1 = np.array(data_y1[0:split_index])

    test_x = np.array(data_x[split_index:])
    test_y1 = np.array(data_y1[split_index:])

    return train_x, train_y1, test_x, test_y1


def format_time_series_data(raw_data_unit,with_label=True):
    """
    open,close,high,low,volume
    :param raw_data:
    :param with_label
    :return:
    """
    if with_label:
        start_low = raw_data_unit['low'][raw_data_unit.index[0]]
        start_volume = raw_data_unit['volume'][raw_data_unit.index[0]]
    else:
        start_low = raw_data_unit['low'][raw_data_unit.index[0]]
        start_volume = raw_data_unit['volume'][raw_data_unit.index[0]]
    data_0 = raw_data_unit[['open', 'close', 'high', 'low']]
    data_1 = raw_data_unit[['volume']]
    data_0 = data_0.apply(lambda x: x / start_low)
    data_1 = data_1.apply(lambda x: x / start_volume)  #
    data = np.stack((data_0['open'].tolist(), data_0['close'].tolist(), data_0['high'].tolist(), data_0['low'].tolist(),
                     data_1['volume'].tolist()), axis=1)
    return data

def format_hot_label(y,num_classes=40,up=1.1,down=0.9):
    dx=(up-down)/num_classes
    if y<down:
        y=0
    elif y>=up:
        y=num_classes-1
    else:
        y=(y-down)//dx
    return int(y)


def sigmiod(x):
    return 1 / (1 + math.exp(-x))


def predict_simulation_next_2d(model, data):
    """
    calculate the 2nd_high-1st_low
    :param model:
    :param data:
    :return:
    """
    data = format_time_series_data(data)
    input = np.array([data[0:-1]])
    y1, y2 = model.predict(input)

    ret_trend = -10

    # ret_trend+=1
    # if data[-2,2]<data[-3,2] or data[-2,3]<data[-3,3]:
    #     ret_trend=-1
    # else:

    #
    # if abs(data[-2, 2] - data[-2, 3]) < 1e-3:
    #     alpha = -10
    # else:
    #     x = (data[-2, 1] - data[-2, 0]) / (data[-2, 2] - data[-2, 3])
    #     alpha = sigmiod(x)
    ret_trend = (y2[0, 0] - y1[0, 1]) / y1[0, 1]
    # ret_trend=ret_trend*y2[0,1]/y1[0,0]
    # ret_trend *= alpha
    # ret_trend += (alpha * 0.1) stock_pattern_by_main.three_black_crows( data) oror y1[0,1]<input[0,-1,3]

    if stock_pattern_by_main.pattern_analysis(data) < 0 or stock_pattern_by_main.previous_h_line(data)<0:
        ret_trend = -1
    # if stock_pattern_by_main.previous_h_line(data) < 0 or y1[0,1]<input[0,-1,3]:
    #     ret_trend = -1
    return ret_trend, y1[0], y2[0]

def predict_next_2d(model, data):
    """
    calculate the 2nd_high-1st_low
    :param model:
    :param data:
    :return:
    """
    data = format_time_series_data(data)
    input = np.array([data])
    y1, y2 = model.predict(input)
    ret_trend = (y2[0, 0] - y1[0, 1]) / y1[0, 1]
    if stock_pattern_by_main.previous_h_line(data) < 0 or y1[0,1]<input[0,-1,3]:
        ret_trend = -1
    return ret_trend, y1[0], y2[0]


def hope(data_dict, industry_name='jisuanji', max_file_count=None, seq_dim=5, input_dim=5, out_dim=4):
    model = stock_model_v3(industry_name=industry_name, source_dir='data/industry_sw/' + industry_name,
                           max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model.load_model()
    input_seq_size = seq_dim + 2
    date_indexs = data_dict.values()[0]['date']
    date_indexs = date_indexs[-input_seq_size-100:-input_seq_size]
    rs_flag = []
    rs_income = []
    L0 = []
    L1 = []
    L2 = []
    L3 = []
    rs_train_data = []
    select_flag_gt_high = 0
    select_flag_lt_low = 0

    for date_index in date_indexs:
        # print date_index
        max_delta = -10
        max_high = -10
        min_low = -10
        select_code = ''
        raw_stock_data = None
        fix_y1 = None
        fix_y2 = None
        for code in data_dict.iterkeys():
            # if code=='002265':
            #     print code
            stock_data = data_dict[code]
            stock_data = stock_data[stock_data.date >= date_index].head(input_seq_size)
            if stock_data.empty or stock_data.ix[stock_data.index[0], 'date'] != date_index or len(
                    stock_data) < input_seq_size:
                continue
            delta, y1, y2 = predict_simulation_next_2d(model, stock_data[:-1])
            if delta > max_delta:
                max_delta = delta
                max_high = y2[0]
                min_low = y1[1]
                fix_y1 = y1
                fix_y2 = y2
                select_code = code
                raw_stock_data = stock_data



        if max_delta > 0:
            # min_low=raw_stock_data.ix[raw_stock_data.index[seq_dim-1], 'close'] / raw_stock_data.ix[
            #         raw_stock_data.index[1], 'low']
            # min_low-=0.012
            # max_high-=0.05
            # max_high=min_low*(1+max_delta/5)

            min_low_ = raw_stock_data.ix[raw_stock_data.index[seq_dim], 'low'] / raw_stock_data.ix[
                raw_stock_data.index[0], 'low']
            max_high_ = raw_stock_data.ix[raw_stock_data.index[seq_dim + 1], 'high'] / raw_stock_data.ix[
                raw_stock_data.index[0], 'low']

            yesterday_close = raw_stock_data.ix[raw_stock_data.index[seq_dim - 1], 'close'] / raw_stock_data.ix[
                raw_stock_data.index[0], 'low']
            today_open = raw_stock_data.ix[raw_stock_data.index[seq_dim], 'open'] / raw_stock_data.ix[
                raw_stock_data.index[0], 'low']
            today_close = raw_stock_data.ix[raw_stock_data.index[seq_dim], 'close'] / raw_stock_data.ix[
                raw_stock_data.index[0], 'low']
            tomorrow_open = raw_stock_data.ix[raw_stock_data.index[seq_dim + 1], 'open'] / raw_stock_data.ix[
                raw_stock_data.index[0], 'low']
            tomorrow_close = raw_stock_data.ix[raw_stock_data.index[seq_dim + 1], 'close'] / raw_stock_data.ix[
                raw_stock_data.index[0], 'low']
            tomorrow_low = raw_stock_data.ix[raw_stock_data.index[seq_dim + 1], 'low'] / raw_stock_data.ix[
                raw_stock_data.index[0], 'low']
            fix_min_low = min_low
            fix_max_high = max_high
            dy = max_high - min_low
            # min_low=min([yesterday_close,today_open,min_low])

            if min_low > today_open:
                min_low = today_open
            else:
                for i in range(10):
                    min_low=today_open*(1-i*0.01)
                    if min_low<min_low_:
                        break
                min_low+=today_open*0.01

            if min_low > today_open:
                min_low = today_open


            # if min_low>today_open-0.02:
            #     min_low = today_open - 0.02

            max_high = min_low*1.015  # make sure to income 3%


            if max_high < tomorrow_open:#期望上涨10个点,从开盘价开始监控，如果10min之内未越级则在该级内卖出,每级0.5个点
                for i in range(20):
                    max_high = tomorrow_open + tomorrow_open * 0.005*(19-i)
                    if max_high > max_high_:
                        continue
                    else:
                        break

            L0.append(min_low_)
            L1.append(min_low)
            L2.append(max_high_)
            L3.append(max_high)
            # print(date_index, select_code, max_high_, min_low_, (max_high_ - min_low_))
            # if min_low_ > min_low:  # can`t buy
            #     rs_flag.append(1)
            #     rs_income.append(0)
            #     print(date_index, select_code,max_high_,max_high , min_low,min_low_, (max_high - min_low),(max_high_ - min_low_), 1)
            #
            # if  max_high_< max_high:  # can`t sell
            #     rs_flag.append(2)
            #     rs_income.append(0)
            #     print(date_index, select_code, max_high_, max_high, min_low, min_low_, (max_high - min_low),
            #           (max_high_ - min_low_), 2)
            # else:
            #     print(date_index, select_code, max_high_, max_high, min_low, min_low_, (max_high - min_low),
            #           (max_high_ - min_low_), 0)
            #     rs_flag.append(0)
            #     rs_income.append(max_high - min_low)
            flag = 0
            today_income = 0
            if min_low_ > min_low:  # can't buy
                flag += 1
                select_flag_lt_low += 1
            else:
                if tomorrow_low/min_low<0.98:# stop loss at 2% point
                    if tomorrow_open/min_low<0.98:
                        today_income=(tomorrow_open-min_low)/min_low
                    else:
                        today_income = -0.02
                    if max_high_<max_high:#由于止损卖出失误
                        flag+=1000
                    else:
                        flag += 100
                    select_flag_gt_high += 1
                else:
                    if max_high_ < max_high:  # can`t sell
                        flag += 10
                        today_income =-0.02 #(tomorrow_open * 0.97 - min_low) / min_low  # stop loss at 2% point
                        select_flag_gt_high += 1
                    else:
                        today_income = (max_high - min_low) / min_low
            rs_income.append(today_income)
            rs_train_data.append(
                [max_high_, max_high, min_low, yesterday_close, today_open, min_low_, (max_high - min_low),
                 (max_high_ - min_low_)])
            print(
            date_index, select_code, max_high_, max_high, min_low, yesterday_close, today_open, min_low_, today_income,
            (max_high_ - min_low_) / min_low_, flag)
            # print(
            #     date_index, select_code, fix_y1[1], fix_y1[0], fix_y2[1], fix_y2[0],
            #     today_income,max_delta,(fix_y2[0] - fix_y1[1]) / fix_y1[1],
            #     (max_high_ - min_low_) / min_low_, flag)

    my_sum_income = 1
    for income in rs_income:
        my_sum_income *= (1 + income - 0.005)
    print('total income:%f\navg income:%f\navg exp:%f ' % (
        my_sum_income, my_sum_income / len(rs_income), sum(rs_income) / len(rs_income)))
    print(
        'gt high%f  lt low%f' % (
        (select_flag_gt_high * 1.0 / len(rs_income)), (select_flag_lt_low * 1.0 / len(rs_income))))
    # plt.plot(rs_flag, 'r*')
    # plt.plot(rs_income, 'g*')
    plt.plot(L0, 'r*')
    plt.plot(L1, 'g*')
    plt.plot(L2, 'b*')
    plt.plot(L3, 'y*')
    plt.show()
    np.savetxt('data/result/rs_train_data' + '.txt', np.array(rs_train_data), delimiter=',')


def print_best_stock_at_today(start_date='', industry_name='jisuanji', max_file_count=None, seq_dim=5, input_dim=5, out_dim=2,stocks=[]):

    model = stock_model_v3(industry_name=industry_name, source_dir='data/industry_sw/' + industry_name,
                           max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model.load_model()

    import tushare as ts
    for code in stocks:
        raw_data = ts.get_k_data(code, start=start_date)
        today_open=ts.get_realtime_quotes(code)
        today_open=float(today_open.ix[today_open.index[-1],'open'])
        if today_open is None:
            continue
        if len(raw_data) == seq_dim:
            print code
            delta, y1, y2 = predict_next_2d(model, raw_data)
            base_price=raw_data.ix[raw_data.index[0], 'low']
            print('=============%s==============',code)
            print(y1*base_price)
            print(y2*base_price)

def select_best_stock_at_yestoday(start_date='', industry_name='jisuanji', max_file_count=None, seq_dim=5, input_dim=5, out_dim=2):
    stocks = pd.read_csv('data/resource/all_industry_stock_from_sw.csv', encoding='utf8')
    stocks = stocks[
        ~(stocks.name.str.startswith('ST') | stocks.name.str.startswith('*ST') | stocks.name.str.startswith(u'退'))]
    stocks = stocks['code']
    stocks = stocks.drop_duplicates()
    stocks=stocks[1:]
    model = stock_model_v3(industry_name=industry_name, source_dir='data/industry_sw/' + industry_name,
                           max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model.load_model()

    import tushare as ts
    select_stocks = pd.DataFrame(None, columns=['code', 'delta', 'high', 'low','base_price'])
    i = 0
    for code in stocks:
        # if code=='300656':
        # print code

        raw_data = ts.get_k_data(code, start=start_date)
        today_open=ts.get_realtime_quotes(code)
        today_open=float(today_open.ix[today_open.index[-1],'open'])
        if today_open is None:
            continue
        # raw_data=raw_data.append(pd.DataFrame([[today_open,today_open,today_open,today_open,today_open]],index=[-1],columns=['open','close','high','low','volume']))

        if len(raw_data) == seq_dim:
            print code
            delta, y1, y2 = predict_next_2d(model, raw_data)
            base_price=raw_data.ix[raw_data.index[0], 'low']
            select_stocks = select_stocks.append(
                pd.DataFrame([[code, delta, y2,y2*base_price, y1,y1*base_price, base_price]], columns=['code', 'delta', 'high_precent','tomorrow', 'low_precent','today','base_price']), True)
        if i % 100 == 0:
            print i, len(stocks)
        i += 1
    select_stocks = select_stocks.sort_values(by='delta', ascending=False)
    # select_stocks =select_stocks[0:100]
    select_stocks.to_csv('data/result/hope_stock_'+datetime.now().strftime('%Y-%m-%d')+'.csv')

def select_best_stock_after_open(start_date='', industry_name='jisuanji', max_file_count=None, seq_dim=5, input_dim=5, out_dim=2):
    stocks = pd.read_csv('data/result/hope_stock_'+datetime.now().strftime('%Y-%m-%d')+'.csv', encoding='utf8')

    stocks=stocks[1:300]
    import tushare as ts
    for code in stocks['code']:
        # print code

        raw_data = ts.get_k_data(code, start=start_date)
        today_open=ts.get_realtime_quotes(code)
        today_open=float(today_open.ix[today_open.index[-1],'open'])
        if today_open is None:
            continue
        raw_data=raw_data.append(pd.DataFrame([[today_open,today_open,today_open,today_open,today_open]],index=[-1],columns=['open','close','high','low','volume']))

        if len(raw_data) == seq_dim+1:
            if stock_pattern_by_main.pattern_analysis(format_time_series_data(raw_data))>0:
                print(stocks[stocks.code==code])

            # select_stocks = select_stocks.append(
            #     pd.DataFrame([[code, delta, y2,y2*base_price, y1,y1*base_price, base_price]], columns=['code', 'delta', 'high_precent','tomorrow', 'low_precent','today','base_price']), True)
            #
    # select_stocks = select_stocks.sort_values(by='delta', ascending=False)
    # select_stocks.to_csv('data/result/hope_stock_final.csv')

def show(y_list, color=['r*', 'g*']):
    plt.subplot(211)
    for i in range(len(y_list)):
        plt.plot(y_list[i], color[i])
    # plt.subplot(212)
    # for i in range(len(y_list)):
    #     plt.plot(y_list[i], color[i])
    plt.show()


def run(industry_name='jisuanji', max_file_count=10, seq_dim=5, input_dim=5, out_dim=2):
    model = stock_model_v3(industry_name=industry_name, source_dir='data/industry_sw/' + industry_name,
                           max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model.build_model()

    model.train(epochs=30)

    # x, y1,y2 = load_data(path=model.source_dir + '/test')
    # x = np.array(x)
    # y1 = np.array(y1)
    # y2=np.array(y2)
    # y_1,y_2 = model.predict(x)
    # show([y1[1:], y_1[1:],y_2],color=['r*','g*','b*'])


def model_test(industry_name='jisuanji', max_file_count=10, seq_dim=5, input_dim=5, out_dim=2):
    model = stock_model_v3(industry_name=industry_name, source_dir='data/industry_sw/' + industry_name,
                           max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model.load_model()
    x, y1 = load_data(path=model.source_dir + '/test/600701.csv',
                          seq_len=model.seq_dim)  # data/industry_sw/gangtie/test
    x = np.array(x)
    # y1 = to_categorical(y1,out_dim)

    y_1 = model.predict([x])
    y_1=np.argmax(y_1,axis=1)
    show([y1, y_1], color=['r*', 'g*', 'b*'])
    # show([y1, y_1], color=['r*', 'g*'])
    # show([y2, y_2], color=['r*', 'g*'])


def read_data(dirs, industry_name='jisuanji', max_file_count=None, seq_dim=5, input_dim=5, out_dim=4):
    data_dict = {}
    for dir in dirs:
        for f in os.listdir(dir):  # 'data/industry_sw/jisuanji'
            if os.path.isfile(dir + '/' + f):
                data_dict[f[0:6]] = pd.read_csv(dir + '/' + f)

    hope(data_dict, industry_name=industry_name, max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim,
         out_dim=out_dim)


if __name__ == '__main__':
    industry = 'all'
    max_file_count = 1
    seq_dim = 10
    input_dim = 5
    out_dim = 40
    run(industry_name=industry,max_file_count=max_file_count,seq_dim=seq_dim,input_dim=input_dim,out_dim=out_dim)
    # model_test(industry_name=industry,max_file_count=max_file_count,seq_dim=seq_dim,input_dim=input_dim,out_dim=out_dim)
    # from data_collect import stock_data_downloader
    #
    # source_dirs = []
    # for code in stock_data_downloader.stock_code_dict_sw.itervalues():
    #     source_dirs.append('data/industry_sw/' + code)
    # read_data(source_dirs[0:10], industry_name=industry, max_file_count=max_file_count, seq_dim=seq_dim,
    #           input_dim=input_dim, out_dim=out_dim)

    # start=datetime.today()-timedelta(days=14)
    # start=start.strftime('%Y-%m-%d')
    # select_best_stock_at_yestoday(start_date=start, industry_name=industry, max_file_count=max_file_count, seq_dim=seq_dim,
    #           input_dim=input_dim, out_dim=out_dim)

    # select_best_stock_after_open(start_date=start, industry_name=industry, max_file_count=max_file_count,
    #                               seq_dim=seq_dim,input_dim=input_dim, out_dim=out_dim)

    # stocks=['002901','000063','600487','002127','603305']#前一天选的股
    # print_best_stock_at_today(start_date=start, industry_name=industry, max_file_count=max_file_count,
    #                               seq_dim=seq_dim,input_dim=input_dim, out_dim=out_dim,stocks=stocks)