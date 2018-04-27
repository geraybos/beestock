# -*- encoding:utf8 -*-
from keras.models import Sequential,Model
from keras.layers import Dense, GRU,merge,Input,Lambda,Layer,Merge
from keras.optimizers import RMSprop
from keras.models import load_model
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class stock_model(object):
    def __init__(self, industry_name='industry_name', source_dir=None, max_file_count=None, seq_dim=5, input_dim=5,
                 out_dim=2):
        self.industry_name = industry_name
        self.model_path = source_dir + '/model'
        self.source_dir = source_dir
        self.max_file_count = max_file_count
        self.seq_dim = seq_dim
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(GRU(self.seq_dim * 2, input_shape=(self.seq_dim, self.input_dim)))
        self.model.add(Dense(self.seq_dim * 2))
        self.model.add(Dense(self.seq_dim))
        self.model.add(Dense(self.out_dim))

        self.model.compile(optimizer=RMSprop(0.0005), loss='mse', metrics=['accuracy'])

    def train(self, batch_size=10, epochs=20):
        X, Y = load_data(self.source_dir, self.seq_dim, self.max_file_count)
        X, Y, _, _ = split_dataset(1, X, Y, batch_size)
        self.model.fit(X, Y, batch_size=batch_size, epochs=epochs)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model.save(self.model_path + '/model_' + self.industry_name + '_in_' + str(self.input_dim) + '_seq_' + str(
            self.seq_dim) + '_out_' + str(self.out_dim) + '_filecount_' + str(self.max_file_count) + '.h5')

    def load_model(self):
        model_file = self.model_path + '/model_' + self.industry_name + '_in_' + str(self.input_dim) + '_seq_' + str(
            self.seq_dim) + '_out_' + str(self.out_dim) + '_filecount_' + str(self.max_file_count) + '.h5'

        self.model = load_model(model_file)

    def predict(self, X, batch_size=32, verbose=0):
        return self.model.predict(x=X, batch_size=batch_size, verbose=verbose)




class stock_model_dev(object):
    def __init__(self):
        self.model=Sequential()


def load_data(path, seq_len=5, max_size=10):
    """

    :param path:
    :param seq_len:
    :param max_size: the count of source file that allowed to train
    :return:
    """
    data_x = []
    data_y = []
    if os.path.isfile(path):
        raw_data = pd.read_csv(path)

        # raw_data = filter_data(raw_data)

        for i in range(raw_data.shape[0] - seq_len - 1):
            ret = format_time_series_data(raw_data[i:i + seq_len + 1])
            data_x.append(ret[0:-1, :])
            data_y.append(ret[-1, 2:4])
    else:
        dirs = os.listdir(path)
        file_count = 0
        sum_file_count = len(dirs)
        if max_file_count is not None:
            sum_file_count = max_file_count
        for name in dirs:
            name = path + '/' + name
            if os.path.isfile(name):
                raw_data = pd.read_csv(name)

                for i in range(raw_data.shape[0] - seq_len - 1):
                    ret = format_time_series_data(raw_data[i:i + seq_len + 1])
                    data_x.append(ret[0:-1, :])
                    data_y.append(ret[-1, 2:4])
                print('load file:%s\n%d/%d' % (name, file_count, sum_file_count))
            file_count += 1
            if max_size is not None and file_count > max_size:
                break
    return data_x, data_y


def split_dataset(train_rate=0.9, data_x=None, data_y=None, batch_size=10):
    split_index = int(train_rate * len(data_x))
    split_index //= batch_size
    split_index *= batch_size

    train_x = np.array(data_x[0:split_index])
    train_y = np.array(data_y[0:split_index])

    test_x = np.array(data_x[split_index:])
    test_y = np.array(data_y[split_index:])

    return train_x, train_y, test_x, test_y


def filter_data(data):
    """
    filter dataitem when volume=0
    :param data:
    :return:
    """
    data = data[data['volume'] > 1e-3]
    return data


def format_time_series_data(raw_data_unit):
    """
    open,close,high,low,volume
    :param raw_data:
    :return:
    """
    start_low = raw_data_unit['low'][raw_data_unit.index[0]]
    start_volume = raw_data_unit['volume'][raw_data_unit.index[0]]
    data_0 = raw_data_unit[['open', 'close', 'high', 'low']]
    data_1 = raw_data_unit[['volume']]
    data_0 = data_0.apply(lambda x: x / start_low)
    data_1 = data_1.apply(lambda x: x / start_volume)  #
    data = np.stack((data_0['open'].tolist(), data_0['close'].tolist(), data_0['high'].tolist(), data_0['low'].tolist(),
                     data_1['volume'].tolist()), axis=1)
    return data


def format_n_max_min(data=None, n=3):
    """
    calculate the max and min value in the next n unit
    :param data:
    :param n:
    :return:
    """
    pass


def predict_simulation_next_2d(model, data):
    """
    calculate the 2nd_high-1st_low
    :param model:
    :param data:
    :return:
    """
    data = format_time_series_data(data)
    input = np.array([data])
    y = model.predict(input)

    # open_=data[-1,0]*y[0,0]/data[-1,2]
    open_ = data[-1, 1]
    close_=1
    if y[0, 0] > data[-1, 2]:
        close_ = data[-1, 1] + abs(data[-1, 1] - data[-1, 0])
    else:
        close_ = data[-1, 1] - abs(data[-1, 1] - data[-1, 0])
    volume_ = data[-1, 4]
    data = np.append(data, [[open_, close_, y[0, 0], y[0, 1], volume_]], axis=0)

    #-----------
    y /= data[1, 3]
    data[1:,0:4]/=data[1,3]
    data[1:,4]/=data[1,4]

    #-------------
    y2 = model.predict(np.array([data[1:]]))
    ret_trend=-10
    if y[0,1]/data[-1,1]<0.85:
        ret_trend=-1
    else:
        ret_trend=(y2[0, 0]-y[0,1])/y[0,1]
    return ret_trend, y2[0, 0], y[0, 1]#+y2[0,1] - y[0, 0]


def hope(data_dict, industry_name='jisuanji', max_file_count=None, seq_dim=5, input_dim=5, out_dim=2):
    model = stock_model(industry_name=industry_name, source_dir='data/industry_sw/' + industry_name,
                        max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model.load_model()
    date_indexs = data_dict.values()[0]['date']
    date_indexs = date_indexs[1500:-(seq_dim + 2)]
    rs_flag = []
    rs_income = []
    L0=[]
    L1 = []
    L2 = []
    L3 = []
    rs_train_data=[]
    for date_index in date_indexs:
        # print date_index
        max_delta = -10
        max_high = -10
        min_low = -10
        select_code = ''
        raw_stock_data = None
        for code in data_dict.iterkeys():
            stock_data = data_dict[code]
            stock_data = stock_data[stock_data.date >= date_index].head(seq_dim + 2)
            if stock_data.empty or stock_data.ix[stock_data.index[0],'date']!=date_index or len(stock_data) < 7:
                continue
            delta, high, low = predict_simulation_next_2d(model, stock_data[:-2])
            if delta > max_delta:
                max_delta = delta
                max_high = high
                min_low = low
                select_code = code
                raw_stock_data = stock_data

        # min_low=raw_stock_data.ix[raw_stock_data.index[seq_dim-1], 'close'] / raw_stock_data.ix[
        #         raw_stock_data.index[1], 'low']
        # min_low-=0.012
        # max_high-=0.05
        # max_high=min_low*(1+max_delta/5)
        if max_delta > 0:

            min_low_=raw_stock_data.ix[raw_stock_data.index[seq_dim], 'low'] / raw_stock_data.ix[
                raw_stock_data.index[1], 'low']
            max_high_=raw_stock_data.ix[raw_stock_data.index[seq_dim + 1], 'high'] / raw_stock_data.ix[
                    raw_stock_data.index[1], 'low']
            close_0=raw_stock_data.ix[raw_stock_data.index[seq_dim-1], 'close'] / raw_stock_data.ix[
                raw_stock_data.index[1], 'low']
            open_1=raw_stock_data.ix[raw_stock_data.index[seq_dim], 'open'] / raw_stock_data.ix[
                raw_stock_data.index[1], 'low']
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
            flag=0
            if min_low_>min_low:
                flag+=1
                rs_income.append(0)
            if max_high_ < max_high:  # can`t sell
                flag+=10
                rs_income.append(0)
            if flag==0:
                rs_income.append(max_high - min_low)
            rs_train_data.append([max_high_, max_high, min_low,close_0,open_1, min_low_, (max_high - min_low),(max_high_ - min_low_)])
            print(date_index, select_code, max_high_, max_high, min_low,close_0,open_1, min_low_, (max_high - min_low),(max_high_ - min_low_), flag)
    print(sum(rs_income))
    # plt.plot(rs_flag, 'r*')
    # plt.plot(rs_income, 'g*')
    plt.plot(L0,'r*')
    plt.plot(L1, 'g*')
    plt.plot(L2, 'b*')
    plt.plot(L3, 'y*')
    plt.show()
    np.savetxt('data/result/rs_train_data.txt',np.array(rs_train_data),delimiter=',')


def select_best_stock(start_date='', industry_name='jisuanji', max_file_count=None, seq_dim=5, input_dim=5, out_dim=2):
    stocks=pd.read_csv('data/resource/all_industry_stock_from_sw.csv',encoding='utf8')
    stocks=stocks[~(stocks.name.str.startswith('ST') | stocks.name.str.startswith('*ST') | stocks.name.str.startswith(u'退'))]
    stocks=stocks['code']
    stocks=stocks.drop_duplicates()

    model = stock_model(industry_name=industry_name, source_dir='data/industry_sw/' + industry_name,
                        max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model.load_model()
    import tushare as ts
    select_stocks=pd.DataFrame(None,columns=['code','delta','high','low'])
    i=0
    for code in stocks:
        raw_data=ts.get_k_data(code,start=start_date)
        if len(raw_data)==5:
            delta,high,low=predict_simulation_next_2d(model,raw_data)
            buy=raw_data.ix[raw_data.index[-1],'close']*0.988
            sell=buy*1.03
            select_stocks=select_stocks.append(pd.DataFrame([[code,delta,high,low]],columns=['code','delta','high','low']),True)
        if i%100==0:
            print i,len(stocks)
        i+=1
    select_stocks=select_stocks.sort_values(by='delta',ascending=False)
    select_stocks.to_csv('data/result/hope_stock.csv')
    print select_stocks[0:30]



def show(y_list, color=['r*', 'g*']):
    plt.subplot(211)
    for i in range(len(y_list)):
        plt.plot(y_list[i][:, 0], color[i])
    plt.subplot(212)
    for i in range(len(y_list)):
        plt.plot(y_list[i][:, 1], color[i])
    plt.show()


def run(industry_name='jisuanji', source_dir='data/industry_sw/', max_file_count=10, seq_dim=5, input_dim=5, out_dim=2):
    model = stock_model(industry_name=industry_name, source_dir=source_dir + industry_name,
                        max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model_file = model.model_path + '/model_' + model.industry_name + '_in_' + str(model.input_dim) + '_seq_' + str(
        model.seq_dim) + '_out_' + str(model.out_dim) + '_filecount_' + str(model.max_file_count) + '.h5'
    if os.path.exists(model_file):
        print('model has existed,loaded to training...')
        model.load_model()
    else:
        model.build_model()

    model.train(epochs=3)


def model_test(industry_name='jisuanji', source_dir='data/industry_sw/', max_file_count=10, seq_dim=5, input_dim=5, out_dim=2):
    model = stock_model(industry_name=industry_name, source_dir=source_dir + industry_name,
                        max_file_count=max_file_count, seq_dim=seq_dim, input_dim=input_dim, out_dim=out_dim)
    model.load_model()
    x, y = load_data(path=model.source_dir + '/validation/600701.csv',seq_len=seq_dim)  # data/industry_sw/gangtie/test
    x = np.array(x)
    y = np.array(y)
    y_ = model.predict(x)
    show([y[-200:], y_[-200:]])

    # y__=[]
    # y__.append(y_[0])
    # for i in range(len(y_)-1):
    #     x_item=x[i+1]
    #     high_=y_[i,0]/x[i+1,0,3]
    #     low_=y_[i,1]/x[i+1,0,3]
    #
    #     open_=x_item[-2,1]#*high_/x_item[-2,2]
    #     if y_[i,0]>x[i+1,-1,3]:
    #         close_ = x_item[-2, 1] +abs(x_item[-2, 1]-x_item[-2, 0])
    #     else:
    #         close_ = x_item[-2, 1] - abs(x_item[-2, 1] - x_item[-2, 0])
    #     volume_=x_item[-2, 4]
    #     x_item[-1]=[open_,close_,high_,low_,volume_]
    #     # x_item=np.append(x_item,[[open_,close_,high_,low_,volume_]],axis=0)
    #     y_item=model.predict(np.array([x_item]))
    #     y__.append(y_item[0])
    # y__=np.array(y__)
    # show([y[0:200], y__[0:200]],color=['r*','g*','b*'])#,y_[0:200]

    # file_list=os.listdir(model.source_dir)
    # ret=[]
    # for f in file_list:
    #     if os.path.isfile(model.source_dir+'/'+f):
    #         print f
    #
    #         data=pd.read_csv(model.source_dir+'/'+f)#model.source_dir +'data/industry_sw/huagong/600230.csv'
    #         x=format_time_series_data(data.tail(5))
    #         y=model.predict(np.array([x]))
    #         ret.append([f,y[0,0]-y[0,1]])
    #         # print((data.iloc[-5,5],y))
    #         # print y*data.iloc[-5,5]
    # ret=sorted(ret,lambda x,y:cmp(x[1],y[1]))
    # print ret


def read_data(dirs):
    data_dict = {}
    for dir in dirs:
        for f in os.listdir(dir):  # 'data/industry_sw/jisuanji'
            if os.path.isfile(dir + '/' + f):
                data_dict[f[0:6]] = pd.read_csv(dir + '/' + f)

    hope(data_dict)


if __name__ == '__main__':
    source_dir = 'data/nasdaq/'
    industry = 'all'
    max_file_count = 1
    seq_dim = 20
    input_dim = 5
    out_dim = 2
    run(industry_name=industry,source_dir=source_dir,max_file_count=max_file_count,seq_dim=seq_dim,input_dim=input_dim,out_dim=out_dim)
    # model_test(industry_name=industry,source_dir=source_dir,max_file_count=max_file_count,seq_dim=seq_dim,input_dim=input_dim,out_dim=out_dim)
    # from data_collect import stock_data_downloader
    #
    # source_dirs=[]
    # for code in stock_data_downloader.stock_code_dict_sw.itervalues():
    #    source_dirs.append('data/industry_sw/'+code)
    # read_data(source_dirs[20:40])


    # select_best_stock(start_date='2017-11-01')


