from keras.layers import Dense, LSTM,GRU,Reshape,Flatten,Dropout
from keras.models import Sequential,load_model
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

model=None
seq_len=4
batch_size=10
input_dim=5
output_dim=4

raw_data=None
split_index=0

def build_model():
    global model
    model = Sequential()
    # model.add(Dense(10,input_shape=(10,5,),activation='relu'))
    model.add(GRU(10,input_shape=(seq_len,input_dim),return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(GRU(5))
    model.add(Dense(seq_len*2))
    model.add(Dense(seq_len))
    model.add(Dense(output_dim))
    model.compile(optimizer=RMSprop(0.0005),
                  loss='mse',
                  metrics=['accuracy'])

def train(X,Y):
    global model
    model.fit(x=X,y=Y,batch_size=batch_size,epochs=20)
    model.save('check_point/stock_predict_at_'+str(seq_len)+'_out_'+str(output_dim)+'.h5')


def predict(X):
    model=load_model('check_point/stock_predict_at_'+str(seq_len)+'_out_'+str(output_dim)+'.h5')
    return model.predict(X)

def get_data(path):
    global raw_data
    data_x = []
    data_y = []
    if os.path.isfile(path):
        raw_data = pd.read_csv(path)

        raw_data = filter_data(raw_data)

        for i in range(raw_data.shape[0] - seq_len - 1):
            x, y = format_time_series_data(raw_data[i:i + seq_len + 1], i)
            data_x.append(x)
            data_y.append(y)
    else:
        dirs=os.listdir(path)

        for name in dirs:
            name=path+'/'+name
            if os.path.isfile(name):
                raw_data=pd.read_csv(name)

                raw_data=filter_data(raw_data)

                for i in range(raw_data.shape[0]-seq_len-1):
                    x,y=format_time_series_data(raw_data[i:i+seq_len+1],i)
                    data_x.append(x)
                    data_y.append(y)
    train_rate=0.9
    global split_index
    split_index=int(train_rate*len(data_x))
    split_index//=batch_size
    split_index*=batch_size
    end_index=len(data_x)//batch_size
    end_index*=batch_size
    train_x = np.array(data_x[0:split_index])
    train_y = np.array(data_y[0:split_index])

    test_x = np.array(data_x[split_index:])
    test_y = np.array(data_y[split_index:])

    return train_x,train_y,test_x,test_y
def filter_data(data):
    """
    filter dataitem when volume=0
    :param data:
    :return:
    """
    data = data[data['money']>1e-3]
    # data=data.reindex(index=range(data.shape[0]))
    return data

def format_time_series_data(raw_data_unit,i):
    """
    open,close,high,low,volume
    :param raw_data:
    :return:
    """
    start_low=raw_data_unit['low'][raw_data_unit.index[0]]
    start_volume=raw_data_unit['volume'][raw_data_unit.index[0]]
    data_0=raw_data_unit[['open','close','high','low']]
    data_1 = raw_data_unit[['volume']]
    data_0=data_0.apply(lambda x:x/start_low)
    data_1=data_1.apply(lambda x:x/start_volume)#
    data = np.stack((data_0['open'].tolist(),data_0['close'].tolist(),data_0['high'].tolist(),data_0['low'].tolist(),data_1['volume'].tolist()),axis=1)
    return data[0:-1,:],data[-1,0:4]#2:4


def simulation():
    """
    600230-next
    [[ 51.99218369  51.75576019  53.20196152  50.97697449]]
[[ 52.24232483  52.00437164  53.489254    51.22053909]]
[[ 52.28421021  52.13527298  53.45378494  51.25970459]]
[[ 51.99057007  52.09732437  52.66093826  50.97900391]]
[[ 52.28000259  52.26807785  53.40497208  51.62332916]]
[[ 52.23793793  52.47561264  53.17876053  51.55451965]]
[[ 52.23464203  52.54735947  53.18385696  51.58068848]]
[[ 52.24970627  52.62609863  53.21709824  51.63630295]]
[[ 52.32775879  52.72820663  53.33514786  51.76964951]]
[[ 52.33940506  52.79686356  53.31163788  51.77265549]]
600808-next
[[ 4.22570181  4.24470997  4.29227829  4.15515041]]
[[ 4.21763086  4.24302387  4.2829628   4.1473937 ]]
[[ 4.20771646  4.23757601  4.27779579  4.14218378]]
[[ 4.19581985  4.23109722  4.26386261  4.13338184]]
[[ 4.20306301  4.23422766  4.28795767  4.15236855]]
[[ 4.20277691  4.23984146  4.28636122  4.15274811]]
[[ 4.20553207  4.24336243  4.28928804  4.15762901]]
[[ 4.20965004  4.24803686  4.29365873  4.16389894]]
[[ 4.21579695  4.25406361  4.30056     4.17220974]]
[[ 4.22008896  4.25959492  4.30338478  4.17666769]]
    :return:
    """
    raw_data = pd.read_csv("data/df_600230.XSHG2.csv")
    raw_data = filter_data(raw_data)
    data = np.stack((raw_data['open'].tolist(), raw_data['close'].tolist(), raw_data['high'].tolist(), raw_data['low'].tolist()), axis=1)

    model = load_model('check_point/stock_predict_at_' + str(seq_len) + '_out_' + str(output_dim) + '.h5')
    ret=sim(data,10)
    print ret


def sim(data,steps):
    model = load_model('check_point/stock_predict_at_' + str(seq_len) + '_out_' + str(output_dim) + '.h5')
    input = data / data[0][3]
    ret = []
    for i in range(steps):
        y = model.predict(np.array([input]))
        ret.append(np.copy(y))

        y *= data[0][3]
        print y
        data = np.append(data[1:, :], np.array(y), axis=0)
        input = data / data[0][3]
    return ret

def test_dense():
    model=Sequential()
    model.add(LSTM(2,input_shape=(2,2),return_sequences=False))
    # model.add(Reshape((4,)))
    # model.add(Flatten())
    model.add(Dense(1))

    model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
    model.fit(x=np.random.random((100,2,2)),y=np.random.random((100,1)),batch_size=5,epochs=20)
if __name__=='__main__':
    # test_dense()
    train_x, train_y ,test_x, test_y = get_data('data/vildate/df_600701.XSHG_1d.csv')#/data_30m

    # build_model()
    # train(train_x,train_y)
    # predict_y=model.predict(test_x)

    predict_y=predict(test_x)

    test_x_1=np.array(test_x)

    high_index=2
    low_index=3

    y_0=[]
    y_1=[]
    # split_index=0
    # test_y=train_y
    for i in range(len(predict_y)):
        # base=raw_data['low'][raw_data.index[split_index+i]]#low
        base=1
        test_y[i][high_index]*=base
        test_y[i][low_index] *= base
        if predict_y[i][high_index]>predict_y[i][low_index]:
            y_0.append(predict_y[i][high_index]*base)
            y_1.append(predict_y[i][low_index]*base)
        else:
            y_0.append(predict_y[i][low_index]*base)
            y_1.append(predict_y[i][high_index]*base)

    # start_sim_index=split_index
    # sim_data=raw_data[split_index:split_index+seq_len]
    # sim_data = np.stack((sim_data['open'].tolist(), sim_data['close'].tolist(), sim_data['high'].tolist(), sim_data['low'].tolist()),axis=1)
    #
    start_show_index=len(y_0)-50
    end_show_index=len(y_0)
    # sim_y=np.array(sim(sim_data,end_show_index))
    # sim_y=np.squeeze(sim_y)

    plt.subplot(211)
    plt.plot(test_y[start_show_index:end_show_index,high_index],'r*')

    plt.plot(y_0[start_show_index:end_show_index], 'g*')
    # plt.plot(sim_y[:,2],'b-')
    plt.subplot(212)
    plt.plot(test_y[start_show_index:end_show_index, low_index], 'r*')
    # plt.plot(sim_y[:, 3], 'b-.')
    plt.plot(y_1[start_show_index:end_show_index], 'g*')
    plt.show()
    # simulation()