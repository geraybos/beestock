#-*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
import pandas as pd

class bee_stock_rnn():
    def __init__(self,input_dim=2,seq_dim=1,hidden_dim=10,output_dim=3,data_set={}):
        self.input_dim=input_dim
        self.seq_dim=seq_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim

        self.batch_size=20
        self.train_set_index=seq_dim

        self.train_set={}
        self.test_set = {}
        split_index=int(len(data_set['x'])*0.9)
        self.train_set['x']=data_set['x'][0:split_index]
        self.train_set['y'] = data_set['y'][0:split_index]

        self.train_batch=[]
        self.generate_train_batch()

        self.test_set['x'] = data_set['x'][split_index:]
        self.test_set['y'] = data_set['y'][split_index:]
        self.test_set_format()

        self.w_out=tf.Variable(tf.random_normal([hidden_dim,output_dim],1,1),name='w_out')
        self.b_out=tf.Variable(tf.random_normal([output_dim]),name='b_out')

        self.X=tf.placeholder(tf.float32,[None,seq_dim,input_dim])
        self.Y=tf.placeholder(tf.float32,[None,output_dim])

        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.model()))
        #self.cost = tf.reduce_mean(tf.square(self.Y-self.model()))
        self.train_step=tf.train.AdamOptimizer(0.1).minimize(self.cost)

        self.saver=tf.train.Saver()

        self.sess=tf.Session()


    def model(self):
        cell=rnn_cell.BasicLSTMCell(self.hidden_dim,forget_bias=0.5)
        outputs,states=rnn.dynamic_rnn(cell,tf.transpose(self.X,[1,0,2]),dtype=tf.float32,time_major=True)

        #outputs=tf.reshape(outputs,[self.seq_dim,self.batch_size,self.hidden_dim])
        return tf.sigmoid(tf.matmul(outputs[-1],self.w_out)+self.b_out)

    def train(self):
        train_step=1000
        tf.get_variable_scope().reuse_variables()
        self.sess.run(tf.global_variables_initializer())
        batch_data = self.next_train_batch()
        batch_data_t = self.next_train_batch()
        for i in range(train_step):

            # for j in range(10000):
            cost,_=self.sess.run([self.cost,self.train_step],feed_dict={self.X: batch_data['x'], self.Y: batch_data['y']})
            if i%100==0:
                print(i,cost)
                correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.model(), 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #print("accuracy:",self.sess.run([accuracy], feed_dict={self.X: self.test_set['x'],self.Y: self.test_set['y']}))
                print("accuracy:",self.sess.run([accuracy], feed_dict={self.X: batch_data['x'], self.Y: batch_data['y']}))
                #print("W:",self.sess.run([self.w_out,self.b_out]))


    def generate_train_batch(self):
        X=[]
        Y=[]
        size=0
        for index in range(self.seq_dim,len(self.train_set['x'])):
            X.append(self.train_set['x'][index - self.seq_dim:index])
            Y.append(self.train_set['y'][index - 1])
            size+=1
            if size==self.batch_size:
                size=0
                batch_item={}
                batch_item['x']=X
                batch_item['y']=Y
                self.train_batch.append(batch_item)
                X = []
                Y = []
    def next_train_batch(self):
        self.train_set_index+=1
        if self.train_set_index>=len(self.train_batch):
            self.train_set_index=0
        return self.train_batch[self.train_set_index]
    def test_set_format(self):
        X = []
        Y = []
        for i in range(len(self.test_set['x'])-self.seq_dim):
            X.append(self.test_set['x'][i:i+self.seq_dim])
            Y.append(self.test_set['y'][i+1])
        self.test_set['x']=X
        self.test_set['y']=Y
    def predict(self,x):
        return self.sess.run(self.model(),feed_dict={self.X:x})

    def close_session(self):
        self.sess.close()

def format_data(data):
    pass
def demo():
    import numpy as np
    data=np.random.rand(10000,2)
    # data = pd.read_csv("data/df.csv")
    # data=np.stack((data['close'].tolist(),data['volume'].tolist()),axis=1)


    data_set={}
    data_set['x']=data[:len(data)-1]
    y=[]
    for i in range(len(data)-1):

        if data[i][0]<data[i+1][0]:
            y.append([0,0,1])
        elif data[i][0]>data[i+1][0]:
            y.append([1,0,0])
        else:
            y.append([0,1,0])

    data_set['y']=y
    rnn=bee_stock_rnn(seq_dim=5,output_dim=3,data_set=data_set)
    rnn.train()
def demo2():
    data=pd.read_csv("data/df.csv")
    print data['close'].tolist()

analysis_fields=4
previous_bar=5
batch_size=20
L1_size=(analysis_fields*previous_bar)//2
L2_size=L1_size//2

L_out_size=3

def model_x(X):

    w_0=tf.Variable(tf.random_normal((analysis_fields*previous_bar,L1_size),1,1))
    b_0=tf.Variable(tf.zeros(L1_size))
    h1=tf.sigmoid(tf.matmul(X,w_0)+b_0)

    w_1 = tf.Variable(tf.random_normal((L1_size, L2_size), 1, 1))
    b_1 = tf.Variable(tf.zeros(L2_size))
    h2=tf.sigmoid(tf.matmul(h1,w_1)+b_1)

    w_out = tf.Variable(tf.random_normal((L2_size, L_out_size), 1, 1))
    b_out = tf.Variable(tf.zeros(L_out_size))
    return tf.sigmoid(tf.matmul(h2, w_out) + b_out)



if __name__=='__main__':
    demo()