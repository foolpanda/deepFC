#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
#Full Connected  Neural Networks
class NNLayer:
    def __init__(self, shape):
        self.shape = shape
        self.w= np.random.random([shape[1],shape[0]])
        self.b=np.zeros((shape[1],1))
    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
    @staticmethod
    def sigmoid_primez(a):
            return a * (1-a)
    def forward(self,inputx):
        self.z  = np.dot(self.w,inputx)+self.b
        self.a  = NNLayer.sigmoid(self.z)
        self.Dl = NNLayer.sigmoid_primez(self.a)
        return self.a;

class FullCon:
    def __init__(self, layers_dim):
        self.L = len(layers_dim)
        self.layers={}   #注意此处用的是dict,非数组list
        for i in range(1,self.L):
            layer=NNLayer((layers_dim[i-1],layers_dim[i]))
            self.layers[i+1]=layer

    def forward(self,x):
        inputa=x
        for i in range(2,self.L):
            layer=self.layers[i]
            a_temp =layer.forward(inputa)
            inputa=a_temp
        layer=self.layers[self.L]
        layer.z=np.dot(layer.w,inputa)+layer.b
        layer.a=layer.z
        return layer.a

    def backward(self,al,y):
        m = y.shape[1]
        # 假设最后一层不经历激活函数
        # 就是按照上面的图片中的公式写的
        #grades["dz"+str(layers)] = al - y
        #DL=sigmoid_primez(caches["z"][-1])

        lys=self.layers
        layer=lys[self.L]
        error=(al - y)
        layer.delta=error
        for i in  reversed(range(2,self.L)):
            layer=lys[i]
            layer.delta=layer.Dl*(np.dot(lys[i+1].w.T,lys[i+1].delta))

    # 就是把其所有的权重以及偏执都更新一下
    def update_wandb(self,beta,inx):
         lys=self.layers
         pre_a=inx
         for i in range(2,self.L+1):
             layer=lys[i]
             delta=layer.delta
             layer.w=layer.w-beta*np.dot(delta,pre_a.T)
             layer.b=layer.b-beta*delta
             pre_a=lys[i].a

    # 计算误差值
    @staticmethod
    def compute_loss(al,y):
        return np.mean(np.square(al-y))

# 加载数据
def load_data():
    x = np.arange(0.0,1.0,0.01)
    yx =20* np.sin(2*np.pi*x)
    y =20* np.sin(2*np.pi*x)+(np.random.random(1)-0.5)*3;
    # 数据可视化
    plt.plot(x,yx)
    return x,y
#进行测试
m=100;
np.random.seed(300)
x,y = load_data()
x = x.reshape(1,m)
y = y.reshape(1,m)
plt.scatter(x,y,c='r')
fnn=FullCon([1,25,10,1])
al = 0
los=1.0
learning_rate= 0.03
#losmin= 0.001
losmin = 0.0001
learnte=0.03*0.8/10
for i in range(1640000):
    k=np.random.randint(m)
    inx=np.array([x.T[k].T]);
    outy=np.array([y.T[k].T])
    al = fnn.forward(inx)
    fnn.backward( al, outy)
    fnn.update_wandb(learning_rate,inx)
    if los <losmin:
        break;
    if i %1000 ==0:
        aly = fnn.forward(x)
        los=fnn.compute_loss(aly, y)
        print("i=%d los=%f"%(i,los))
    if i==4000:
        learnte=0.1
aly = fnn.forward(x)
plt.scatter(x,aly,c='y',marker='v')
plt.show()
