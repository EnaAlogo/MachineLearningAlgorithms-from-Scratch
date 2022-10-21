import numpy as np
from BitmapImages import bmpimgs
import matplotlib.pyplot as plt


class hopfield:
    def __init__(this,type) -> None:
        this.style(type)
        

    @staticmethod
    def __discrete(x):return -1 if x<0 else 1

    @staticmethod
    def make_weight(vector):
        result=np.ones((len(vector),len(vector)))
        for v in enumerate(vector):
           for i in range ( len(vector) ):
               result[v[0],i]=v[1]*vector[i]
        return result

    def style(this,type:str):
        if type=='async':
            this.ann=this.__r_async
        elif type=='sync':
            this.ann=this.__r_sync
        else :
            raise ValueError('not valid')

    def fit(this):
        this.img=bmpimgs()
        fi,si,ei,ni=this.img.get_perfects()
        this.w=this.make_weight(fi.reshape(77))+\
               this.make_weight(si.reshape(77))+\
               this.make_weight(ei.reshape(77))+\
                this.make_weight(ni.reshape(77))

    def test(this,data,label,max_itters=10):
        for x,y in zip(data,label):
            this.ann(x,y,max_itters)

    def __r_sync(this,data,label ,max_itters):
        col, row = np.shape(this.w)
        output = np.zeros(col)
        temp = data
        for i in range(1, max_itters+1):
            for k in range(col):
                if (this.w[k,].any()!=0.0):
                    temp[k] = this.__discrete(np.matmul(this.w[k,],temp))
                    plt.cla()
                    plt.imshow(temp.reshape(11,7),cmap=plt.cm.gray)
                    plt.title('class : {}'.format(this.img.className(label)))
                    plt.pause(.005)
            if (temp==output).all():
                break
            output = temp
            return output,i

    def __r_async(this,data,label ,max_itters):
        col, row = np.shape(this.w)
        output = np.zeros(col)
        temp = data
        for i in range(1, max_itters+1):
            for k in np.random.permutation(col):
                if (this.w[k,].any()!=0.0):
                    temp[k] = this.__discrete(np.matmul(this.w[k,],temp))
                    plt.cla()
                    plt.imshow(temp.reshape(11,7),cmap=plt.cm.gray)
                    plt.title('class : {}'.format(this.img.className(label)))
                    plt.pause(.005)
            if (temp==output).all():
                break
            output = temp
            return output,i
               

def run():
    onoff=int(input('1->Async\n2->Sync'))

    hpf=hopfield('async' if onoff==1 else 'sync')
    
    hpf.fit()
    x,y=hpf.img.makeimgs(int(input('Dwse arithmo paramorfomenon psifion ana psifio : ')))
    
    hpf.test(x,y)
