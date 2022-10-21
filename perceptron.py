import numpy as np

import functions as fn
import lines as l 
import matplotlib.pyplot as plt


class prc():
   def __init__(self,n_features,β,func=fn.step):
       self.n=n_features
       self.w=(0.5+0.5)*np.random.random_sample((n_features+1,))-0.5
       self.β=β
       self.f=func

   def setlr(self,β):
       self.β=β
          
   def Σ(self,x):
    return np.dot(self.w,x)
   def correct(self,x,d,v):
       for i in range(len(x)):
           self.w[i]+=self.β*(d-v)*x[i]

   def partial_fit(self,X,y):
       preds=[]
       for x,d in zip(X,y):
        u=self.Σ(x)
        v=self.f(u)
        preds.append(v)
        if(v!=d):
           
           self.correct(x,d,v)
       return np.array(preds)
       

   def train(self,X,y,epochs):  
    for i in range(epochs):
     temp=np.copy(self.w)
     for x,d in zip(X,y):
        u=self.Σ(x)
        v=self.f(u)
        if(v!=d):
           self.correct(x,d,v)
     if(np.array_equal(temp,self.w)):break 

   def test(self,X,y):
    p=[]
    if X.shape[1]==2:
        fig,ax=plt.subplots(1,3)
        plot=l.plot02d
    elif X.shape[1]==3:
        fig=plt.figure()
        ax=np.zeros((3,),dtype=object)
        ax[0]=fig.add_subplot(1, 3, 1, projection='3d')
        ax[1]=fig.add_subplot(1, 3, 2, projection='3d')
        ax[2]=fig.add_subplot(1, 3, 3, projection='3d')
        plot=l.plot03d
    for x,d in zip(fn.add_bias(X),y):
        p.append(self.f(self.Σ(x)))
    plot(ax,X,y,np.array(p),self.w)
    plt.show()

   def train_with_plots(this,xtrain , ytrain , epochs):
    twod=xtrain.shape[1]==2
    data=fn.add_bias(np.copy(xtrain))
    if twod:
        fig,ax=plt.subplots(3,1)
        plot=l.plot02d
    else:
        fig=plt.figure()
        ax=np.zeros((3,),dtype=object)
        ax[0]=fig.add_subplot(1, 3, 1, projection='3d')
        ax[1]=fig.add_subplot(1, 3, 2, projection='3d')
        ax[2]=fig.add_subplot(1, 3, 3, projection='3d')
        plot=l.plot03d
    
    for i in range (epochs): 
        stp=np.copy(this.w)
        preds=this.partial_fit(data,ytrain)
        plot(ax,xtrain,ytrain,preds,this.w)
        if(np.array_equal(stp,this.w)):break
        plt.pause(0.05)

   def train_v2(this, xtrain , ytrain , epochs):
         
        twod=xtrain.shape[1]==2
        data=fn.add_bias(np.copy(xtrain))
        if twod:
            fig,ax=plt.subplots(1,3)
            plot=l.plot02d
        else:
            fig=plt.figure()
            ax=np.zeros((3,),dtype=object)
            ax[0]=fig.add_subplot(1, 3, 1, projection='3d')
            ax[1]=fig.add_subplot(1, 3, 2, projection='3d')
            ax[2]=fig.add_subplot(1, 3, 3, projection='3d')
            plot=l.plot03d
        
        for i in range (epochs): 
            stp=np.copy(this.w)
            
            for x,d in zip(data,ytrain):
             u=this.Σ(x)
             v=this.f(u)   
             if(v!=d):
                this.correct(x,d,v)
                plot(ax,xtrain,ytrain,this.predict(data),this.w ,epochs , i)   
                plt.pause(0.05)

            if(np.array_equal(stp,this.w)):break
   def predict(this , x ):
        preds=[]
        for p in  x:
            preds.append(this.f(this.Σ(p)))
        return np.array(preds)
        
def run():
   import data as dt
   from sklearn.model_selection import train_test_split
   choice=int(input('1->Grammika Diaxorisima\n2->Goneia\n3->XOR\n4->Kentro\n5->Grammika Diaxorisima 3d \
   \n6->Xor 3d\n'))
   x,y=None,None

   if choice== 1:x,y=dt.l_sep(int(input('Arithmos protipon:')))
   elif choice==2:x,y=dt.angular(int(input('Arithmos protipon:')))
   elif choice==3:x,y=dt.xor(int(input('Arithmos protipon:')))
   elif choice==4:x,y=dt.ciricular(int(input('Arithmos protipon:')))
   elif choice==5:x,y=dt.l_sep3d(int(input('Arithmos protipon:')))
   elif choice==6:x,y=dt.xor_3d(int(input('Arithmos protipon:')))
   else: raise ValueError('Invalid Input')
   xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3)
   lr=float(input('vima ekpedeusis :'))
   model=prc(x.shape[1],lr)
   model.train_v2(xtrain,ytrain,int(input('dwse epoxes: ')))
   input('Press enter to test...')
   model.test(xtest,ytest)


run()