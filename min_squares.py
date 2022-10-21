
import lines as l
import matplotlib.pyplot as plt
import numpy as np
import functions as fn

class min_sqr():
    def __init__(self,n_features):
       self.w=np.random.random_sample((n_features+1,))
    def fit(self,X,d):
        x=np.linalg.pinv(X)
        self.w=np.matmul(x,d)
    def preds(self,X):
        preds=[]
        for  x in X:
            preds.append(fn.step_1(np.dot(self.w,x)))
        return np.array(preds)
    def predict(self,X):
        preds=[]
        for  x in X:
            preds.append(np.dot(self.w,x))
        return np.array(preds)
        
    def test(self,X,y):
        p=[]
        data=fn.add_bias(np.copy(X))
        if X.shape[1]==2:
            fig,ax=plt.subplots(3,1)
            plot=l.plot02d
        elif X.shape[1]==3:
            fig=plt.figure()
            ax=np.zeros((3,),dtype=object)
            ax[0]=fig.add_subplot(1, 3, 1, projection='3d')
            ax[1]=fig.add_subplot(1, 3, 2, projection='3d')
            ax[2]=fig.add_subplot(1, 3, 3, projection='3d')
            plot=l.plot03d
        else :
            l.plotRegress(self,data,y)
            plt.show()
            return
        for x,d in zip(data,y):
            p.append(np.dot(self.w,data))
        plot(ax,X,y,np.array(p),self.w)
  
    def train_with_plots(this,xtrain,ytrain):
        if xtrain.shape[1]==13:this.regression(xtrain,ytrain)
        else : this.classification(xtrain,ytrain)

    def regression(this,xtrain,ytrain):
        data=fn.add_bias(np.copy(xtrain))
        this.fit(data,ytrain)
        l.plotRegress(this,data,ytrain)
        plt.show()

    def classification(this,xtrain,ytrain):
        twod=xtrain.shape[1]==2
        data=fn.add_bias(xtrain)
        goals=np.where(ytrain==0,-1,ytrain)
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
        this.fit(data,goals)
        plot(ax,xtrain,goals,this.preds(data),this.w)
        plt.show()



def run():
   from sklearn.model_selection import train_test_split
   import data as dt
   from sklearn.datasets import load_boston
   choice=int(input('1->Grammika Diaxorisima\n2->Goneia\n3->XOR\n4->Kentro\n5->Grammika Diaxorisima 3d \
   \n6->Xor 3d\n7->Palindromisi\n'))
   x,y=None,None
  
   if choice== 1:x,y=dt.l_sep(int(input('Arithmos protipon:')))
   elif choice== 2:x,y=dt.angular(int(input('Arithmos protipon:')))
   elif choice== 3:x,y=dt.xor(int(input('Arithmos protipon:')))
   elif choice== 4:x,y=dt.ciricular(int(input('Arithmos protipon:')))
   elif choice== 5:x,y=dt.l_sep3d(int(input('Arithmos protipon:')))
   elif choice== 6:x,y=dt.xor_3d(int(input('Arithmos protipon:')))
   elif choice== 7:x,y=(load_boston().data , load_boston().target)
   else: raise ValueError('Invalid Input')
   xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3)
   model=min_sqr(x.shape[1])
   model.train_with_plots(xtrain,ytrain)
   input('Press enter to test...')
   model.test(xtest,ytest)
        

