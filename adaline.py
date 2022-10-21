import matplotlib.patheffects as pe
import numpy as np
from sklearn.model_selection import train_test_split
import functions as fn
import matplotlib.pyplot as plt
from Kmeans import kmeans, σ
import lines as l

class adln():

    def __init__(self,n_features,β,kernel):
       self.w=np.random.random_sample((n_features+1,))
       self.β=β
       self.type=kernel

    def setNewN(self,n_features):
      self.w=np.random.random_sample((n_features+1,))
    def setlr(self,β):
       self.β=β
    def Σ(self,x):
     return np.dot(self.w,x)
    def predict(self,X):
      preds=[]
      for x in X:
       preds.append(self.Σ(x))
      return np.array(preds)
    def correct(self,u,x,d):
        for i in range(len(x)):
            self.w[i]+=self.β*(d-u)*x[i]

    def test(self,X,y):
     if self.type=='default':
      if X.shape[1]==13:
        x,y=fn.normalize(np.copy(X),np.copy(y))
        x=fn.add_bias(x)
        l.plotRegress(self,x,y)
        plt.show()
      else :
        self.testNtype(X,y)
     else:
       if X.shape[1]==13:
         x=fn.add_bias(X)
         l.plotRegress(self,x,y)
         plt.show()
       else:
         self.plotrbf(X,y)
         plt.show()

    def plotrbf(self,x,y):
       preds=self.predict(fn.add_bias(fn.rbf_pack(x,self.centers,self.sigma,self.type)))
       
       if x.shape[1]==2:
          fig,ax=plt.subplots(3,1)
          l.plot_data2d(ax[0],y,x[:,0],x[:,1] ,'Original Data')
          prediction_map=np.where(preds>0,1,-1)
          l.plot_data2d(ax[1],prediction_map,x[:,0],x[:,1],'Classifed Data')
          ax[1].plot(self.centers[:,0],self.centers[:,1],'wo',markersize=15,lw=2,alpha=0.4, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
          ,label='Centroids' )
          ax[1].legend()
          l.plot_data2d(ax[2],y,x[:,0],preds , 'Output Graph')
          
         
       elif x.shape[1]==3:
          fig=plt.figure()
          ax=np.zeros((3,),dtype=object)
          ax[0]=fig.add_subplot(1, 3, 1, projection='3d')
          ax[1]=fig.add_subplot(1, 3, 2, projection='3d')
          ax[2]=fig.add_subplot(1, 3, 3,)
          l.plot_data3d(ax[0],y,x[:,0],x[:,1],x[:,2],'Original Data')
          prediction_map=np.where(preds>0,1,-1)
          l.plot_data3d(ax[1],prediction_map,x[:,0],x[:,1],x[:,2],'Classifed Data')
          ax[1].plot(self.centers[:,0],self.centers[:,1],self.centers[:,2],'wo',markersize=15,lw=2,alpha=0.4, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()] 
           ,label='Centroids')
          l.plot_data2d(ax[2],y,x[:,0],preds , 'Output Graph')

      
    def testNtype(self,X,y):
      p=[]
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
      for x,d in zip(fn.add_bias(X),y):
          p.append(self.Σ(x))
      plot(ax,X,y,np.array(p),self.w)
      plt.show()

    def partial_fit(self,X,y):
        yd=[]
        for x,d in zip(X,y):
         u=self.Σ(x)
         yd.append(u)
         self.correct(u,x,d)
        err=np.mean((y-yd)**2)
        return err,yd

    def __train_with_plots_norbf(this,twod,xtrain,ytrain,epochs,update_every=0):
        if twod:
           fig,ax=plt.subplots(2,2)
           plot=l.plot12d
        else:
           fig=plt.figure()
           ax=np.zeros((2,2),dtype=object)
           ax[0,0]=fig.add_subplot(2, 2, 1, projection='3d')
           ax[0,1]=fig.add_subplot(2, 2, 2, projection='3d')
           ax[1,0]=fig.add_subplot(2, 2, 3)
           ax[1,1]=fig.add_subplot(2, 2, 4)
           plot=l.plot13d
        if update_every==0:
           cond=lambda it:True
        else:
           cond=lambda it :it%update_every==0
        data=fn.add_bias(xtrain)
        goals=np.where(ytrain==0,-1,ytrain)
        line=[]
        for i in range(1,int(epochs)+1):
            err,pred=this.partial_fit(data,goals)
            line.append(err)
            if cond(i):
               plot(ax,xtrain,goals,np.array(pred),this.w,err,line)
               plt.pause(0.005)

    def __train_with_plots_rbf(this,twod,xtrain,ytrain,epochs,centroids,update_every=0 ):
        if twod:
           fig,ax=plt.subplots(2,2)
           plot=l.plotkmn
        else:
           fig=plt.figure()
           ax=np.zeros((2,2),dtype=object)
           ax[0,0]=fig.add_subplot(2, 2, 1, projection='3d')
           ax[0,1]=fig.add_subplot(2, 2, 2, projection='3d')
           ax[1,0]=fig.add_subplot(2, 2, 3)
           ax[1,1]=fig.add_subplot(2, 2, 4)
           plot=l.plotkmn3d
        if(update_every)==0:
           cond=lambda it:True
        else:
           cond=lambda it :it%update_every==0
        k=centroids
        ctr=kmeans(k,xtrain)
        sigma=σ(k,ctr,xtrain)
        this.sigma=sigma
        data=fn.rbf_pack(xtrain,ctr,sigma,this.type)
        goals=np.where(ytrain==0,-1,ytrain)
        this.setNewN(data.shape[1])
        data=fn.add_bias(data)
        this.centers=ctr
        line=[]
        for i in range(1,epochs+1):
            err,preds=this.partial_fit(data,goals)
            line.append(err)
            if cond(i):
               plot(ax,xtrain,ytrain,np.array(preds),ctr,err,line)
               plt.pause(0.005)

    def train_with_plots(this,xtrain,ytrain,epochs,update_every=0 , centroids=4):
     twod=xtrain.shape[1]==2
     if xtrain.shape[1]==13 : this.regression(xtrain,ytrain,epochs,centroids)

     elif this.type=='default':
        this.__train_with_plots_norbf(twod,xtrain,ytrain,epochs,update_every)
        
     else:
        this.__train_with_plots_rbf(twod,xtrain,ytrain,epochs,centroids,update_every)


    def regression(this,xtrain,ytrain,epochs ,centroids):
        
     if this.type=='default':
        data,goals=fn.normalize(np.copy(xtrain),np.copy(ytrain))
        data=fn.add_bias(data)
        for i in range (epochs):
            this.partial_fit(data,goals)
            plt.cla()
            l.plotRegress(this,data,goals)
            plt.pause(.0005)
     else:
        k=centroids
        ctr=kmeans(k,xtrain)
        sigma=σ(k,ctr,xtrain)
        data=fn.rbf_pack(xtrain,ctr,sigma,this.type)
        this.setNewN(data.shape[1])
        data,goals=fn.normalize(data,np.copy(ytrain))
        data=fn.add_bias(data)
        for i in range(epochs):
            this.partial_fit(data,goals)
            plt.cla()
            l.plotRegress(this,data,goals)
            plt.pause(.0005)

def run():
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
   else : raise ValueError('Invalid Input')
      
   xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3)
   lr=float(input('vima ekpedeusis :'))
   kernel=int(input('Pirinas: \n1->default\n2->Multiquadric\n3->Gauss\n4->Cauchy\n'))
   kernels={1:'default',2:'Multiquadric',3:'Gauss',4:'Cauchy'} 
   model= adln(x.shape[1],lr,kernels[kernel])
   ctr=0
   if kernel!=1 : ctr=int(input('arithmos kentron:'))
   model.train_with_plots(xtrain,ytrain,int(input('epoxes :')),centroids=ctr)
   input('Press enter to test...')
   model.test(xtest,ytest)
   
if __name__=='__main__':run()
