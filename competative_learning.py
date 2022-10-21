import numpy as np
import lines as l
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

class comp_lr:
    def __init__(this,k,β):
        this.k=k
        this.β=β
    def set_k(this,x,lr):
      c=np.random.random_sample((this.k,x.shape[1]))
      this.c=c
      this.β=lr
     
    def fit(this,X,epochs,current_epoch):
        for x in X:
          norms=[]
          for ct in range (this.c.shape[0]):
            norms.append(np.linalg.norm(this.c[ct]-x) )
          indx=np.argmin(norms)
          this.c[indx]+=this.β*(x-this.c[indx])
        this.β*=(1-(current_epoch/epochs))
        
    def test(this,x,d):
     
      if x.shape[1]==2:
          this.test2d(x,d)
      elif x.shape[1]==3:
          this.test3d(x,d)
      elif x.shape[1]== 4:
          this.testflw(x,d)

    def test2d(this,x,d):
      fig,ax=plt.subplots(1,2)
      l.plot_data2d(ax[0],d,x[:,0],x[:,1])
      l.plot_data2d(ax[1],d,x[:,0],x[:,1])
     
      ax[1].plot(this.c[:,0],this.c[:,1],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
      ,label='Neurons')
      ax[1].legend()
      plt.show()

    def testflw(this,x,d):
      fig,ax=plt.subplots(1,2)
      l.plot_data2d(ax[0],d,x[:,0],x[:,2])
      l.plot_data2d(ax[1],d,x[:,0],x[:,2])

      ax[1].plot(this.c[:,0],this.c[:,2],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
      ,label='Neurons')
      ax[1].legend()
      plt.show()

    def test3d(this,x,d):
      fig=plt.figure()
      ax=np.zeros((2,),dtype=object)
      ax[0]=fig.add_subplot(1, 2, 1, projection='3d')
      ax[1]=fig.add_subplot(1, 2, 2, projection='3d')
      l.plot_data3d(ax[0],d,x[:,0],x[:,1],x[:,2])
      l.plot_data3d(ax[1],d,x[:,0],x[:,1],x[:,2]) 

      ax[1].plot(this.c[:,0],this.c[:,1],this.c[:,2],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
      ,label='Neurons')
      ax[1].legend()
      plt.show()
    
    def train_with_plots(this, xtrain,ytrain ,epochs):
      plotmap={2:this.__plot2d , 3 : this.__plot3d , 4:this.__plotflwr}
      plotmap[xtrain.shape[1]](xtrain,ytrain,epochs)
    
    def __plotflwr(this, xtrain ,ytrain , epochs ):
        rn=lambda  max,min,shape:(max-min)*np.random.random_sample(shape)+min
        sp=(this.k,)
        c1=rn(np.max(xtrain[:,0]),np.min(xtrain[:,0]),sp)
        c2=rn(np.max(xtrain[:,1]),np.min(xtrain[:,1]),sp)
        c3=rn(np.max(xtrain[:,2]),np.min(xtrain[:,2]),sp)
        c4=rn(np.max(xtrain[:,3]),np.min(xtrain[:,3]),sp)
        c=np.vstack((np.vstack((c1,c2)),np.vstack((c3,c4))))
        this.c=c.T
        fig,ax=plt.subplots(1,2)
        l.plot_data2d(ax[0],ytrain,xtrain[:,0],xtrain[:,2])
        for i in range(epochs):
               this.fit(xtrain,epochs,i)
               ax[1].cla()
               l.plot_data2d(ax[1],ytrain,xtrain[:,0],xtrain[:,2],'EPOCHS : {}/{}'
            .format(i,epochs)) 
               ax[1].plot(this.c[:,0],this.c[:,2],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
                  ,label='Neurons')
               ax[1].legend()
               plt.pause(0.005)


    def __plot2d(this,xtrain,ytrain,epochs):
      fig,ax=plt.subplots(1,2)
      l.plot_data2d(ax[0],ytrain,xtrain[:,0],xtrain[:,1])
      for i in range(epochs):
               this.fit(xtrain,epochs,i)
            
               ax[1].cla()
               l.plot_data2d(ax[1],ytrain,xtrain[:,0],xtrain[:,1],'EPOCHS : {}/{}'
            .format(i,epochs)) 
            
               ax[1].plot(this.c[:,0],this.c[:,1],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
               ,label='Neurons')
               ax[1].legend()
               plt.pause(0.005)

    def __plot3d(this, xtrain,ytrain ,epochs):
        fig=plt.figure()
        ax=np.zeros((2,),dtype=object)
        ax[0]=fig.add_subplot(1, 2, 1, projection='3d')
        ax[1]=fig.add_subplot(1, 2, 2, projection='3d')
        l.plot_data3d(ax[0],ytrain,xtrain[:,0],xtrain[:,1],xtrain[:,2])
        for i in range(epochs):
                this.fit(xtrain,epochs,i)
       
                ax[1].cla()
                l.plot_data3d(ax[1],ytrain,xtrain[:,0],xtrain[:,1],xtrain[:,2],'EPOCHS : {}/{}'
            .format(i,epochs)) 
            
                ax[1].plot(this.c[:,0],this.c[:,1],this.c[:,2],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
                ,label='Neurons')
                ax[1].legend()
                plt.pause(0.005)

def run():
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import load_iris
   import data as dt
   choice=int(input('1->Grammika Diaxorisima\n2->Goneia\n3->XOR\n4->Kentro\n5->Grammika Diaxorisima 3d \
   \n6->Xor 3d\n7->Iris louloudia\n'))
   x,y=None,None

   if choice==1:x,y=dt.l_sep(int(input('Arithmos protipon:')))
   elif choice==2:x,y=dt.angular(int(input('Arithmos protipon:')))
   elif choice==3:x,y=dt.xor(int(input('Arithmos protipon:')))
   elif choice==4:x,y=dt.ciricular(int(input('Arithmos protipon:')))
   elif choice==5:x,y=dt.l_sep3d(int(input('Arithmos protipon:')))
   elif choice==6:x,y=dt.xor_3d(int(input('Arithmos protipon:')))
   elif choice==7:x,y = (load_iris().data , load_iris().target)
   else: raise ValueError('Invalid Input')
   
   xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3)
   lr = float(input('Dwse arxiko vima ekpedeusis: '))
   k= int(input('Dwse arithmo antagonistikon neuronon : '))
   model=comp_lr(k,lr)
   model.set_k(x,lr)
   model.train_with_plots(xtrain,ytrain,int(input('Dwse arithmo epoxon : ')))
   input('Press enter to test...')
   model.test(xtest,ytest)


if __name__=='__main__':run()