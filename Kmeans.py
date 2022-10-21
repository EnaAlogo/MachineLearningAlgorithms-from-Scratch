import matplotlib.pyplot as plt
import numpy as np
import lines as l
import matplotlib.patheffects as pe



rn=lambda  max,min,shape:(max-min)*np.random.random_sample(shape)+min
class kmn():
    def __init__(this,k):
        this.k=k

    def fitanim(this,X,y):
        ndim=X.shape[1]
        sp=(this.k,)
        c=np.zeros((this.k,ndim))
        for k in range(ndim):
            c[:,k]=rn(np.max(X[:,k]),np.min(X[:,k]),sp)
        this.c=c
        #c=0.9*np.random.random_sample((this.k,ndim))
        c_old=np.zeros((this.k,ndim))
        if ndim==2 or ndim==4:
            fig,ax=plt.subplots(2,1)
        elif ndim==3:
           fig=plt.figure()
           ax=np.zeros((2,),dtype=object)
           ax[0]=fig.add_subplot(1, 2, 1, projection='3d')
           ax[1]=fig.add_subplot(1, 2,2, projection='3d')
        
        while not np.array_equal(c_old,this.c):
            c_old=np.copy(this.c)
            inx=[]
            for x in X:
                dst=[]
                for ctr in this.c:
                    dst.append(np.linalg.norm(x-ctr)**2)
                inx.append(np.argmin( np.array(dst) ) )
            this.c=np.zeros((this.k,ndim))
            cnt=np.zeros((this.k,))
            for x,inc in zip(X,inx):
                this.c[inc]+=x
                cnt[inc]+=1
            for j in range(this.k):
                if cnt[j]!=0:
                    this.c[j]/=cnt[j]
                
            if ndim==2:
                    ax[0].cla()
                    ax[1].cla()
                    l.plot_data2d(ax[0],y,X[:,0],X[:,1] ,'Original Data')
                    l.plot_data2d(ax[1],y,X[:,0],X[:,1])
                    ax[1].plot(this.c[:,0],this.c[:,1],'wo',markersize=15,lw=2,alpha=0.4, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()] 
                    ,label='Centroids') 
                    ax[1].legend()
                    plt.pause(0.05)
                
            elif ndim==3:
                ax[0].cla()
                ax[1].cla()
                l.plot_data3d(ax[0],y,X[:,0],X[:,1],X[:,2] ,'Original Data')
                l.plot_data3d(ax[1],y,X[:,0],X[:,1],X[:,2])
                ax[1].plot(this.c[:,0],this.c[:,1],this.c[:,2],'wo',markersize=15,lw=2,alpha=0.4, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()] 
                ,label='Centroids') 
                ax[1].legend()
                plt.pause(0.05)
            elif ndim==4:
                ax[0].cla()
                ax[1].cla()
                l.plot_data2d(ax[0],y,X[:,0],X[:,2] ,'Original Data')
                l.plot_data2d(ax[1],y,X[:,0],X[:,2])
                ax[1].plot(this.c[:,0],this.c[:,2],'wo',markersize=15,lw=2,alpha=0.4, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()] 
                ,label='Centroids') 
                ax[1].legend()
                plt.pause(0.05)
                
    def test(this,x,d):
     
      if x.shape[1]== 2:
          this.test2d(x,d)
      elif x.shape[1]==3:
          this.test3d(x,d)
      elif x.shape[1]== 4:
          this.testflw(x,d)

    def test2d(this,x,d):
      fig,ax=plt.subplots(1,2)
      l.plot_data2d(ax[0],d,x[:,0],x[:,1] ,'Original Data')
      l.plot_data2d(ax[1],d,x[:,0],x[:,1])

      ax[1].plot(this.c[:,0],this.c[:,1],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
      ,label='Centroids')
      ax[1].legend()
      
      plt.show()

    def testflw(this,x,d):
      fig,ax=plt.subplots(1,2)
      l.plot_data2d(ax[0],d,x[:,0],x[:,2] ,'Original Data')
      l.plot_data2d(ax[1],d,x[:,0],x[:,2])

      ax[1].plot(this.c[:,0],this.c[:,2],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
                 ,label='Centroids')
      ax[1].legend()
      
      plt.show()

    def test3d(this,x,d):
      fig=plt.figure()
      ax=np.zeros((2,),dtype=object)
      ax[0]=fig.add_subplot(1, 2, 1, projection='3d')
      ax[1]=fig.add_subplot(1, 2, 2, projection='3d')
      l.plot_data3d(ax[0],d,x[:,0],x[:,1],x[:,2] ,'Original Data')
      l.plot_data3d(ax[1],d,x[:,0],x[:,1],x[:,2]) 
     
      ax[1].plot(this.c[:,0],this.c[:,1],this.c[:,2],'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
      ,label='Centroids')
      ax[1].legend()
      
      plt.show()

            


def kmeans(k,X):
    ndim=X.shape[1]
    c=np.random.random_sample((k,ndim))
    c_old=np.zeros((k,ndim))
    while not np.array_equal(c_old,c):
        c_old=np.copy(c)
        inx=[]
        for x in X:
            dst=[]
            for ctr in c:
                dst.append(np.linalg.norm(x-ctr)**2)
            inx.append(np.argmin( np.array(dst) ) )
        c=np.zeros((k,ndim))
        cnt=np.zeros((k,))
        for x,inc in zip(X,inx):
            c[inc]+=x
            cnt[inc]+=1
        for j in range(k):
            if cnt[j]!=0:
                c[j]/=cnt[j]
    return c

def σ(k,c,X):
    ap=np.zeros((k*k))
    for i in range(k):
        for j in range(k):
            ap[((i-1)*k+j)] = np.linalg.norm(c[i] - c[j])**2
    megisto = np.max(ap)
    σ=megisto/np.sqrt(2*X.shape[0])
    return σ


def run():
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import load_iris
   import data as dt
   choice=int(input('1->Grammika Diaxorisima\n2->Goneia\n3->XOR\n4->Kentro\n5->Grammika Diaxorisima 3d \
   \n6->Xor 3d\n7->Iris louloudia\n'))
   x,y=None,None
   
   if choice==1:x,y=dt.l_sep(int(input('Arithmos protipon:')))
   elif choice== 2:x,y=dt.angular(int(input('Arithmos protipon:')))
   elif choice== 3:x,y=dt.xor(int(input('Arithmos protipon:')))
   elif choice== 4:x,y=dt.ciricular(int(input('Arithmos protipon:')))
   elif choice== 5:x,y=dt.l_sep3d(int(input('Arithmos protipon:')))
   elif choice== 6:x,y=dt.xor_3d(int(input('Arithmos protipon:')))
   elif choice== 7:x,y = (load_iris().data , load_iris().target)
   else: raise ValueError('Invalid Input')
     
   xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3)

   model=kmn(int(input('Dwse arithmo kentron : ')))
   model.fitanim(xtrain,ytrain)
   input('Press enter to test...')
   model.test(xtest,ytest)

if __name__=='__main__':run()
