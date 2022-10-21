from turtle import update
import numpy as np
import lines as l
import matplotlib.pyplot as plt

rn=lambda  som,m,n:(m-n)*np.random.random_sample((
        som.k,))+n

class som :
    def __init__(this,k:int):
        this.β=1
        this.Γ=1
        this.k=k
    
    _find_winner=lambda this,nv:np.unravel_index(
        np.argmin(nv,axis=None), nv.shape
    )
    _ind = lambda this,data :np.sum(np.square(this.w-data),axis=2)
    
    def __updw(this,xtr,wnr):
        x,y=wnr
        nrs=[]
        if this.Γ< 1e-3:
            this.w[x,y,:]+=this.β*(xtr-this.w[x,y,:])
            return nrs 
        step=int(this.Γ*10)
      
        """
        for i in range(max(0,x-step) , min(this.w.shape[0],x+step)):
             for j in range(max(0,y-step) , min(this.w.shape[1],y+step)):
                if((i,j)!=(x,y)):
                     this.w[i,j,:]+=(this.β/2)*(xtr-this.w[i,j,:])
                     nrs.append((i,j)) 
       """ 
        for i in range(max(0,x-step) , min(this.w.shape[0],x+step)):
             for j in range(max(0,y-step) , min(this.w.shape[1],y+step)):
                 if((i,j)!=(x,y)):nrs.append((i,j))
                 nrm=np.exp(-(np.square(i-x)+np.square(j-y))/2/this.Γ)
                 this.w[i,j,:]+=this.β*nrm*(xtr-this.w[i,j,:])
         
        return nrs

    def _get_2dcoords(this):
        s=np.array(this.w[:,:,1])
        d=np.array(this.w[:,:,0])
        s=s.reshape( (len(s)**2) )
        d=d.reshape( (len(d)**2) )
        return d,s

    def _get_coords(this):
        s=np.array(this.w[:,:,1])
        d=np.array(this.w[:,:,0])
        f=np.array(this.w[:,:,2])
        s=s.reshape( (len(s)**2) )
        d=d.reshape( (len(d)**2) )
        f=f.reshape( (len(f)**2) )
        return d,s,f

    def __plot2(this,wn,gamma,X,y,ax ):
        i,j=wn
        ax[0].cla()
        ax[1].cla()
        w1,w2=this._get_2dcoords()
        l.plot_data2d(ax[0],y,X[:,0],X[:,1], 'Original Data')
        l.plot_data2d(ax[1],y,X[:,0],X[:,1],'EPOCHS {}/{}'.format(this.r,this.epochs))
        ax[1].plot(w1,w2,'ko')
        ax[1].plot(this.w[i,j,0],this.w[i,j,1],'ro',markersize=13, label='Winner Neuron')
        for g in gamma:
            ax[1].plot(this.w[g[0],g[1],0],this.w[g[0],g[1],1],'bo',markersize=7)
        
             
        ax[1].legend(labels=['class 0','class 1','Other Neurons','Winner Neuron','Winners Neighborhood'])
      
    def __pltflwr(this,wn,gamma,X,y,ax):
        i,j=wn
        ax[0].cla()
        ax[1].cla()
        w1,w2,w3=this._get_coords()
        l.plot_data2d(ax[0],y,X[:,0],X[:,2], 'Original Data')
        l.plot_data2d(ax[1],y,X[:,0],X[:,2],'EPOCHS {}/{}'.format(this.r,this.epochs))
        ax[1].plot(w1,w3,'ko')
        ax[1].plot(this.w[i,j,0],this.w[i,j,2],'ro',markersize=13, label='Winner Neuron')
        for g in gamma:
            ax[1].plot(this.w[g[0],g[1],0],this.w[g[0],g[1],2],'bo',markersize=7,)
        ax[0].legend(
            labels=['{}/{} Epochs'.format(this.r,this.epochs)] 
            ,loc=4,fontsize='large'
        )
             
        ax[1].legend(labels=['class 0','class 1','class 2','Other Neurons','Winner Neuron','Winners Neighborhood'])

    def __plot3(this,wn,gamma,X,y,ax):
        ax[0].cla()
        ax[1].cla()
        i,j=wn
        w1,w2,w3=this._get_coords()
        l.plot_data3d(ax[0],y,X[:,0],X[:,1],X[:,2], 'Original Data')
        l.plot_data3d(ax[1],y,X[:,0],X[:,1],X[:,2],'EPOCHS {}/{}'.format(this.r,this.epochs))
        ax[1].plot(w1,w2,w3,'ko')
        ax[1].plot(this.w[i,j,0],this.w[i,j,1],this.w[i,j,2],'ro',markersize=13, label='Winner Neuron')
        for g in gamma:
            ax[1].plot(this.w[g[0],g[1],0],this.w[g[0],g[1],1],this.w[g[0],g[1],2],'bo',markersize=9)
        ax[0].legend(
            labels=['{}/{} Epochs'.format(this.r,this.epochs)] 
            ,loc=4,fontsize='large'
        )
     
        ax[1].legend(labels=['class 0','class 1','Other Neurons','Winner Neuron','Winners Neighborhood'])

    def test(this,X,y):
        import matplotlib.patheffects as pe
        
        if X.shape[1]==2:
                fig,ax=plt.subplots(1,2)
                l.plot_data2d(ax[0],y,X[:,0],X[:,1])
                l.plot_data2d(ax[1],y,X[:,0],X[:,1])
                ax[1].plot(this.w[:,:,0],this.w[:,:,1],'wo',markersize=10 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
               )
        elif X.shape[1]==3:
                fig=plt.figure(figsize=(8,8))
                ax=np.zeros((2,),dtype=object)
                ax[0]=fig.add_subplot(1, 2, 1, projection='3d')       
                ax[1]=fig.add_subplot(1, 2, 2, projection='3d')
                l.plot_data3d(ax[0],y,X[:,0],X[:,1],X[:,2])
                l.plot_data3d(ax[1],y,X[:,0],X[:,1],X[:,2])
                wx,wy,wz=this._get_coords()
                ax[1].plot(wx,wy,wz,'wo',markersize=10 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
                )
        elif X.shape[1]==4:
                fig,ax=plt.subplots(1,2)
                l.plot_data2d(ax[0],y,X[:,0],X[:,2])
                l.plot_data2d(ax[1],y,X[:,0],X[:,2])
                ax[1].plot(this.w[:,:,0],this.w[:,:,2],'wo',markersize=10 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
                )
        
        plt.show()
#'wo',markersize=15 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
            
    def __init_components(this,X,figure=True):
        ndim=X.shape[1]
        plot=None
        ax=None
        if figure:
            if ndim==3 :
                plot=this.__plot3
                fig=plt.figure(figsize=(8,8))
                ax=np.zeros((2,),dtype=object)
                ax[0]=fig.add_subplot(1, 2, 1, projection='3d')       
                ax[1]=fig.add_subplot(1, 2, 2, projection='3d')
            elif ndim==4 :
                plot=this.__pltflwr
                fig,ax=plt.subplots(1,2)
            else:
                plot=this.__plot2
                fig,ax=plt.subplots(1,2)

        c=np.zeros((this.k,this.k,ndim))
        for k in range(ndim):
            c[:,:,k]=rn(this,np.max(X[:,k]),np.min(X[:,k]))
        this.w=c
        return ax,plot
    
    def fitkohonen(this,x,epochs=5000):
        geitonia=500
        this.__init_components(x,False)
        β_old=this.β
        Γ_old=this.Γ
        this.r=0
        data=np.copy(x)
        for e in range(epochs):
            np.random.shuffle(data)
            for x in data:
                i,j=this._find_winner(
                    this._ind(x)
                )
                this.__updw(x,(i,j))
            this.β=β_old*(1-e/epochs) 
            this.Γ=Γ_old*(1-e/geitonia)
    @staticmethod
    def srt(x,y):
          data=np.hstack((x,y.reshape((len(y),1))))
          data=np.sort(data,axis=0)
          return data[:,:-1],data[:,-1]

    def som_fit(this,X,y,epochs,cond):
        this.epochs=epochs
        ax,plot=this.__init_components(X)
        #data,goals=this.srt(X,y)
        β_old=this.β
        Γ_old=this.Γ
       
        for epoch in range(epochs):
            this.r=epoch
            i,j=None,None
            gamma=None
            #np.random.shuffle(data)
            for x in X:
                i,j=this._find_winner(
                    this._ind(x)
                )
                gamma=this.__updw(x,(i,j))
                if cond(epoch):
                  plot((i,j),gamma,X,y,ax)
                  plt.pause(0.00005) 
            this.β=β_old*(1-this.r/this.epochs) 
            this.Γ=Γ_old*(1-this.r/500) 

    def fit(this,X,y,epochs):
        this.epochs=epochs
        ax,plot=this.__init_components(X)
        data=np.copy(X)
        
        
        for epoch in range(epochs):
            β_old=this.β
            Γ_old=this.Γ
            this.r=epoch
            i,j=None,None
            gamma=None
            np.random.shuffle(data)
            for x in data:
                i,j=this._find_winner(
                    this._ind(x)
                )
                gamma=this.__updw(x,(i,j))
            plot((i,j),gamma,X,y,ax)
            plt.pause(0.005) 
            this.β*=(1-this.r/this.epochs) 
            this.Γ*=(1-this.r/this.epochs) 
            #this.β=β_old*np.exp(-this.r*.1)
            #this.Γ=Γ_old*np.exp(-(this.r/this.epochs)*.1)
    

 
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
   
   model=som(int(input('Dwse arithmo k gia kxk neurones : ')))
   update_rate=int(input('Dwse arithmo ana poses epoxes tha kanei refresh to plot : '))
   model.som_fit(x , y ,int(input('Dwse arithmo epoxon : ')),lambda epoch : True
   if update_rate==1 or update_rate==0 else epoch%update_rate==0)
   input('Press enter to test...')
   model.test(x,y)

if __name__=='__main__':run()
