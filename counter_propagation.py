import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from SOM import som
import numpy as np
import lines as l

class cpn(som):
    def __init__(this,k):
        super().__init__(k)
        this.α=.1

    def __updw(this, xtr, wnr:tuple):
        this.w[wnr[0],wnr[1],:]+=this.β*(xtr-this.w[wnr[0],wnr[1],:])

    def __gsb_updw(this,x,d,wnr):
        for i in range(this.g_w.shape[2]): 
            if i==d:
                this.g_w[wnr[0],wnr[1],i]+=this.α*(1-this.g_w[wnr[0],wnr[1],i])
            else:
                this.g_w[wnr[0],wnr[1],i]+=this.α*(0-this.g_w[wnr[0],wnr[1],i])

    def __plot2d(this,ax,x,y,winner=None):
        ax[1].cla()
        ax[2].cla()
        ax[0].cla()
        l.plot_data2d(ax[0],y,x[:,0],x[:,1])
        l.plot_data2d(ax[1],y,x[:,0],x[:,1])
        ax[1].plot(this.w[:,:,0],this.w[:,:,1],'wo',markersize=7 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
      ,label='Neurons')
       
        ax[2].set_xlim(x[:,0].min(),x[:,0].max())
        ax[2].set_ylim(x[:,1].min(),x[:,1].max())
        l.plot_lims(ax[2],x[:,0],x[:,1],this,y)
        ax[2].set_title('accuracy : {}'.format(this.score(x,y)))

    def __plotFlower(this,ax,x,y,winner=None):
        ax[1].cla()
        ax[2].cla()
        ax[0].cla()
        preds=this.predict(x)
        l.plot_data2d(ax[0],y,x[:,0],x[:,2])
        l.plot_data2d(ax[1],y,x[:,0],x[:,2])
        ax[1].plot(this.w[:,:,0],this.w[:,:,2],'wo',markersize=7 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
      ,label='Neurons')
       
        ax[2].plot(x[:,0][preds==y],x[:,2][preds==y],'go' , label='Correctly Classified')
        ax[2].plot(x[:,0][preds!=y],x[:,2][preds!=y],'ro' ,label='Wrongly Classified')
        ax[2].set_title('accuracy : {}'.format(this.score(x,y)))
        ax[2].legend()
   

    def __plot3d(this,ax,x,y,winner=None ):
        ax[1].cla()
        ax[2].cla()
        ax[0].cla()
        preds=this.predict(x)
        wx,wy,wz=this._get_coords()
        l.plot_data3d(ax[0],y,x[:,0],x[:,1],x[:,2])
        ax[1].plot(x[:,0][y==preds],x[:,1][y==preds],x[:,2][y==preds] , 'go', label='Correctly Classified')
        ax[1].plot(x[:,0][y!=preds],x[:,1][y!=preds],x[:,2][y!=preds] , 'ro',label='Wrongly Classified')
        ax[1].plot(wx,wy,wz,'wo',markersize=7 ,alpha=0.5, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]
      ,label='Neurons')
       
        l.plot_data2d(ax[2],y,x[:,0],preds)
        ax[2].set_title('accuracy : {}'.format(this.score(x,y)))
     


    
    def __definePlot(this,x):
        if x.shape[1]==2:
            fig,ax=plt.subplots(1,3)
            plot=this.__plot2d
        elif x.shape[1]==3:
            fig=plt.figure(figsize=(8,8))
            ax=np.zeros((3,),dtype=object)
            ax[0]=fig.add_subplot(1, 3, 1, projection='3d')       
            ax[1]=fig.add_subplot(1, 3, 2, projection='3d')
            ax[2]=fig.add_subplot(1, 3, 3,)
            plot=this.__plot3d
        elif x.shape[1]==4:
             fig,ax=plt.subplots(1,3)
             plot=this.__plotFlower
        return ax,plot

    def test(this,X,y):
        ax,plot=this.__definePlot(X)
        plot(ax,X,y)
        plt.show()


    def fit(this,X,y,
                     grossberg_epochs=5000 ,
                     kohonen_epochs=1000   ):
        #this.som_fit(X,y,kohonen_epochs, lambda i:i%100==0)
        this.fitkohonen(X,kohonen_epochs)
        this.g_w=np.random.random_sample(
        (this.k,this.k,len(np.unique(y)))
           )
        ax,plot=this.__definePlot(X)
        α_old=this.α
        this.β=β_old=.1
        for e in range ( grossberg_epochs ):
            for x,d in zip(X,y):
                i,j=this._find_winner(
                     this._ind(x)
                )
                this.__updw(x , (i,j))
                this.__gsb_updw(x,d,(i,j))
                
            
            this.α=α_old*(1-e/grossberg_epochs)
            this.β=β_old*(1-e/grossberg_epochs)
            plot(ax,X,y,(i,j))
            ax[1].set_title('EPOCHS : {}/{}'
            .format(e,grossberg_epochs))
            
            
            plt.pause(.005)

     
    def grossberg(this,X,y,epochs,kohonen_epochs=1000):
        this.fitkohonen(X,epochs=kohonen_epochs)
        this.g_w=np.random.random_sample(
            (this.k,this.k,len(np.unique(y)))
        )
        α_old=this.α
        this.β=β_old=.1
        for e in range ( epochs ):
            
            
            for x,d in zip(X,y):
                i,j=this._find_winner(
                     this._ind(x)
                )
                this.__updw(x , (i,j))
                this.__gsb_updw(x,d,(i,j))
            this.α=α_old*(1-e/epochs)
            this.β=β_old*(1-e/epochs)

    def predict(this,x):
        preds=[]
        for e in x:
            i,j=this._find_winner(
                this._ind(e)
            )
            preds.append(np.argmax(
                this.g_w[i,j]
            ))
        return np.array(preds)
    
    def score(this,X,y):
        ct=0
        for x,d in zip(X,y):
            i,j=this._find_winner(
                     this._ind(x)
                )
            klasi=np.argmax(
                this.g_w[i,j]
            )
            if klasi==d:ct+=1
            
        return np.round(ct/len(y),2)


 
def run():
    import data as dt
    from sklearn.datasets import load_iris
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
   
    model=cpn(int(input('Dwse arithmo kxk neuronon stromatos kohonen : ')))

    k_epochs=int(input('Dwse arithmo epoxon gia to strwma kohonen(protinete 1000) : '))
    g_epochs=int(input('Dwse arithmo epoxon gia to strwma grossberg(kalitera oxi panw apo 50-100) : '))
    
    print('Kohonen layer is being trained please wait...')

    model.fit(xtrain,ytrain,g_epochs,k_epochs)

    input('Press enter to test...')
    model.test(xtest,ytest)

if __name__=='__main__':run()
