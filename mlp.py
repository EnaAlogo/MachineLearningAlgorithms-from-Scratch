

import numpy as np
import functions as fn
import matplotlib.pyplot as plt
class mlp:
  
    def __init__(this,β, hidden_layer_size , activation_function):
        this.β=β
        this.trained=False
        this.activation_function=activation_function
        this.layer_size=hidden_layer_size

    def __init_components(this ,X, y):
        classes=np.unique(y)
        n_classes=len(classes)
        this.hidden=layer(X.shape[1],this.layer_size,this.β)
        this.hidden.setfunc(fn.f(this.activation_function))
        if n_classes==2:
            this.f=fn.f('tanh' if classes[0]==-1 else 'sigmoid')
            this.w=(0.02+0.02)*np.random.random_sample((this.layer_size+1,))-0.02

            this.__back_propagation=this.__train_2_classes
        else:
            this.output=layer(this.layer_size,n_classes,this.β)
            this.output.setfunc(fn.f('sigmoid')) 
            
            this.__back_propagation=this.__train_multiple_classes
   
    def fit_noplot(this, X ,y , epochs):
        if not this.trained: 
            this.__init_components(X,y)
            this.trained=True
        data= fn.add_bias(X)
        for e in range(epochs):
            mean_sqr_error,predictions=this.partial_fit(data,y)
            print(np.round(mean_sqr_error , decimals=3))
            

    def partial_fit(this ,X, y) :  
        data=fn.add_bias(np.copy(X))
        error=[]
        predictions=[]
        for x,d in zip(data,y):
            err,pred=this.__back_propagation(x,d)
            error.append(err)
            predictions.append(pred)
        return np.mean(np.array(err)),np.array(predictions)
 


    def setlr(this,β):
        this.hidden.β=β
        try:
            this.output.β=β
        except:
            this.β=β

    def __deltas(this,v,u,d):
        return (d-v)*this.output.f.d(u)
        
    def __train_2_classes(this,x,d):

        v_hidden,u_hidden=this.hidden.output(x,True)
        u_out=np.dot(this.w , v_hidden)
        v_out=this.f.a(u_out)
     
        delta=(d-v_out)*this.f.d(u_out)
        hidden_deltas=[]
        for w,u in zip(this.w[1:],u_hidden):
            hidden_deltas.append((delta*w)*this.hidden.f.d(u))
        hidden_deltas=np.array(hidden_deltas)            
        for ct in range(len(this.w)):
            this.w[ct]+=this.β*delta*v_hidden[ct]
        this.hidden.correct(x, hidden_deltas)
        return ((d-v_out)**2),v_out

    def __train_multiple_classes(this,x,d):
        
        v_hidden , u_hidden = this.hidden.output(x,True)
        v_out , u_out = this.output.output(v_hidden,False)
        
        deltas=[]
        neural_sum=[]
        hidden_deltas=[]
        for δ in range(len(this.output.nr)):
            if(δ==d):
                deltas.append(this.__deltas(v_out[δ],u_out[δ],1))
                neural_sum.append([((1-v_out[δ])**2)])
            else:
                deltas.append(this.__deltas(v_out[δ],u_out[δ],0))
                neural_sum.append([((0-v_out[δ])**2)])
        for hn in range(len(this.hidden.nr)):
            Σ=0
            for on in range(len(this.output.nr)):
                Σ+=deltas[on]*this.output.nr[on].w[hn+1]
            hidden_deltas.append(Σ*this.hidden.f.d(u_hidden[hn]))
        this.output.correct(v_hidden, deltas)
        this.hidden.correct(x, hidden_deltas)
        return np.sum(neural_sum), v_out

    def predict(this,X):
        preds=[]
        X=fn.add_bias(np.copy(X))
        for x in X:
            u=this.hidden.predict(x,bias=True)
            try:
               v=np.argmax(this.output.predict(u,bias=False) )
            except AttributeError:
                v=this.f.a(np.dot(this.w,u))
            preds.append(v)
        return np.array(preds)

    def test(this,x,y ,myimg=None):
        import matplotlib.pyplot as plt
        this.__testNtype(x,y ,myimg)
        plt.legend()
        plt.show()

    def __testNtype(this,X,y ,myimg):
        import lines as l
        import matplotlib.pyplot as plt
        p=[]
        if X.shape[1]==2:
            fig,ax=plt.subplots(3,1)
            plot=l.plot02d
            p=this.predict(X)
            plot(ax,X,y,p,this.hidden.nr[0].w)
            for i in range (1 , len(this.hidden.nr)):
               xs,yeq=l.line(X,this.hidden.nr[i].w)
               ax[1].plot(xs,yeq,'b-')
            
        elif X.shape[1]==3:
            fig=plt.figure()
            ax=np.zeros((3,),dtype=object)
            ax[0]=fig.add_subplot(1, 3, 1, projection='3d')
            ax[1]=fig.add_subplot(1, 3, 2, projection='3d')
            ax[2]=fig.add_subplot(1, 3, 3, projection='3d')
            plot=l.plot03d
            p=this.predict(X)
            plot(ax,X,y,p,this.hidden.nr[0].w)
            for i in range (1 , len(this.hidden.nr)):
               xs,yeq,zeq=l.surface(X,this.hidden.nr[i].w)
               ax[1].plot_surface(xs,yeq,zeq,cmap='viridis')
        elif X.shape[1]==4:
            fig,ax=plt.subplots(3,1)
            l.plot_data2d(ax[0],y,X[:,0],X[:,2] ,'Original Data')
            l.plot_data2d(ax[1],this.predict(X),X[:,0],X[:,2] ,'Classified Data')
            ax[2].plot(X[:,0][this.predict(X)==y],X[:,2][this.predict(X)==y],'go'
            ,label='classifed correctly')
            ax[2].plot(X[:,0][this.predict(X)!=y],X[:,2][this.predict(X)!=y],'r*'
            ,label='classifed wrongly')
            ax[2].set_title('Classified vs Original comparison')
        elif X.shape[1]==77:
            from BitmapImages import bmpimgs
            bmpimgs.plotImgs(X,y,this.predict(X),myimg)


    def __setupAxes(this,x):
        import lines as l
        if x.shape[1]==2:
           fig,ax=plt.subplots(2,2)
           plot=l.plotmlp2d
        else:
           fig=plt.figure()
           ax=np.zeros((2,2),dtype=object)
           ax[0,0]=fig.add_subplot(2, 2, 1, projection='3d')
           ax[0,1]=fig.add_subplot(2, 2, 2, projection='3d')
           ax[1,0]=fig.add_subplot(2, 2, 3)
           ax[1,1]=fig.add_subplot(2, 2, 4)
           plot=l.plotmlp3d
        return ax,plot

    def train_with_plots(this,xtrain,ytrain,epochs,imgs=None,cond=None):
        if not this.trained: 
            this.__init_components(xtrain,ytrain)
            this.trained=True
        if xtrain.shape[1]==77:
            this.__images(xtrain,ytrain,epochs,imgs) 
        elif xtrain.shape[1]==4:
            this.__flowers(xtrain,ytrain,epochs,cond)
        else:
            this.__train_plots(xtrain,ytrain,epochs,cond)  


    def __flowers(this,xtrain,ytrain , epochs , cond ):
        line=[]
        import lines as l
        fig,ax=plt.subplots(2,2)
        for i in range(epochs):
            err,preds=this.partial_fit(xtrain,ytrain)
            line.append(err)
            ax[0,1].cla()
            ax[1,1].cla()
            ax[1,0].cla()
            if cond(i):
                l.plot_flowers_mlp(ax,xtrain,ytrain,err,line,this.predict(xtrain))
                plt.pause(.005)
    
    def __train_plots(this,xtrain,ytrain,epochs , cond ):

        line=[]

        ax,plot = this.__setupAxes(xtrain)
   
        for i in range(epochs):
            err,preds=this.partial_fit(xtrain,ytrain)
            line.append(err)
            if cond(i):
               plot(ax,ytrain,xtrain,np.array(preds),this.hidden.nr,err,line)
               plt.pause(0.005)

    def __images(this ,xtrain ,ytrain , epochs , imgs):
        line=[]
        import lines as l
        from BitmapImages import bmpimgs
        for i in range (epochs):
                this.partial_fit(xtrain,ytrain)
                err=np.mean((ytrain-this.predict(xtrain))**2)
                line.append(err)
          
                l.plot_minsquared_errror(plt,line,err)
                plt.pause(.0005)
        #bmpimgs.plotImgs(xtest,ytest,curr.predict(xtest),imgs)
        #plt.show()
       

class layer:
  
    def __init__(this,n,n_neur,β):
        this.n=n
        this.β=β
        n_nr=[]
        for i in range(n_neur):
           n_nr.append(neuron(this.n))
        this.nr=np.array(n_nr)

    def correct(this,x,δ):
        for neuron,deltas in zip(this.nr,δ):
            for i in range(len(neuron.w)):
                neuron.w[i]+=this.β*deltas*x[i]

    def predict(this,x,bias):
        v=[]
        if bias:v.append(-1)

        for n in this.nr:
            v.append(this.f.a(n.Σ(x)))
        return np.array(v)
    def output(this , x , bias):
        v=[]
        u=[]
        if bias:v.append(-1)
        for n in this.nr:
            u0=n.Σ(x)
            u.append(u0)
            v.append(this.f.a(u0))
        return np.array(v),np.array(u)
    def setfunc(this,f):
        this.f=f
    def u(this,x):
        v=[]
        for n in this.nr:
            v.append(n.Σ(x))
        return np.array(v)      
class neuron:
    def __init__(this,n):
        this.w=(0.02+0.02)*np.random.random_sample((n+1,))-0.02
    def Σ(this,x):
     return np.dot(this.w,x)


