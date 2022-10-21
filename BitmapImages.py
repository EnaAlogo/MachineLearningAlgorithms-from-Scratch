
import numpy as np
import matplotlib.pyplot as plt

class bmpimgs:
    def __init__(this) -> None:
        this.__m_eight()
        this.__m_five()
        this.__m_nine()
        this.__m_six()
        this.__names={0:'five',
        1:'six',2:'eight',3:'nine'}

    def sampling(this):
        w=[]
        w.append(this.__five)
        w.append(this.__six)
        w.append(this.__eight)
        w.append(this.__nine)
        return np.array(w)
        
    def __distortion(this,x,times):
        rn=np.zeros((times,77))
        rn[0]=np.copy(np.reshape(x,77))
        for t in range(1,times):
          p=np.copy(np.reshape(x,77))
          for i in range (5):
              i=np.random.randint(0,77)
              p[i]*=-1
          rn[t]=np.copy(p)
        return np.array(rn)

    def makeimgs(this,numeach=5): 
        this.n=numeach
        fiw=this.__distortion(this.__five,numeach)
        sew=this.__distortion(this.__six,numeach)
        oiw=this.__distortion(this.__eight,numeach)
        niw=this.__distortion(this.__nine,numeach)
        y=np.zeros(numeach)
        y1=np.ones(numeach)
        y2=2*np.ones(numeach)
        y3=3*np.ones(numeach)
        y=np.append(y,y1,axis=0)
        y=np.append(y, np.append(y2,y3,axis=0),axis=0)
        x=np.append(fiw,sew,axis=0)
        x=np.append(x, np.append(oiw,niw,axis=0) ,axis=0)
        data=np.hstack((x,y.reshape(y.shape[0],1)))
        np.random.shuffle(data)
        x=data[:,:-1]
        y=data[:,-1]
        return x,y

    def get_perfects(this)->tuple:
        return this.__five,\
               this.__six,\
               this.__eight,\
               this.__nine 

    def __m_eight(this):
        oit=np.ones((11,7))
        oit[1,1:-1]=-1
        oit[2:-1,1]=-1
        oit[5,1:-1]=-1
        oit[1:-1,-2]=-1
        oit[-2,1:-1]=-1
        this.__eight=oit

    def __m_nine(this):
        noin=np.ones((11,7))
        noin[1,1:-1]=-1
        noin[2:6,1]=-1
        noin[5,1:-1]=-1
        noin[1:-1,-2]=-1
        noin[-2,1:-1]=-1
        this.__nine=noin

    def __m_five(this):
        fahve= np.ones((11,7))
        fahve[1,1:-1]=-1
        fahve[2:6,1]=-1
        fahve[5,1:-1]=-1
        fahve[6:-1,-2]=-1
        fahve[-2,1:-1]=-1
        this.__five=fahve
  
    def __m_six(this):
        sex=np.ones((11,7))
        sex[1,1:-1]=-1
        sex[2:-1,1]=-1
        sex[5,1:-1]=-1
        sex[6:-1,-2]=-1
        sex[-2,1:-1]=-1
        this.__six=sex
        
    className =lambda this,n:this.__names[n]
    
    @staticmethod
    def plotImgs(x,d,preds,clf):
        n=clf.n//2
        fig,ax=plt.subplots(4,n)
        fig.tight_layout()
        i,j=0,0
        for p ,g ,prd in zip(x,d,preds):
            ax[i,j].imshow(p.reshape((11,7)),cmap=plt.cm.gray)
            ax[i,j].set_title('class : {} \n predicted : {}'.format(
               clf.className(g),clf.className(prd)),color='green' if g==prd else 'red')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            if j==n-1:
                j=0;i+=1
            else : j+=1
            


