from matplotlib.axes import Axes
import numpy as np
import matplotlib.patheffects as pe
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

clr=np.array(['c^','ms','y*','g+'])
def line(x,w):
     x1 = np.amin(x)
     x2 = np.amax(x)
     Yline=-(w[1]/w[2])*x1+w[0]/w[2]
     Ysus=-(w[1]/w[2])*x2+w[0]/w[2]
     return [x1,x2],[Yline,Ysus]

def surface(x,w):
      x1=np.linspace(np.min(x[:,1]),np.max(x[:,1]))
      y=np.linspace(np.min(x[:,2]),np.max(x[:,2]))
      x1,y=np.meshgrid(x1,y)
      z=-(w[1]*x1+w[2]*y-w[0])/w[3]
      return x1,y,z

def plot_minsquared_errror(ax,line,errplot):
    ax.plot(line,'r')
    ax.legend(['MEAN\nSQUARED\nERROR: '+str(np.round(errplot,decimals=4))],fontsize=13,loc=3)

def plot_data2d(ax,d,x,y , title=''):
      ax.clear()
      gc=len(np.unique(d))
      goals=np.unique(d)
      for i in range(gc):
            ax.plot(x  [d==goals[i]], y [d==goals[i]],clr[i]
            ,label='class :{}'.format(i))
      ax.set_title(title)
      ax.legend()

def plot_data3d(ax,d,x,y,z , title=''):
      ax.clear()
      gc=len(np.unique(d))
      goals=np.unique(d)
      for i in range(gc):
            ax.plot(x [d==goals[i]],y  [d==goals[i]],z [d==goals[i]],clr[i]
            ,label='class :{}'.format(i))
      ax.set_title(title)
      ax.legend()
      



def plotmlp3d(ax,d,x,preds,nrs,err,l):
      
      ax[0,0].cla()
      ax[1,1].cla()
      ax[0,1].cla()
      ax[1,0].cla()
      plot_data3d(ax[0,0],d,x[:,0],x[:,1],x[:,2] ,'Original Data')
      plot_data2d(ax[1,0],d,x[:,0],preds , 'Output Graph')
       
      plot_data3d(ax[0,1],d,x[:,0],x[:,1],x[:,2] )
      for k in nrs:
         xs,ys,zeq=surface(x,k.w)
         ax[0,1].plot_surface(xs,ys,zeq,cmap='viridis')
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1])) 
      ax[0,1].set_zlim(np.min(x[:,2]),np.max(x[:,2]))
      plot_minsquared_errror(ax[1,1],l,err)

def plotmlp2d(ax,d,x,preds,nrs,err,l):
      ax[0,0].cla()
      ax[1,1].cla()
      ax[0,1].cla()
      ax[1,0].cla()
      plot_data2d(ax[0,0],d,x[:,0],x[:,1] ,'Original Data')
      plot_data2d(ax[1,0],d,x[:,0],preds ,'Ouput Graph')
      
      plot_data2d(ax[0,1],d,x[:,0],x[:,1])  
      for k in nrs:
          xs,yeq=line(x,k.w)
          ax[0,1].plot(xs,yeq,'b')
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1])) 
      plot_minsquared_errror(ax[1,1],l,err)

def plot12d(ax,x,d,preds,w,err,l):
      xs,yeq=line(x,w)
      ax[0,0].cla()
      ax[1,1].cla()
      ax[0,1].cla()
      ax[1,0].cla()
      plot_data2d(ax[0,0],d,x[:,0],x[:,1]  ,'Original Data')
      plot_data2d(ax[1,0],d,x[:,0],preds  ,'Ouput Graph')
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1])) 
      plot_data2d(ax[0,1],d,x[:,0],x[:,1])  
      ax[0,1].plot(xs,yeq,'b')
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1])) 
      plot_minsquared_errror(ax[1,1],l,err)

def plot13d(ax,x,d,preds,w,err,l):
      ax[0,0].cla()
      ax[1,1].cla()
      ax[0,1].cla()
      ax[1,0].cla()
      plot_data3d(ax[0,0],d,x[:,0],x[:,1],x[:,2] ,'Original Data')
      plot_data2d(ax[1,0],d,x[:,0],preds  ,'Ouput Graph')
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1]))
      ax[0,1].set_zlim(np.min(x[:,2]),np.max(x[:,2]))  
      plot_data3d(ax[0,1],d,x[:,0],x[:,1],x[:,2]) 
      xs,yeq,zeq=surface(x,w) 
      ax[0,1].plot_surface(xs,yeq,zeq)
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1]))
      ax[0,1].set_zlim(np.min(x[:,2]),np.max(x[:,2])) 
      plot_minsquared_errror(ax[1,1],l,err)

def plot02d(ax,x,d,preds,w ,epochs='TEST' , epoch=''):
      ax[0].cla()
      ax[1].cla()
      ax[2].cla()
      plot_data2d(ax[0],d,x[:,0],x[:,1] ,'Original Data')
           
      xs,yeq=line(x[:,0],w)
      plot_data2d(ax[1],d,x[:,0],x[:,1] ,'EPOCHS : {}/{}'.format(epoch,epochs))
      ax[1].plot(xs,yeq,'b')
      ax[1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[1].set_ylim(np.min(x[:,1]),np.max(x[:,1]))
      plot_data2d(ax[2],d,x[:,0],preds ,'Ouput Graph')

def plot03d(ax,x,d,preds,w , epochs='TEST', epoch=''):
      ax[0].cla()
      ax[1].cla()
      ax[2].cla()
      plot_data3d(ax[0],d,x[:,0],x[:,1],x[:,2] ,'Original Data')
      
    
      xs,ys,zeq=surface(x,w)
      plot_data3d(ax[1],d,x[:,0],x[:,1],x[:,2],'EPOCHS : {}/{}'.format(epoch,epochs))
      ax[1].plot_surface(xs,ys,zeq)
      ax[1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[1].set_ylim(np.min(x[:,1]),np.max(x[:,1]))
      ax[1].set_zlim(np.min(x[:,2]),np.max(x[:,2]))
      plot_data3d(ax[2],d,x[:,0],preds,x[:,2] ,'Ouput Graph')

def plotkmn(ax,x,d,preds,ctr,err,l):
      ax[0,0].cla()
      ax[1,1].cla()
      ax[0,1].cla()
      ax[1,0].cla()
      plot_data2d(ax[0,0],d,x[:,0],x[:,1] ,'Original Data')
      plot_data2d(ax[1,0],d,x[:,0],preds ,'Ouput Graph')
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1])) 
      prediction_map=np.where(preds>0,1,-1)
      plot_data2d(ax[0,1],prediction_map,x[:,0],x[:,1],'Classifed Data')
      ax[0,1].plot(ctr[:,0],ctr[:,1],'wo',markersize=15,lw=2,alpha=0.4, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()] 
      ,label='Centroids')
      ax[0,1].legend()
      plot_minsquared_errror(ax[1,1],l,err)
      
      
def plotkmn3d(ax,x,d,preds,ctr,err,l):
      ax[0,0].cla()
      ax[1,1].cla()
      ax[0,1].cla()
      ax[1,0].cla()
      plot_data3d(ax[0,0],d,x[:,0],x[:,1],x[:,2] ,'Original Data')
      plot_data2d(ax[1,0],d,x[:,0],preds ,'Ouput Graph')
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1])) 
      ax[0,1].set_zlim(np.min(x[:,2]),np.max(x[:,2])) 
      prediction_map=np.where(preds>0,1,-1)
      plot_data3d(ax[0,1],prediction_map,x[:,0],x[:,1],x[:,2],'Classifed Data')
      ax[0,1].plot(ctr[:,0],ctr[:,1],ctr[:,2],'wo',markersize=15,lw=2,alpha=0.4, path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()] 
      ,label='Centroids')
      ax[0,1].legend()
      plot_minsquared_errror(ax[1,1],l,err)

      
def SupportVectors2d(ax,clf:SVC,x,y,s):
    plot_data2d(ax,y,x[:,0],x[:,s])
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    margin = 1 / np.linalg.norm(clf.coef_)
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin
    ax.plot(xx, yy, "b-")
    ax.plot(xx, yy_down, "k--")
    ax.plot(xx, yy_up, "k--")
    ax.set_xlim(np.min(x[:,0]),np.max(x[:,0]))
    ax.set_ylim(np.min(x[:,s]),np.max(x[:,s]))
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, s],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
        cmap=cm.get_cmap("RdBu"),
    )
def SupportVector3d(ax,clf:SVC,x,y):
    plot_data3d(ax,y,x[:,0],x[:,1],x[:,2])

    x1=np.linspace(np.min(x[:,1]),np.max(x[:,1]))
    y1=np.linspace(np.min(x[:,2]),np.max(x[:,2]))
    x1,y1=np.meshgrid(x1,y1)
    zeq=(-clf.intercept_[0]-clf.coef_[0][0]*x1 -clf.coef_[0][1]*y1) / clf.coef_[0][2]
    ax.plot_surface(x1,y1,zeq,cmap='magma')
    
    w = clf.coef_[0]
    a = -w[0] / w[1]

    margin = 1.3 / np.linalg.norm(clf.coef_)

    down=zeq-np.sqrt(1+a**2)*margin
    up=zeq+np.sqrt(1+a**2)*margin
    ax.plot_surface(x1,y1,down,cmap='viridis')
    ax.plot_surface(x1,y1,up,cmap='viridis')
    ax.set_xlim(np.min(x[:,0]),np.max(x[:,0]))
    ax.set_ylim(np.min(x[:,1]),np.max(x[:,1]))
    ax.set_zlim(np.min(x[:,2]),np.max(x[:,2]))
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        clf.support_vectors_[:,2],
        s=80,
        facecolors="none",
        zorder=10,
        edgecolors="k",
        cmap=cm.get_cmap("RdBu"),
    )
    
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax:Axes, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, **params)


def plot_lims(ax,x,y,clf,d):
      xx,yy=make_meshgrid(x,y)
      plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
      ax.scatter(x, y, c=d, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
      ax.set_title('Descision Boundary')

def plotsckitMLP(ax,x,y,model:MLPClassifier,line=[],err=0,test=True):
      plot_data2d(ax[1,0],y,x[:,0],model.predict(x) ,'Ouput Graph')
      plot_data2d(ax[0,0],y,x[:,0],x[:,1] ,'Original Data')
      if not test:plot_minsquared_errror(ax[1,1],line,err)
      ax[0,1].set_xlim(np.min(x[:,0]),np.max(x[:,0]))
      ax[0,1].set_ylim(np.min(x[:,1]),np.max(x[:,1]))
      plot_lims(ax[0,1],x[:,0],x[:,1],model,y)

def plotRegress(clf,x,y):
    g=[]
    for i,j in enumerate(y):
        g.append(i)
    err=np.mean((y-clf.predict(x))**2)
    plt.plot(g,y,'bo',markersize=7,label='Goals')
    plt.plot(g,clf.predict(x),'ro',markersize=4,label='Predictions')
    plt.title(label='MSE : {}'.format(np.round(err,decimals=2)))
    plt.legend()

def plot_flowers_mlp(ax,x,y,err,line,predictions):
      plot_data2d(ax[0,0],y,x[:,0],x[:,2] ,'Original Data')
      plot_minsquared_errror(ax[1,1],line,err)
      plot_data2d(ax[0,1],predictions,x[:,0],x[:,2] ,'Classifed Data')
      plot_data2d(ax[1,0],y,x[:,0],predictions, 'Output Graph')
    