import numpy as np


sgm_d=lambda u: (np.exp(-u)/((1+np.exp(-u))**2))
sgmd=lambda u: 1/(1+np.exp(-u))    

tanh= lambda u:(np.exp(u)-np.exp(-u))/(np.exp(u)+np.exp(-u))
tanh_d =lambda u: 1-tanh(u)**2

step= lambda u: 1 if u>0 else 0

step_1= lambda u: ({True:1,False:-1} [u>0])

def normalize(x,y):
    x=x/np.max(x)
    y=y/np.max(y)
    return x,y
def θ_S(u):
    return 1 if u>0.5 else 0

add_bias= lambda X: np.hstack([-1*np.ones((len(X),1)),X])

gauss = lambda x,c,σ : np.exp(-( (np.linalg.norm(x-c)**2) / (σ**2) )  )

cauchy= lambda x,c,σ : 1/σ*((np.linalg.norm(x-c)**2 +σ**2))

mltqdrt= lambda x,c,σ : np.sqrt((np.linalg.norm(x-c)**2)+ (σ**2) )

def rbf_pack(X,c,σ,type):
    new_x=np.zeros((X.shape[0],len(c)))
    
    if type =='Gauss':
            f=gauss
    elif type=='Cauchy':
            f=cauchy
    elif type=='Multiquadric':
            f=mltqdrt
    else : raise ValueError('Invalid Input')
    for i in range(X.shape[0]):
        for j in range(len(c)):
            new_x[i,j]= f(X[i],c[j],σ) 
    return np.array(new_x)



class f():
    def __init__(this,f):
        
        if f=='sigmoid': 
                this.a=sgmd
                this.d=sgm_d
                this.goal=0
        elif f=='tanh':
                this.a=tanh
                this.d=tanh_d
                this.goal=-1
            #!#
        elif f=='x=y':
                this.a= lambda a : a
                this.d= lambda a : 1
                this.goal = -1
        else : raise  ValueError('Invalid Input')