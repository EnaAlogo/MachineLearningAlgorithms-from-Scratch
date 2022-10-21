import numpy as np

def xor(n):
    c0_1=(0.3)*np.random.random_sample((n//4,2))
    c0_2=(0.9-0.7)*np.random.random_sample((n//4,2))+0.7
    c1=np.append(c0_1,c0_2,axis=0)
    tmp1=(0.9-0.7)*np.random.random_sample((n//4,))+0.7
    tmp2=(0.3)*np.random.random_sample((n//4,))
    c1_1=np.vstack((tmp1,tmp2)).T
    tmp1=(0.3)*np.random.random_sample((n//4,))
    tmp2=(0.9-0.7)*np.random.random_sample((n//4,))+0.7
    c1_2=np.vstack((tmp1,tmp2)).T
    c2=np.append(c1_1,c1_2,axis=0)
    y1=np.zeros(len(c1))
    y2=np.ones(len(c2))
    return np.append(c1,c2,axis=0),np.append(y1,y2)


def angular(n):
    X1=(0.3)*np.random.random_sample((n//2,2))
    tmp1=(0.3)*np.random.random_sample((n//4,))
    tmp2=(0.9-0.4)*np.random.random_sample((n//4,))+0.4
    class1_0=np.append(tmp1,tmp2,axis=0)
    tmp1=(0.9-0.4)*np.random.random_sample((n//4,))+0.4
    tmp2=(0.9)*np.random.random_sample((n//4,))
    class1_1=np.append(tmp1,tmp2,axis=0)
    X2=np.vstack((class1_0,class1_1)).T
    y1=np.zeros(len(X1))
    y2=np.ones(len(X2))
    return np.append(X1,X2,axis=0),np.append(y1,y2)

def ciricular(n):
    X1=(0.6-0.4)*np.random.random_sample((n//2,2))+0.4
    tmp1=(0.9)*np.random.random_sample((n//8,))
    tmp2=(0.3)*np.random.random_sample((n//8,))
    class1_0=np.vstack((tmp1,tmp2)).T
    tmp1=(0.9)*np.random.random_sample((n//8,))
    tmp2=(0.9-0.7)*np.random.random_sample((n//8,))+0.7
    class1_1=np.vstack((tmp1,tmp2)).T
    tmp1=(0.3)*np.random.random_sample((n//8,))
    tmp2=(0.9)*np.random.random_sample((n//8,))
    class1_2=np.vstack((tmp1,tmp2)).T
    tmp1=(0.9-0.7)*np.random.random_sample((n//8,))+0.7
    tmp2=(0.9)*np.random.random_sample((n//8,))
    class1_3=np.vstack((tmp1,tmp2)).T
    X2_1=np.append(class1_0,class1_1,axis=0)
    X2_2=np.append(class1_2,class1_3,axis=0)
    X2_f=np.append(X2_1,X2_2,axis=0)
    y1=np.zeros(len(X1))
    y2=np.ones(len(X2_f))
    return np.append(X1,X2_f,axis=0),np.append(y1,y2)

def l_sep(n):
    X1=(0.3)*np.random.random_sample((n//2,2))
    X2=(0.9-0.7)*np.random.random_sample((n//2,2))+0.7
    y1=np.zeros(len(X1))
    y2=np.ones(len(X2))
    X=np.append(X1,X2,axis=0)
    y=np.append(y1,y2,axis=0)
    return X,y

def non_sep(n,type):
   
    if type=='xor':X,y=xor(n)
    elif type=='circ':X,y=ciricular(n)
    elif type=='ang':X,y=angular(n)
    return X,y


def l_sep3d(n):
    X1=(0.3)*np.random.random_sample((n//2,3))
    X2=(0.9-0.7)*np.random.random_sample((n//2,3))+0.7
    y1=np.zeros(len(X1))
    y2=np.ones(len(X2))
    X=np.append(X1,X2,axis=0)
    y=np.append(y1,y2,axis=0)
    return X,y

def xor_3d(n):
    c1_0=(0.3)*np.random.random_sample((n//4,3))
    c1_1=(0.9-0.7)*np.random.random_sample((n//4,3))+0.7
    c1=np.append(c1_0,c1_1,axis=0)
    xy=(0.9-0.7)*np.random.random_sample((n//4,2))+0.7
    z=(0.3)*np.random.random_sample((1,n//4))
    c2_0=np.append(xy,z.T,axis=1)
    xy=(0.3)*np.random.random_sample((n//4,2))
    z=(0.9-0.7)*np.random.random_sample((1,n//4))+0.7
    c2_1=np.append(xy,z.T,axis=1)
    c2=np.append(c2_0,c2_1,axis=0)
    y1=np.zeros(len(c1))
    y2=np.ones(len(c2))
    return np.append(c1,c2,axis=0),np.append(y1,y2,axis=0)

