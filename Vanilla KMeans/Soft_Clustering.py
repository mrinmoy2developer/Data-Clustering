from cv2 import kmeans
import numpy as np
import random as rnd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets    
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score as ARI

def show2D(X,M,Z,K):
    colors=np.random.random((K,3))  #creates a random numpy array of shape[k,3] filled with iid unif(0,1) rvs
    colors=Z.dot(colors)    #we take a dot product of the colour matrix with the membership matrix
    plt.scatter(X[:,0],X[:,1],c=colors,s=10)
    plt.scatter(M[:,0],M[:,1],marker='^',c='black',s=50)
    plt.show()

def show3D(X,M,Z,K):
    colors=np.random.random((K,3))  #creates a random numpy array of shape[k,3] filled with iid unif(0,1) rvs
    colors=Z.dot(colors)    #we take a dot product of the colour matrix with the membership matrix
    ax=plt.axes(projection ="3d")
    ax.scatter3D(X[:,0],X[:,1],X[:,2],c=colors,s=10)
    ax.scatter3D(M[:,0],M[:,1],M[:,2],marker='^',c='black',s=50)
    plt.show()
    
def show_pcord(X,M,Z,K):
    _,D=X.shape
    colors=np.random.random((K,3))  #creates a random numpy array of shape[k,3] filled with iid unif(0,1) rvs
    colors=Z.dot(colors)    #we take a dot product of the colour matrix with the membership matrix
    P=[predict(M,X[i]) for i in range(X.shape[0])]
    df=pd.DataFrame(X,columns=[a+str(b) for a,b in zip(['x']*D,list(range(1,D+1)))])
    df['Cluster']=P
    pd.plotting.parallel_coordinates(df,'Cluster')
    plt.show()
    
def init(X,K):
    N,_=X.shape
    M=X[rnd.sample(range(N),K)]  #assign any k random pts as the k centres
    return M

def update(X,Z,K):
    _,D=X.shape
    M=np.zeros((K,D))
    for k in range(K):
        M[k]=Z[:,k].dot(X)/Z[:,k].sum()   #new centre is the weighted avg of all data pts with membership as weights
    return M

def cost(X,Z,M,K):
    obj=0
    for k in range(K):
        norm=np.linalg.norm(X-M[k],2)   #computes the Frobenius norm of the origin shifted data matrix
        obj+=(norm*np.expand_dims(Z[:,k],axis=1)).sum()  
    return obj

def membership(M,X):
    N,_=X.shape
    K,_=M.shape
    Z=np.zeros((N,K))
    for n in range(N):        
        Z[n]=np.exp(-np.linalg.norm(M-X[n],2,axis=1)) #computes the inverse exponential of L2 norms rowise of the origin shifted data matrix
        # Z[n]=np.linalg.norm(M-X[n],2,axis=1)
    Z/=Z.sum(axis=1,keepdims=True)
    return Z

# def membership(M,X,m=2):
#     N,_=X.shape
#     K,_=M.shape
#     Z=np.zeros((N,K))
#     for i in range(N):
#         for j in range(K):
#             Z[i,j]=1/sum([(np.linalg.norm(X[i]-M[j],2)/np.linalg.norm(X[i]-M[k],2))**(2/(m-1)) for k in range(K)])
#     return Z

def soft_kmeans(X,K,max_iters=100,tol=1e-5,plot=1):
    M=init(X,K)
    lcost=0
    for _ in range(max_iters):
        Z=membership(M,X)     #expectation step
        M=update(X,Z,K)            #maximization step
        pcost=cost(X,Z,M,K)         #computes the likelihood
        if np.abs(pcost-lcost)<tol:    #stop if clusters stabilize 
            break
        lcost=pcost
    if plot:
        if X.shape[1]==2:
            show2D(X,M,Z,K)
        elif X.shape[1]==3:
            show3D(X,M,Z,K)
        else:
            show_pcord(X,M,Z,K)
    return M,Z

def predict(M,Y):
    D=outer(Y,M)
    P=D.argmin(axis=1)
    return P

# def ari(L,P):
    
    
def main():
    # """Generate data from gaussian mixture models and cluster them or use the iris dataset"""
    # x1=np.random.multivariate_normal([1,2],[[1,0],[0,2]],60)
    # x2=np.random.multivariate_normal([5,6],[[4,0],[0,4]],80)
    # x3=np.random.multivariate_normal([10,4],[[3,0],[0,1]],100)
    # X=np.vstack([x1,x2,x3])
    # idx=np.random.rand(X.shape[0]).argsort()
    # X=np.take(X,idx,axis=0)
    # L=[0]*60+[1]*80+[2]*100
    # L=list(np.take(L,idx,axis=0))
    #######################################Iris Dataset######################################
    iris=datasets.load_iris()
    X=iris.data
    L=list(iris.target)
    #################################Test Run#######################################
    M,Z=soft_kmeans(X,3)
    P=predict(X,M)
    # print(Z)
    print(f'P:-\n{P}')
    print(f'L:-\n{L}')
    print(f'M:-\n{M}')
    print(f"Adjusted Random Index is {ARI(L,P)}")
    ############################Determining the no. of clusters#########################
    # ari=[]
    # for k in range(2,4):
    #     m,_=soft_kmeans(X,k)
    #     p=[predict(m,X[i]) for i in range(X.shape[0])]
    #     ari.append(ARI(L,p))
    # print(ari)
    # plt.plot(range(10),ari)
    # plt.show()
    #####################################Using In-built function###########################
    # model=kmeans.fit(X)
    # fig,ax1,ax2=plt.subplot(1,2)
    df2=pd.DataFrame(X,columns=[a+str(b) for a,b in zip(['x']*4,list(range(1,4+1)))])
    df2['Cluster']=P
    plt.subplot(1,2,2)
    pd.plotting.parallel_coordinates(df2,'Cluster')
    df1=pd.DataFrame(iris.data,columns=iris.feature_names)
    df1['Flowers']=[iris.target_names[0] if i==0 else iris.target_names[1] if i==1 else iris.target_names[2] for i in iris.target]
    plt.subplot(1,2,1)
    pd.plotting.parallel_coordinates(df1,'Flowers')
    plt.show()

if __name__=="__main__":
    main()