#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sympy import diff,log,symbols,exp,simplify
from sympy.utilities.lambdify import lambdify

def w_FMT(eps):
    #d=cal_eff_diameter (eps)
    R = 1.0/2
    k=np.linspace(0,N//2,N//2+1)*2*np.pi/L
    w0=2*np.cos(k*R)/2
    k[0]=1 #keep notebook shutup
    w1=2*np.sin(k*R)/k
    w1[0]=2*R
    return w0,w1;

def cal_n(rho,w):
    return np.fft.irfft(np.fft.rfft(rho)*w)

def cal_c1_FMT(rho,eps=0):
    w0,w1 = w_FMT(eps)
    n0=cal_n(rho,w0)
    n1=cal_n(rho,w1)
    F0=-np.log(1-n1)
    F1=n0/(1-n1)
    return np.zeros(len(rho))


def load_data(path_to_file,max_train=(2**31-1)):
    f=np.loadtxt(path_to_file+"parameter.dat")
    global N,L
    L = float(f[0])
    N = int(f[1])
    dx=L/N
    input_shape = (N,1)
    f = open(path_to_file+'rho_z.dat', 'r')
    MC_inform = f.read().splitlines()
    for i in range(len(MC_inform)):
        #print(MC_inform[i])
        MC_inform[i]=MC_inform[i].split("\t")
    f.close()
    MC_inform =MC_inform[:min(len(MC_inform),max_train)] 
    batch_size = len(MC_inform)
    return L,N,dx,batch_size,MC_inform



def train_val_data(path_to_file,N,MC_inform,test_size=0.1,random_state=42):
    rho= []
    c1_HR = []
    Vext= []
    mu = []
    eps= []
    batch_size = len(MC_inform)
    for i in range (batch_size):
        Vext +=[np.loadtxt(path_to_file+"Vext_"+str(i)+".dat")]
        rho += [np.loadtxt(path_to_file+"rho_"+str(i)+".dat")]
        c1_HR+=[cal_c1_FMT(rho[i])]
        mu += [np.log(np.float(MC_inform[i][2]))]
        eps += [0]

    # flip data to increase traiing size     
    for i in range (batch_size):
        Vext +=[np.flip(np.loadtxt(path_to_file+"Vext_"+str(i)+".dat"))]
        rho += [np.flip(np.loadtxt(path_to_file+"rho_"+str(i)+".dat"))]
        c1_HR+=[cal_c1_FMT(rho[batch_size+i])]
        mu += [np.log(np.float(MC_inform[i][2]))]
        eps += [0]
        
    rho_train=np.asarray(rho)
    c1_HR_train=np.asarray(c1_HR)
    Vext_train=np.asarray(Vext)
    eps_train = np.asarray(eps)
    mu_train = np.asarray(mu)
    deltaF_train = np.asarray(np.zeros(rho_train.shape))
    if(test_size==0):
        return rho_train,Vext_train,mu_train

    rho_train = rho_train.reshape(rho_train.shape[0], N)
    c1_HR_train = c1_HR_train.reshape(c1_HR_train.shape[0], N)
    Vext_train = Vext_train.reshape(Vext_train.shape[0], N)
    eps_train = eps_train.reshape(eps_train.shape[0])
    mu_train = mu_train.reshape(mu_train.shape[0])
    deltaF_train=deltaF_train.reshape(deltaF_train.shape[0], N)

    rho_train, rho_test ,c1_HR_train, c1_HR_test,Vext_train,     Vext_test,eps_train,eps_test, mu_train, mu_test, deltaF_train, deltaF_test=     train_test_split(
        rho_train,c1_HR_train,Vext_train,eps_train,mu_train, deltaF_train , \
        test_size=test_size, random_state=random_state)
    return rho_train, rho_test ,c1_HR_train, c1_HR_test,Vext_train,     Vext_test,eps_train,eps_test, mu_train, mu_test, deltaF_train, deltaF_test

def train_varible(fed):
    #print(fed)
    name = "all_symbols.dat"
    with open(name, "w") as text_file:
        text_file.write(str(fed.free_symbols))
    tmp = open(name, "r")
    syms= tmp.read()
    #print(syms)
    tmp.close()
    syms = syms.replace("{", "")
    syms = syms.replace("}", "")
    syms = syms.replace(" ", "")
    syms = syms.split(',')
    syms=sorted(tuple(syms))
    syms2 = ['s' + i for i in syms]

    
    #initial all symbols and syms
    
    name = "assign_symbols.dat"
    with open(name, "w") as text_file:
        text_file.write("ini")
        text_file.write(str(syms2)+"=symbols("+str(syms)+")\n")
        text_file.write("syms="+str(syms2))
    tmp = open(name, "r")
    tmp=tmp.read()
    tmp=tmp.replace("'", "")
    tmp=tmp.replace("[", "(")
    tmp=tmp.replace("]", ")")
    tmp=tmp.replace("((", "('")
    tmp=tmp.replace("))", "')")
    tmp=tmp.replace("ini(", "")
    tmp=tmp.replace(")=symbols", "=symbols")
    activate_symbols = tmp
    name = "assign_symbols_final.dat"
    with open(name, "w") as text_file:
        text_file.write(activate_symbols)
    
    return syms,activate_symbols



