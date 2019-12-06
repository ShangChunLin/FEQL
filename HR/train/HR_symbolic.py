
# coding: utf-8

# # FEQL for hard rod (hard sphere in 1D)
# ## The training result may differ from the paper, but performance show be close 

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
#set_session(tf.Session(config=config))

# Do other imports now...

from scipy.integrate import simps
from numpy import exp, absolute
from numpy import exp, asarray, empty
from numpy import zeros, array, float, random
import itertools as it
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


import keras
from keras.layers import Input, Dense,Conv1D,Reshape,Flatten
from keras.layers import Add,Subtract,Multiply
from keras.layers import Lambda,RepeatVector

from keras.models import Model
from keras.models import model_from_json
from keras import backend as K 
from keras.optimizers import Adam,RMSprop,SGD
from keras.initializers import RandomNormal,Constant
from keras import regularizers
from keras.layers import Layer
from keras.callbacks import ModelCheckpoint,LambdaCallback,Callback,EarlyStopping
K.clear_session()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras.utils import plot_model

from Custom_layers import conv_pbc,variable,const,kill_small,weight_kernel #self_defined_layers


#from equation_gen_new import equation_gen_couple_eps_non_bias  #Full 
#from equation_gen_new import equation_gen_couple_HR_non_bias   #only athermal term
#from equation_gen_new import equation_gen_couple_tail_non_bias #only tail
from equation_gen_new import equation_gen_sparse
from sympy import diff,log,symbols,exp,simplify
from sympy.utilities.lambdify import lambdify

from sklearn.model_selection import train_test_split

from shutil import copyfile


# ## Load data and etc.

# In[ ]:


data_file = '../HR_data/'
f=np.loadtxt(data_file+"parameter.dat")
L = float(f[0])
N = int(f[1])
dx=L/N
input_shape = (N,1)
print(L,N,dx)


# In[ ]:


f = open(data_file+'rho_z.dat', 'r')
MC_inform = f.read().splitlines()
for i in range(len(MC_inform)):
    #print(MC_inform[i])
    MC_inform[i]=MC_inform[i].split("\t")
f.close()
#MC_inform = np.asarray(MC_inform)
batch_size = len(MC_inform)


# In[ ]:


batch_size 


# In[ ]:


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
    #print(np.min(1-n1))
    #return cal_n(F0,w0)+cal_n(F1,w1)
    return np.zeros(len(rho))
    


# In[ ]:


MC_inform[-1]


# In[ ]:


rho= []
c1_HR = []
Vext= []
mu = []
eps= []

for i in range (batch_size):
    Vext +=[np.loadtxt(data_file+"Vext_"+str(i)+".dat")]
    rho += [np.loadtxt(data_file+"rho_"+str(i)+".dat")]
    c1_HR+=[cal_c1_FMT(rho[i])]
    mu += [np.log(np.float(MC_inform[i][2]))]
    eps += [0]
    
# flip data to increase traiing size     
for i in range (batch_size):
    Vext +=[np.flip(np.loadtxt(data_file+"Vext_"+str(i)+".dat"))]
    rho += [np.flip(np.loadtxt(data_file+"rho_"+str(i)+".dat"))]
    c1_HR+=[cal_c1_FMT(rho[batch_size+i])]
    mu += [np.log(np.float(MC_inform[i][2]))]
    eps += [0]
    
#inputs[0].shape


# In[ ]:


np.exp(np.min(mu)),np.exp(np.max(mu))


# In[ ]:


rho_mean = np.mean(rho,axis=0)
print(max(rho_mean),min(rho_mean))


# In[ ]:


plt.plot(np.exp(mu[0]-c1_HR[0]-Vext[0]))
plt.plot(rho[0])


# In[ ]:


rho_train=np.asarray(rho)
rho_train.shape


# In[ ]:


rho_train=np.asarray(rho)
c1_HR_train=np.asarray(c1_HR)
Vext_train=np.asarray(Vext)
eps_train = np.asarray(eps)
mu_train = np.asarray(mu)
deltaF_train = np.asarray(np.zeros(rho_train.shape))


rho_train = rho_train.reshape(rho_train.shape[0], N , 1)
c1_HR_train = c1_HR_train.reshape(c1_HR_train.shape[0], N , 1)
Vext_train = Vext_train.reshape(Vext_train.shape[0], N , 1)
eps_train = eps_train.reshape(eps_train.shape[0], 1 , 1)
mu_train = mu_train.reshape(mu_train.shape[0], 1 , 1)
deltaF_train=deltaF_train.reshape(deltaF_train.shape[0], N , 1)

test_size=0.1

rho_train, rho_test ,c1_HR_train, c1_HR_test,Vext_train, Vext_test,eps_train,eps_test, mu_train, mu_test, deltaF_train, deltaF_test= train_test_split(rho_train,c1_HR_train,Vext_train,eps_train,mu_train, deltaF_train , test_size=test_size, random_state=42)
#c1_HR_train, c1_HR_test    = train_test_split(c1_HR_train, test_size=test_size)
#Vext_train, Vext_test= train_test_split(Vext_train, test_size=test_size)
#eps_train, eps_test  = train_test_split(eps_train, test_size=test_size)
#mu_train, mu_test    = train_test_split(mu_train, test_size=test_size)


print(rho_train.shape)
print(rho_test.shape)
print(c1_HR_train.shape)
print(Vext_train.shape)
print(mu_train.shape)
print(eps_train.shape)
print(deltaF_test.shape)


# In[ ]:


plt.plot(rho_train[0],label="rho1")
#plt.plot(Vext_train[0]/100,label="V1")

plt.plot(rho_test[0],label="rho2")
#plt.plot(Vext_test[0]/100,label="V2")
plt.legend()


# In[ ]:


rho_train[0];


# In[ ]:


rho_train[0].shape


# ## Generate free energy density on the fly

# In[ ]:


n_layer=3
n_density=3
n_parameter=0
n_id = 1
n_log = 1
n_exp = 1
n_mul = 3
n_div = 1

fed = equation_gen_sparse(n_layer,n_density,n_parameter,n_id,n_log,n_exp,n_mul,n_div)
# no bias is dangerous for log, non-couple may have particle density not coupled with others 


# In[ ]:


#fed.subs('eps0',0)
fed


# In[ ]:


#save all symbols to all_symbols.dat
name = "all_symbols.dat"
with open(name, "w") as text_file:
    text_file.write(str(fed.free_symbols))
tmp = open(name, "r")
syms= tmp.read()
tmp.close()
syms = syms.replace("{", "")
syms = syms.replace("}", "")
syms = syms.replace(" ", "")
syms = syms.split(',')
syms=sorted(tuple(syms))
syms2 = ['s' + i for i in syms]


# In[ ]:


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
exec(tmp)
name = "assign_symbols_final.dat"
with open(name, "w") as text_file:
    text_file.write(tmp)


# In[ ]:


#def fed_eqn
fed_string = str(fed)
fed_string=fed_string.replace("a", "sa")
fed_string=fed_string.replace("b", "sb")
fed_string=fed_string.replace("n", "sn")
fed_string=fed_string.replace("eps", "seps")

name = "free_energy_final.dat"
with open(name, "w") as text_file:
    text_file.write(fed_string)

name = "free_energy_def.dat"
with open(name, "w") as text_file:
    text_file.write("def fed_eqn():\n")
    text_file.write("     return "+fed_string) 
    text_file.write("\n")
tmp = open(name, "r")
tmp=tmp.read()
exec(tmp)
fed_tf= lambdify(syms,fed_eqn(),'tensorflow')


# In[ ]:


fed_tf


# In[ ]:


len(syms)


# In[ ]:


n_conv = n_density*(1)  #(n_parameter+1) for full 


# In[ ]:


name = "diff_equations.dat"
with open(name, "w") as text_file:
    for i in range (n_conv):
        text_file.write("def Df_n"+str(i)+"():\n")
        text_file.write("     return diff(fed_eqn(),sn"+str(i)+")\n")
        text_file.write("\n")
        text_file.write("df_n"+str(i)+"= lambdify(syms,Df_n"+str(i)+"(),\"tensorflow\")\n") 
        text_file.write("\n")
        
tmp = open(name, "r")
tmp=tmp.read()
exec(tmp)


# In[ ]:


s1=fed_eqn().free_symbols

gen_s2="s2=("
for i in range (n_conv):
    gen_s2+="+diff(fed_eqn(),sn"+str(i)+")"
gen_s2+=").free_symbols"
#print(gen_s2)
exec(gen_s2)



# In[ ]:


fed_eqn()


# In[ ]:


diff(fed_eqn(),sn0).subs({"n0":0,"n1":0}).free_symbols;


# In[ ]:


sub_str = "({"
for i in range (n_conv):
    sub_str+="\"n"+str(i)+"\":0"
    if(i!=n_conv-1):
        sub_str+=","
sub_str += "})"


# In[ ]:


sub_str


# In[ ]:



gen_s3="s3=("
for i in range (n_conv):
    gen_s3+="+diff(fed_eqn(),sn"+str(i)+").subs"+sub_str
gen_s3+=").free_symbols"
print(gen_s3)
exec(gen_s3)


# In[ ]:


tmp=str(s3)
tmp=tmp.replace("'", "")
tmp=tmp.replace(" ", "")
tmp=tmp.replace("{", "")
tmp=tmp.replace("}", "")
tmp = tuple(tmp.split(','))
tmp = sorted(tmp)
print(tmp)
redunt = ""

for i in range(len(tmp)):
    if tmp[i][1].isdigit():
        if int(tmp[i][1])>1:
            redunt+="s"+tmp[i]+"+"
redunt+="0"  


# In[ ]:


print(redunt)
redunt_eq=eval(redunt)
s3=redunt_eq.free_symbols
s1=fed_eqn().free_symbols
s1-s3;


# In[ ]:


redunt = str(s3)
redunt=redunt.replace("'", "")
redunt=redunt.replace(" ", "")
redunt=redunt.replace("{", "")
redunt=redunt.replace("}", "")
redunt = tuple(redunt.split(','))
redunt = sorted(redunt)

train_val = str(s2-s3)
train_val=train_val.replace("'", "")
train_val=train_val.replace(" ", "")
train_val=train_val.replace("{", "")
train_val=train_val.replace("}", "")
train_val= tuple(train_val.split(','))
train_val = sorted(train_val)


# In[ ]:


name = "redudent.dat"
with open(name, "w") as text_file:
    text_file.write(str(redunt))

name = "trainable.dat"
with open(name, "w") as text_file:
    text_file.write(str(train_val))
    
print(len(s1),len(s2),len(s3))


# ## Generate model on the fly

# In[ ]:


conv_dim = int((int(8/dx/2))*2+1)
conv_h = int((conv_dim-1)/2) # half of conv_dim
print(conv_dim,conv_h,conv_dim*dx)

K.clear_session()
np.random.seed(424242)


# In[ ]:


def cal_z(s): #particle N conserved
    [rho,rhoML]=s
    return K.log(K.sum(rho,axis=1,keepdims=True)/K.sum(rhoML,axis=1,keepdims=True))

def cal2_z(s): #(rho-z*rhoML)**2 mini
    [rho,rhoML]=s
    #res = K.sum(rho[1]*rhoML[1],axis=1)/K.sum(rhoML[1]*rhoML[1],axis=1)
    res1=K.sum(rho*rhoML,axis=1,keepdims=True)
    res2=K.sum(rhoML*rhoML,axis=1,keepdims=True)
    return K.log(res1/res2)

def pbc_cross(s):
    a,w=s
    print("cross")
    a = K.tile(a,[1,3,1])
    a = a[:,N-conv_h:2*N+conv_h,:]
    a = K.conv1d(a,w, padding='valid')
    a = a*dx
    return a

def pbc_conv(s):
    a,w=s
    print("conv")
    a = K.tile(a,[1,3,1])
    a = a[:,N-conv_h:2*N+conv_h,:]
    a = K.conv1d(a,K.reverse(w,axes=0), padding='valid')
    #a = K.reverse(a,axes=1)
    a = a*dx
    return a


# In[ ]:


syms


# In[ ]:



model_name = "The_model.dat"
with open(model_name, "w") as file:
    file.write("def encoder(penalty,kill,conv_penalty):\n")
 
        
    
    file.write("    rho = Input(shape=input_shape)\n")
    file.write("    c1_HR = Input(shape=input_shape)\n")
    file.write("    Vext = Input(shape=input_shape)\n")
    for i in range (n_parameter):
        file.write("    eps"+str(i)+" = Input(shape=(1,1))\n")
    file.write("    mu = Input(shape=(1,1))\n")
    
    file.write("\n")
    file.write("    dummy = mu\n")
    for i in range (n_conv):
        file.write("    w"+str(i)+" = weight_kernel(conv_dim,mean=(np.random.rand()-0.5)*0.02,penalty=conv_penalty,name=\"w"+str(i)+"\")(dummy)\n")    
    #for i in range (n_density):
    file.write("    print(w0.shape)\n")
    for i in range (n_conv):
        file.write("    n"+str(i)+" = Lambda(pbc_conv)([rho,w"+str(i)+"])\n")
    file.write("\n")           
    
    for i in range(len(train_val)):
        if(str(train_val[i])[0:3]!="eps" and str(train_val[i])[0:1]!="n"):
            if(str(train_val[i])[0]=="a"):
                file.write("    "+str(train_val[i])+"=variable((np.random.rand()-0.5)*0.02,penalty"+",name=\""+str(train_val[i])+"\")(dummy)\n") 
            elif(str(train_val[i])[0]=="b"):
                file.write("    "+str(train_val[i])+"=variable(0.01,penalty"+",name=\""+str(train_val[i])+"\")(dummy)\n") ## ini some number not zero for safety
            
    file.write("\n")
    
    for i in range(len(redunt)):
        #if(str(redunt[i])[0]=="a" or str(redunt[i])[0]=="b"):
        if(str(redunt[i])[0]=="a"):
            file.write("    "+str(redunt[i])+"=const(name=\""+str(redunt[i])+"\")(dummy)\n") ## non trainble , just set to zero
    file.write("\n")
    
    #file.write("    kill=const("+str(kill)+")\n")
    for i in range(len(train_val)):
        #if(str(train_val[i])[0]=="a" or str(train_val[i])[0]=="b"):
        if(str(train_val[i])[0]=="a"):
            file.write("    "+str(train_val[i])+ "=kill_small(kill)("+str(train_val[i])+")\n")
    file.write("\n")
    
    file.write("    s = ["+str(syms)[1:-1]+"]\n")
    file.write("\n")
    
    for i in range (n_conv):
        file.write("    f"+str(i)+" = Lambda(lambda x: df_n"+str(i)+"(")
        for ii in range (len(syms)):
            file.write("x["+str(ii)+"]")
            if(ii!=len(syms)-1):
                file.write(",")
        file.write("))(s)\n")
        file.write("    f"+str(i)+" = Lambda(pbc_cross)([f"+str(i)+","+"w"+str(i)+"])\n")
    file.write("\n")
    file.write("    c1 = Add()([")
    for i in range (n_conv):
        file.write("f"+str(i))
        if(i!=n_conv-1):
            file.write(",")
    file.write("])")
    file.write("\n")
    file.write("    rhoML = Lambda(lambda x: K.exp(-x[0]-x[1]-x[2]))([c1,c1_HR,Vext])\n")
    file.write("    muML = Lambda(cal_z)([rho,rhoML])# dont change order of muML and rhoML\n")  
    file.write("    rhoML= Lambda(lambda x: K.exp(x[0])*x[1])([muML,rhoML])\n")
    for i in range (n_conv):
        file.write("    asymw"+str(i)+" = Lambda(lambda x:dx*K.sum(K.abs(K.abs(x)-K.abs(K.reverse(x,axes=0))),axis=0,keepdims=True))(w"+str(i)+")\n")
    file.write("    asym = Add()([")
    for i in range (n_conv):
        file.write("asymw"+str(i))
        if(i!=n_conv-1):
            file.write(",")
    file.write("])")
    file.write("\n")
    file.write("    print(asym.shape)\n")
    file.write("    asym= Lambda(lambda x:K.abs(x[0])/K.abs(x[0])*x[1])([mu,asym])\n")
    file.write("    print(asym.shape)\n")
    
    file.write("    model = Model([rho,c1_HR,Vext")
    for i in range (n_parameter):
        file.write(",eps"+str(i))         
    file.write(",mu], [rhoML,muML,asym] , name = \"F_ML\")\n")
    #file.write("    model.summary()\n")
    file.write("\n")
    file.write("    return model")


# In[ ]:


tmp = open(model_name, "r")
tmp=tmp.read()
exec(tmp)


# In[ ]:



#F_ML.summary()


# ## choo choo ~ (training part)

# In[ ]:


logname="log_file.dat"
with open(logname, "w") as file:
    file.write("\n")
def show_loss(epoch, logs):
    if(epoch%50==0):
        print("epcoh\t=\t"+str(epoch)+"\t"+str(logs))
    logname="log_file.dat"
    with open(logname, "a") as file:
        file.write(str(logs)+"\n")


# In[ ]:


class StoppingByLossNan(Callback):
    def __init__(self, monitor='val_loss', monitor2='loss', verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.monitor2 = monitor2
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current2 = logs.get(self.monitor2)
        
        if(current is None):
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if(np.isnan(current) or np.isinf(current) or np.isnan(current2) or np.isinf(current2)):
            print("loss diverge at epoch = " + str(epoch))
            self.model.stop_training = True


# In[ ]:


file = "F_ML.hdf5"
checkpoint=ModelCheckpoint(file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
stop=EarlyStopping(monitor='val_loss', min_delta=0, patience=2000,  
                   verbose=0, mode='min', baseline=None, restore_best_weights=True)
epochLogCallback = LambdaCallback(on_epoch_end=show_loss)

callbacks_list = [checkpoint,epochLogCallback,StoppingByLossNan(monitor='val_loss',monitor2='loss', verbose=1)]


# In[ ]:


def output_parameters(weight_dict,kill,phase=-1):
    w_array = np.zeros([n_conv,conv_dim])
    for i in range (n_conv):
        for j in list(weight_dict.keys()):
            arr = weight_dict[j][1]
            if(n_conv<10):
                if(j[0]=="w" and j[1]==str(i)):
                    w_array[i]= np.copy(arr.flatten())
            elif(n_conv<100):
                if(j[0]=="w" and j[1:3]==str(i)):
                    w_array[i]= np.copy(arr.flatten())
            else:
                print("n>100? serious?")
    np.savetxt("w_array.dat",w_array)

    name="all_parameter_"+str(phase)+".dat"        
    with open(name, "w") as file:
        for i in list(weight_dict.keys()):
            arr = weight_dict[i][1]
            arr = np.array(arr.flatten())
            #arr = np.roll(arr,conv_h)
            if(i[0]!="w"):
                file.write(str(i)+" "+str(weight_dict[i][1])+"\n")
    name = "ML_parameter.dat"
    with open(name, "w") as text_file:
        text_file.write("n_parameter="+str(n_parameter)+"\n")
        text_file.write("n_conv="+str(n_conv)+"\n")
        text_file.write("conv_dim="+str(conv_dim)+"\n")
        text_file.write("kill="+str(kill)+"\n")
    
    print("output_parameters")
    return w_array


# In[ ]:


# the 3 step procedure descripted in the paper 
for phase in range (0,3):
    
    print("phase = "+str(phase))
    
    penalty = 0#0.00001
    conv_penalty=10**-7
    kill = 0
    epochs=1000
    lr = 10**-2
    asym=0.001 # just for a (anti-)symmetry initial condition and stable training in first step,
               # not important. Or one could set lr =10**-3, asym = 0 and epochs=5000 

    if(phase==1):
        epochs=5000
        penalty = 8*10**-5
        conv_penalty=10**-7
        kill = 0
        lr = 10**-3
        asym=0
        
    if(phase==2):
        epochs=1000
        penalty = 0
        conv_penalty=0
        kill = 0.05
        lr = 10**-3
        asym=0
        
    x1 = Input(shape=input_shape)
    x2 = Input(shape=input_shape)
    x3 = Input(shape=input_shape)
    x4 = Input(shape=(1,1))
    x5 = Input(shape=(1,1))

    build = encoder(penalty,kill,conv_penalty)

    #build.
    [y1,y2,y3] = build([x1,x2,x3,x4])

    F_ML = Model([x1,x2,x3,x4],[y1,y2,y3], name='F_learn')

    optimal = 'mae'
    F_ML.compile(optimizer=Adam(lr=lr), loss=optimal,loss_weights=[0.9, 0.1,asym]) 
    if(os.path.isfile(file) ):
         if(phase !=0):
            F_ML.load_weights(file)
    F_ML.fit(x=[rho_train,c1_HR_train,Vext_train,mu_train], 
             y=[rho_train,mu_train,mu_train*0],
             epochs=epochs,
             shuffle=True,
             batch_size=64,
             validation_data=([rho_test,c1_HR_test,Vext_test,mu_test], [rho_test,mu_test,mu_test*0]),
             callbacks=callbacks_list,
             verbose=2
            )
    F_ML.load_weights(file)

    names = [weight.name for layer in F_ML.layers for weight in layer.weights]
    weights = F_ML.get_weights()
    weight_dict = {}
    for name, weight in zip(names, weights):
        #print(name)
        weight_dict[str(name)]=[weight.shape,weight]
    w_array=output_parameters(weight_dict,kill,phase)
    copyfile(file, "meta_phase_"+str(phase)+".h5")
    print("lr="+str(lr)+"\tstep\t=\t"+str(i))
    #if(lr<10**-6):
    #    break


# In[ ]:


for i in range (n_conv):
    plt.plot(w_array[i])

