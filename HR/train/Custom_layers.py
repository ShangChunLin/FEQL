from keras.layers import Layer
from keras.initializers import RandomNormal,Constant
from keras import backend as K 
from keras import regularizers 
# coding: utf-8

# In[ ]:


class conv_pbc(Layer):
    def __init__(self,kernel_size, dx,mean=0,penalty=0,  **kwargs):
        self.kernel_dim = kernel_size
        self.kernel_mean = mean
        self.kernel_penalty = penalty
        self.dx = dx
        #self.name=name
        super(conv_pbc, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(name="w", shape=(self.kernel_dim,1,1), #why 1,1 must in the end??
                                 initializer=RandomNormal(mean=self.kernel_mean, stddev=0.01, seed=None)
                                 ,regularizer=regularizers.l1(self.kernel_penalty)
                                 ,trainable=True)
        
        super(conv_pbc, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        conv_h = int((self.kernel_dim-1)/2)
        a,b,c=x
        N = a.shape[1]
        
        a = K.tile(a,[1,3,1])
        a = a[:,N-conv_h:2*N+conv_h,:]
        a*=self.dx
        #print(a.shape)
        #print(self.w.shape)
        if(b==c): #covolution
            a = K.conv1d(a, K.reverse(self.w,axes=0) , padding='valid')
            print("conv")
        else: #cross corelation
            a = K.conv1d(a, self.w , padding='valid')
            print("cross")
        #print(a.shape)
        return a
    
    def compute_output_shape(self, input_shape):
        #print(input_shape[0])
        a_shape,b_shape,c_shape=input_shape
        return a_shape
        #return input_shape



class linear(Layer):

    def __init__(self,ini=[0,0],**kwargs):
        self.ini=ini
        super(linear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # n = a*n + b
        self.kernel1 = self.add_weight(name='kernel1', 
                                      shape=(1,1),
                                      initializer=Constant(value = self.ini[0]),
                                      trainable=True)
        
        self.kernel2 = self.add_weight(name='kernel2', 
                                      shape=(1,1),
                                      initializer=Constant(value = self.ini[1]),
                                      trainable=True)
        
        super(linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        #print(x.shape, self.kernel.shape)
        return [x * self.kernel1 + self.kernel2, self.kernel1, self.kernel2 ]
        #return x * self.kernel1 + self.kernel2

    def compute_output_shape(self, input_shape):
        #print(input_shape[0])
        return [input_shape,(1,1),(1,1)]
        #return input_shape

class variable(Layer):

    def __init__(self,ini=0,penalty=0,**kwargs):
        self.ini=ini
        self.penalty=penalty
        super(variable, self).__init__(**kwargs)

    def build(self,input_shape):
        # Create a trainable weight variable for this layer.
        # n = a*n + b
        self.kernel1 = self.add_weight(name="varible",
                                      shape=(1,1),
                                      #initializer=Constant(value = self.ini),
                                      initializer=RandomNormal(mean=self.ini, stddev=0.01, seed=None),
                                      regularizer=regularizers.l1(self.penalty),
                                      trainable=True)

        super(variable, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self,x):
        #print(x.shape, self.kernel.shape)
        return  self.kernel1
        #return x * self.kernel1 + self.kernel2

    def compute_output_shape(self,input_shape):
        return (1,1)

class const(Layer):

    def __init__(self,ini=0,**kwargs):
        self.ini=ini
        super(const, self).__init__(**kwargs)

    def build(self,input_shape):
        # Create a trainable weight variable for this layer.
        # n = a*n + b
        self.kernel1 = self.add_weight(name="const",
                                      shape=(1,1),
                                      initializer=Constant(value = self.ini),
                                      trainable=False)

        super(const, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self,x):
        #print(x.shape, self.kernel.shape)
        return  self.kernel1
        #return x * self.kernel1 + self.kernel2

    def compute_output_shape(self,input_shape):
        return (1,1)


def pbc(x): #peridoic boundary condition
    a = K.tile(x,[1,3,1])
    a = a[:,N-conv_h:2*N+conv_h,:]
    #print(a.shape)
    return a

def normal(x):
    x = x*dx
    return x

class kill_small(Layer):

    def __init__(self,kill=0,**kwargs):
        self.kill=kill
        super(kill_small, self).__init__(**kwargs)

    def call(self, x):
        mask = K.cast(K.greater(K.abs(x),self.kill),'float32')
        return x*mask
        

    def compute_output_shape(self, input_shape):
        #print(input_shape[0])
        return (1,1)
        #return input_shape

class weight_kernel(Layer):

    def __init__(self,dim,mean=0,penalty=0,**kwargs):
        self.dim=dim
        self.mean=mean
        self.penalty=penalty
        super(weight_kernel, self).__init__(**kwargs)

    def build(self,input_shape):
        # Create a trainable weight variable for this layer.
        # n = a*n + b
        self.kernel1 = self.add_weight(name="w",
                                      shape=(self.dim,1,1),
                                      #initializer=Constant(value = self.ini),
                                      initializer=RandomNormal(mean=self.mean, stddev=0.01, seed=None),
                                      regularizer=regularizers.l1(self.penalty),
                                      trainable=True)

    def call(self,x):
        #print(x.shape, self.kernel.shape)
        return  self.kernel1
        #return x * self.kernel1 + self.kernel2

    def compute_output_shape(self,input_shape):
        return (self.dim,1,1)
