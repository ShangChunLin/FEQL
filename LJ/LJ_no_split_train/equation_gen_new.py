
# coding: utf-8


from sympy import MatrixSymbol, Matrix
from sympy import symbols
from sympy import log,exp,diff
from sympy import Symbol


def chcek(n_layer,n_density,n_parameter,n_id,n_exp,n_log,n_mul,n_div):
    if(n_layer<1):
        print("n_layer must larger than 1")
        return False
    if(n_density<2):
        print("n_density must >= than 2")
        return False
    #if(n_parameter<1):
    #    print("n_parameter<1? just use FMT")
    #    return False
    
    if(n_id<0 or n_exp<0 or n_log<0 or n_mul<0 or n_div<0):
        print("n_operation must >=0")
        return False
    
    n_dim1 = n_id+n_log+n_exp+n_mul*2+n_div*2
    if(n_dim1<0):
        print("no operation!")
        return False

#---------------------------------------------------------------------------------
def equation_gen_couple_eps_non_bias(n_layer,n_density,n_parameter,n_id = 2,n_log = 2
                 ,n_exp = 2,n_mul = 1,n_div = 1):
    
    if(chcek(n_layer,n_density,n_parameter,n_id,n_exp,n_log,n_mul,n_div)==False):
        return 
     
    n_dim1 = n_id+n_log+n_exp+n_mul*2+n_div*2
    n_dim1p= n_id+n_log+n_exp+n_mul+n_div
    n_dim0 = int(n_density*(n_density+1)/2*(n_parameter+1)) # couple in first layer ? 
    
    layer = 0
    name = "layer_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_parameter):
            text_file.write("seps"+str(i)+'=Symbol(\'eps'+str(i)+'\')'+"\n")
        for i in range (n_density*(n_parameter+1)):
            text_file.write("sn"+str(i)+'=Symbol(\'n'+str(i)+'\')'+"\n")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"= Matrix("+str(n_dim0)+",1,[")
        #for i in range (n_parameter):
        #    text_file.write("seps"+str(i)+",")
        for i in range (n_density):
            for j in range (i,n_density):
                text_file.write("sn"+str(i)+"*"+"sn"+str(j))
                #if(i!=(n_density-1) or j!=(n_density-1)):
                text_file.write(",")
        for eps in range (n_parameter):
            for i in range ((eps+1)*n_density,(eps+2)*n_density):
                for j in range (i,2*n_density):
                    text_file.write("seps"+str(eps)+"*"+"sn"+str(i)+"*"+"sn"+str(j))
                    if(i!=(2*n_density-1) or j!=(2*n_density-1)):
                        text_file.write(",")
        text_file.write("])")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())  
    
    layer = 1
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_dim1*n_dim0):
            text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
        for i in range (n_div):
            text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
        text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim0)+",[")
        for i in range (n_dim1*n_dim0-1):
            text_file.write("sa"+str(layer)+"L"+str(i)+",")
        text_file.write("sa"+str(layer)+"L"+str(n_dim1*n_dim0-1)+"])")
        text_file.write("\n")
        #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
        #for i in range (n_div-1):
        #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
        #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
        #text_file.write('\n')
        #text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"+B"+str(layer))
        text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1))

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genXp_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
        for i in range(n_id):
            text_file.write("X"+str(layer)+"["+str(i)+"],")
        for i in range(n_id,n_id+n_exp):
            text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
        base=n_id+n_exp
        for i in range(n_id+n_exp,n_id+n_exp+n_log):
            text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
        base=n_id+n_exp+n_log+n_mul*2
        for i in range(base,base+n_div*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
            if(i!=base+n_div*2-2):
                text_file.write(",")
        text_file.write("])")

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode)   

    for L in range (2,n_layer):
        layer = L
        print("processing "+str(L)+"th layer")
        name = "genX_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            for i in range (n_dim1p*n_dim1):
                text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
            for i in range (n_div):
                text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
            text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim1p)+",[")
            for i in range (n_dim1*n_dim1p-1):
                text_file.write("sa"+str(layer)+"L"+str(i)+",")
            text_file.write("sa"+str(layer)+"L"+str(n_dim1p*n_dim1-1)+"])")
            text_file.write("\n")
            #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
            #for i in range (n_dim1-1):
            #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
            #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
            text_file.write('\n')
            text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
        name = "genXp_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
            for i in range(n_id):
                text_file.write("X"+str(layer)+"["+str(i)+"],")
            for i in range(n_id,n_id+n_exp):
                text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
            base=n_id+n_exp
            for i in range(n_id+n_exp,n_id+n_exp+n_log):
                text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
            base=n_id+n_exp+n_log+n_mul*2
            for i in range(base,base+n_div*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
                if(i!=base+n_div*2-2):
                    text_file.write(",")
            text_file.write("])")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
    
    name = "free_energy_density.dat"
    with open(name, "w") as text_file:
        text_file.write("fed=")
        for i in range(n_dim1p-1):
            text_file.write("X"+str(n_layer-1)+"p"+"["+str(i)+"]+")
        text_file.write("X"+str(n_layer-1)+"p"+"["+str(n_dim1p-1)+"]")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    
    return fed

#-----------------
def equation_gen_couple_HR_non_bias(n_layer,n_density,n_parameter=0,n_id = 2,n_log = 2
                 ,n_exp = 2,n_mul = 1,n_div = 1):
    
    if(chcek(n_layer,n_density,n_parameter,n_id,n_exp,n_log,n_mul,n_div)==False):
        return 
     
    n_dim1 = n_id+n_log+n_exp+n_mul*2+n_div*2
    n_dim1p= n_id+n_log+n_exp+n_mul+n_div
    n_dim0 = int(n_density*(n_density+1)/2) # couple in first layer ? 
    
    layer = 0
    name = "layer_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_parameter):
            text_file.write("seps"+str(i)+'=Symbol(\'eps'+str(i)+'\')'+"\n")
        for i in range (n_density*(n_parameter+1)):
            text_file.write("sn"+str(i)+'=Symbol(\'n'+str(i)+'\')'+"\n")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"= Matrix("+str(n_dim0)+",1,[")
        #for i in range (n_parameter):
        #    text_file.write("seps"+str(i)+",")
        for i in range (n_density):
            for j in range (i,n_density):
                text_file.write("sn"+str(i)+"*"+"sn"+str(j))
                #if(i!=(n_density-1) or j!=(n_density-1)):
                text_file.write(",")
        #for eps in range (n_parameter):
        #    for i in range (n_density,2*n_density):
        #        for j in range (i,2*n_density):
        #            text_file.write("seps"+str(eps)+"*"+"sn"+str(i)+"*"+"sn"+str(j))
        #            if(i!=(2*n_density-1) or j!=(2*n_density-1)):
        #                text_file.write(",")
        text_file.write("])")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())  
    
    layer = 1
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_dim1*n_dim0):
            text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
        for i in range (n_div):
            text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
        text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim0)+",[")
        for i in range (n_dim1*n_dim0-1):
            text_file.write("sa"+str(layer)+"L"+str(i)+",")
        text_file.write("sa"+str(layer)+"L"+str(n_dim1*n_dim0-1)+"])")
        text_file.write("\n")
        #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
        #for i in range (n_div-1):
        #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
        #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
        #text_file.write('\n')
        #text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"+B"+str(layer))
        text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1))

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genXp_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
        for i in range(n_id):
            text_file.write("X"+str(layer)+"["+str(i)+"],")
        for i in range(n_id,n_id+n_exp):
            text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
        base=n_id+n_exp
        for i in range(n_id+n_exp,n_id+n_exp+n_log):
            text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
        base=n_id+n_exp+n_log+n_mul*2
        for i in range(base,base+n_div*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
            if(i!=base+n_div*2-2):
                text_file.write(",")
        text_file.write("])")

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode)   

    for L in range (2,n_layer):
        layer = L
        print("processing "+str(L)+"th layer")
        name = "genX_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            for i in range (n_dim1p*n_dim1):
                text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
            for i in range (n_div):
                text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
            text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim1p)+",[")
            for i in range (n_dim1*n_dim1p-1):
                text_file.write("sa"+str(layer)+"L"+str(i)+",")
            text_file.write("sa"+str(layer)+"L"+str(n_dim1p*n_dim1-1)+"])")
            text_file.write("\n")
            #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
            #for i in range (n_dim1-1):
            #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
            #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
            text_file.write('\n')
            text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
        name = "genXp_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
            for i in range(n_id):
                text_file.write("X"+str(layer)+"["+str(i)+"],")
            for i in range(n_id,n_id+n_exp):
                text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
            base=n_id+n_exp
            for i in range(n_id+n_exp,n_id+n_exp+n_log):
                text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
            base=n_id+n_exp+n_log+n_mul*2
            for i in range(base,base+n_div*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
                if(i!=base+n_div*2-2):
                    text_file.write(",")
            text_file.write("])")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
    
    name = "free_energy_density.dat"
    with open(name, "w") as text_file:
        text_file.write("fed=")
        for i in range(n_dim1p-1):
            text_file.write("X"+str(n_layer-1)+"p"+"["+str(i)+"]+")
        text_file.write("X"+str(n_layer-1)+"p"+"["+str(n_dim1p-1)+"]")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    
    return fed

def equation_gen_couple_tail_non_bias(n_layer,n_density,n_parameter=1,n_id = 2,n_log = 2
                 ,n_exp = 2,n_mul = 1,n_div = 1):
    
    if(chcek(n_layer,n_density,n_parameter,n_id,n_exp,n_log,n_mul,n_div)==False):
        return 
     
    n_dim1 = n_id+n_log+n_exp+n_mul*2+n_div*2
    n_dim1p= n_id+n_log+n_exp+n_mul+n_div
    n_dim0 = int(n_density*(n_density+1)/2*n_parameter) # couple in first layer ? 
    
    layer = 0
    name = "layer_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_parameter):
            text_file.write("seps"+str(i)+'=Symbol(\'eps'+str(i)+'\')'+"\n")
        for i in range (n_density*(n_parameter)):
            text_file.write("sn"+str(i)+'=Symbol(\'n'+str(i)+'\')'+"\n")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"= Matrix("+str(n_dim0)+",1,[")
        #for i in range (n_parameter):
        #    text_file.write("seps"+str(i)+",")
        #for i in range (n_density):
        #    for j in range (i,n_density):
        #        text_file.write("sn"+str(i)+"*"+"sn"+str(j))
                #if(i!=(n_density-1) or j!=(n_density-1)):
        #        text_file.write(",")
        for eps in range (n_parameter):
            for i in range (0,n_density):
                for j in range (i,n_density):
                    text_file.write("seps"+str(eps)+"*"+"sn"+str(i)+"*"+"sn"+str(j))
                    if(i!=(2*n_density-1) or j!=(2*n_density-1)):
                        text_file.write(",")
        text_file.write("])")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())  
    
    layer = 1
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_dim1*n_dim0):
            text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
        for i in range (n_div):
            text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
        text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim0)+",[")
        for i in range (n_dim1*n_dim0-1):
            text_file.write("sa"+str(layer)+"L"+str(i)+",")
        text_file.write("sa"+str(layer)+"L"+str(n_dim1*n_dim0-1)+"])")
        text_file.write("\n")
        #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
        #for i in range (n_div-1):
        #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
        #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
        #text_file.write('\n')
        #text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"+B"+str(layer))
        text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1))

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genXp_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
        for i in range(n_id):
            text_file.write("X"+str(layer)+"["+str(i)+"],")
        for i in range(n_id,n_id+n_exp):
            text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
        base=n_id+n_exp
        for i in range(n_id+n_exp,n_id+n_exp+n_log):
            text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
        base=n_id+n_exp+n_log+n_mul*2
        for i in range(base,base+n_div*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
            if(i!=base+n_div*2-2):
                text_file.write(",")
        text_file.write("])")

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode)   

    for L in range (2,n_layer):
        layer = L
        print("processing "+str(L)+"th layer")
        name = "genX_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            for i in range (n_dim1p*n_dim1):
                text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
            for i in range (n_div):
                text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
            text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim1p)+",[")
            for i in range (n_dim1*n_dim1p-1):
                text_file.write("sa"+str(layer)+"L"+str(i)+",")
            text_file.write("sa"+str(layer)+"L"+str(n_dim1p*n_dim1-1)+"])")
            text_file.write("\n")
            #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
            #for i in range (n_dim1-1):
            #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
            #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
            text_file.write('\n')
            text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
        name = "genXp_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
            for i in range(n_id):
                text_file.write("X"+str(layer)+"["+str(i)+"],")
            for i in range(n_id,n_id+n_exp):
                text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
            base=n_id+n_exp
            for i in range(n_id+n_exp,n_id+n_exp+n_log):
                text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
            base=n_id+n_exp+n_log+n_mul*2
            for i in range(base,base+n_div*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
                if(i!=base+n_div*2-2):
                    text_file.write(",")
            text_file.write("])")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
    
    name = "free_energy_density.dat"
    with open(name, "w") as text_file:
        text_file.write("fed=")
        for i in range(n_dim1p-1):
            text_file.write("X"+str(n_layer-1)+"p"+"["+str(i)+"]+")
        text_file.write("X"+str(n_layer-1)+"p"+"["+str(n_dim1p-1)+"]")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    
    return fed

#---------------------------------------------------------------------------------
def equation_gen_non_couple_non_bias(n_layer,n_density,n_parameter,n_id = 2,n_log = 2
                 ,n_exp = 2,n_mul = 1,n_div = 1):
    if(chcek(n_layer,n_density,n_parameter,n_id,n_exp,n_log,n_mul,n_div)==False):
        return 
    
    n_dim1 = n_id+n_log+n_exp+n_mul*2+n_div*2
    n_dim1p= n_id+n_log+n_exp+n_mul+n_div
    n_dim0 = n_density*(1+n_parameter) # couple in first layer ? 
    
    layer = 0
    name = "layer_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_parameter):
            text_file.write("seps"+str(i)+'=Symbol(\'eps'+str(i)+'\')'+"\n")
        for i in range (n_density*(1+n_parameter)):
            text_file.write("sn"+str(i)+'=Symbol(\'n'+str(i)+'\')'+"\n")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"= Matrix("+str(n_dim0)+",1,[")
        for i in range (n_density):
            text_file.write("sn"+str(i)+",")
        for eps in range (0,n_parameter):
            for i in range ((eps+1)*n_density,(eps+2)*n_density):
                text_file.write("seps"+str(eps)+"*sn"+str(i)+",")
        text_file.write("])")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())  
    
    layer = 1
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_dim1*n_dim0):
            text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
        for i in range (n_div):
            text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
        text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim0)+",[")
        for i in range (n_dim1*n_dim0-1):
            text_file.write("sa"+str(layer)+"L"+str(i)+",")
        text_file.write("sa"+str(layer)+"L"+str(n_dim1*n_dim0-1)+"])")
        text_file.write("\n")
        #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
        #for i in range (n_dim1-1):
        #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
        #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
        text_file.write('\n')
        text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1))

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genXp_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
        for i in range(n_id):
            text_file.write("X"+str(layer)+"["+str(i)+"],")
        for i in range(n_id,n_id+n_exp):
            text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp,n_id+n_exp+n_log):
            text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
        base=n_id+n_exp+n_log+n_mul*2
        for i in range(base,base+n_div*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
            if(i!=n_id+n_exp+n_log+n_mul*2+n_div*2-2):
                text_file.write(",")
        text_file.write("])")

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode)   

    for L in range (2,n_layer):
        layer = L
        print("processing "+str(L)+"th layer")
        name = "genX_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            for i in range (n_dim1p*n_dim1):
                text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
            for i in range (n_div):
                text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
            text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim1p)+",[")
            for i in range (n_dim1*n_dim1p-1):
                text_file.write("sa"+str(layer)+"L"+str(i)+",")
            text_file.write("sa"+str(layer)+"L"+str(n_dim1p*n_dim1-1)+"])")
            text_file.write("\n")
            #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
            #for i in range (n_dim1-1):
            #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
            #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
            text_file.write('\n')
            #text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p"+"+B"+str(layer))
            text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
        name = "genXp_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
            for i in range(n_id):
                text_file.write("X"+str(layer)+"["+str(i)+"],")
            for i in range(n_id,n_id+n_exp):
                text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp,n_id+n_exp+n_log):
                text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
            base=n_id+n_exp+n_log+n_mul*2
            for i in range(base,base+n_div*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
                if(i!=n_id+n_exp+n_log+n_mul*2+n_div*2-2):
                    text_file.write(",")
            text_file.write("])")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
    
    name = "free_energy_density.dat"
    with open(name, "w") as text_file:
        text_file.write("fed=")
        for i in range(n_dim1p-1):
            text_file.write("X"+str(n_layer-1)+"p"+"["+str(i)+"]+")
        text_file.write("X"+str(n_layer-1)+"p"+"["+str(n_dim1p-1)+"]")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    
    return fed

#---------------------------------------------------------------------------------
#not tested
def equation_gen_non_couple_tail_non_bias(n_layer,n_density,n_parameter,n_id = 2,n_log = 2
                 ,n_exp = 2,n_mul = 1,n_div = 1):
    if(chcek(n_layer,n_density,n_parameter,n_id,n_exp,n_log,n_mul,n_div)==False):
        return 
    
    
    n_dim1 = n_id+n_log+n_exp+n_mul*2+n_div*2
    n_dim1p= n_id+n_log+n_exp+n_mul+n_div
    n_dim0 = n_density*(n_parameter) # couple in first layer ? 
    
    layer = 0
    name = "layer_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_parameter):
            text_file.write("seps"+str(i)+'=Symbol(\'eps'+str(i)+'\')'+"\n")
        for i in range (n_density*(n_parameter)):
            text_file.write("sn"+str(i)+'=Symbol(\'n'+str(i)+'\')'+"\n")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"= Matrix("+str(n_dim0)+",1,[")
        for eps in range (0,n_parameter):
            for i in range ((eps)*n_density,(eps+1)*n_density):
                text_file.write("seps"+str(eps)+"*sn"+str(i)+",")
        text_file.write("])")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())  
    
    layer = 1
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_dim1*n_dim0):
            text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
        for i in range (n_div):
            text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
        text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim0)+",[")
        for i in range (n_dim1*n_dim0-1):
            text_file.write("sa"+str(layer)+"L"+str(i)+",")
        text_file.write("sa"+str(layer)+"L"+str(n_dim1*n_dim0-1)+"])")
        text_file.write("\n")
        #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
        #for i in range (n_dim1-1):
        #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
        #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
        text_file.write('\n')
        text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1))

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genXp_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
        for i in range(n_id):
            text_file.write("X"+str(layer)+"["+str(i)+"],")
        for i in range(n_id,n_id+n_exp):
            text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp,n_id+n_exp+n_log):
            text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
        base=n_id+n_exp+n_log+n_mul*2
        for i in range(base,base+n_div*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
            if(i!=n_id+n_exp+n_log+n_mul*2+n_div*2-2):
                text_file.write(",")
        text_file.write("])")

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode)   

    for L in range (2,n_layer):
        layer = L
        print("processing "+str(L)+"th layer")
        name = "genX_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            for i in range (n_dim1p*n_dim1):
                text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
            for i in range (n_div):
                text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
            text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim1p)+",[")
            for i in range (n_dim1*n_dim1p-1):
                text_file.write("sa"+str(layer)+"L"+str(i)+",")
            text_file.write("sa"+str(layer)+"L"+str(n_dim1p*n_dim1-1)+"])")
            text_file.write("\n")
            #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
            #for i in range (n_dim1-1):
            #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
            #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
            text_file.write('\n')
            #text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p"+"+B"+str(layer))
            text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
        name = "genXp_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
            for i in range(n_id):
                text_file.write("X"+str(layer)+"["+str(i)+"],")
            for i in range(n_id,n_id+n_exp):
                text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp,n_id+n_exp+n_log):
                text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
            base=n_id+n_exp+n_log+n_mul*2
            for i in range(base,base+n_div*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
                if(i!=n_id+n_exp+n_log+n_mul*2+n_div*2-2):
                    text_file.write(",")
            text_file.write("])")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
    
    name = "free_energy_density.dat"
    with open(name, "w") as text_file:
        text_file.write("fed=")
        for i in range(n_dim1p-1):
            text_file.write("X"+str(n_layer-1)+"p"+"["+str(i)+"]+")
        text_file.write("X"+str(n_layer-1)+"p"+"["+str(n_dim1p-1)+"]")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    
    return fed

#--------------------------------------------
#---------------------------------------------------------------------------------
def equation_gen_sparse(n_layer,n_density,n_parameter,n_id = 2,n_log = 2
                 ,n_exp = 0,n_mul = 1,n_div = 1,n_density_eps=0):
    if(chcek(n_layer,n_density,n_parameter,n_id,n_exp,n_log,n_mul,n_div)==False):
        return 
    if(n_exp!=0 and n_log!=0):
        print("mix exp and log may go carzy")
    if(n_log>=2):
        print("more than 1 log may go carzy")
    N_div = n_div
    #n_div=0
    n_dim1 = n_id+n_log+n_exp+n_mul*2+n_div*2
    n_dim1p= n_id+n_log+n_exp+n_mul+n_div
    n_dim0 = n_density+(n_parameter*n_density_eps) # couple in first layer ? 
    if(n_parameter>=1 and n_density_eps<=0):
        print("coupling wrong\t","n_parameter=",n_parameter,"\t","n_density_eps\t=\t",n_density_eps,"\n")
        
    layer = 0
    name = "layer_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_parameter):
            text_file.write("seps"+str(i)+'=Symbol(\'eps'+str(i)+'\')'+"\n")
        for i in range (n_density+(n_parameter*n_density_eps)):
            text_file.write("sn"+str(i)+'=Symbol(\'n'+str(i)+'\')'+"\n")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"= Matrix("+str(n_dim0)+",1,[")
        for i in range (n_density):
            text_file.write("sn"+str(i)+",")
        for eps in range (0,n_parameter):
            for i in range (n_density+(eps)*n_density_eps,n_density+(eps+1)*n_density_eps):
                text_file.write("seps"+str(eps)+"*sn"+str(i)+",")
        text_file.write("])")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())  
    
    layer = 1
    name = "genX_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        for i in range (n_dim1*n_dim0):
            text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
        for i in range (n_div):
            text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
        text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim0)+",[")
        for i in range (n_dim1*n_dim0):
            text_file.write("sa"+str(layer)+"L"+str(i)+",")
        text_file.write("])")
        text_file.write("\n")
        #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
        #for i in range (n_dim1-1):
        #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
        #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
        text_file.write('\n')
        text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1))

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    name = "genXp_"+str(layer)+".dat"
    with open(name, "w") as text_file:
        text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
        for i in range(n_id):
            text_file.write("X"+str(layer)+"["+str(i)+"],")
        for i in range(n_id,n_id+n_exp):
            text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp,n_id+n_exp+n_log):
            text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
        for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
            text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
        base=n_id+n_exp+n_log+n_mul*2
        for i in range(base,base+n_div*2,2):
            #text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")
            text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+1)")
            if(i!=n_id+n_exp+n_log+n_mul*2+n_div*2-2):
                text_file.write(",")
        text_file.write("])")

    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode)   

    for L in range (2,n_layer):
        if(L==n_layer-1):
            n_div=N_div
            n_dim1 = n_id+n_log+n_exp+n_mul*2+n_div*2
            #n_dim1p= n_id+n_log+n_exp+n_mul+n_div
        layer = L
        print("processing "+str(L)+"th layer")
        name = "genX_"+str(layer)+".dat"
        with open(name, "w") as text_file:
            for i in range (n_dim1p*n_dim1):
                text_file.write("sa"+str(layer)+"L"+str(i)+'=Symbol(\'a'+str(layer)+"L"+str(i)+'\')'+"\n")
            for i in range (n_div):
                text_file.write("sb"+str(layer)+"L"+str(i)+'=Symbol(\'b'+str(layer)+"L"+str(i)+'\')'+"\n")
            text_file.write("W"+str(layer)+"=Matrix("+str(n_dim1)+","+str(n_dim1p)+",[")
            for i in range (n_dim1*n_dim1p):
                if(i%n_dim1p<n_id+n_log+n_exp and int(i/n_dim1p)<n_id+n_log+n_exp):
                #    if(L==2 or L==3):
                     text_file.write("0"+",")
                #    elif(i%n_dim1p<n_id and int(i/n_dim1p)<n_id):
                #        text_file.write("sa"+str(layer)+"L"+str(i)+",")    
                #elif(i%n_dim1p>=n_id+n_log+n_exp and int(i/n_dim1p)>=n_id+n_log+n_exp):
                #    text_file.write("0"+",")
                #elif(int(i/n_dim1p)>=n_id+n_log+n_exp+n_mul*2 and i%n_dim1p>=n_id+n_exp+n_log+n_mul and (int(i/n_dim1p)-(n_id+n_log+n_exp+n_mul*2))%2==1):
                #    text_file.write("0"+",")
                    
                else:
                    text_file.write("sa"+str(layer)+"L"+str(i)+",")
            text_file.write("])")
            text_file.write("\n")
            #text_file.write("B"+str(layer)+"=Matrix("+str(n_dim1)+","+str(1)+",[")
            #for i in range (n_dim1-1):
            #    text_file.write("sb"+str(layer)+"L"+str(i)+",")
            #text_file.write("sb"+str(layer)+"L"+str(n_dim1-1)+"])")
            text_file.write('\n')
            #text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p"+"+B"+str(layer))
            text_file.write('X'+str(layer)+"=W"+str(layer)+"*"+"X"+str(layer-1)+"p")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
        name = "genXp_"+str(layer)+".dat"
        if(L==n_layer-1):
            n_dim1p= n_id+n_log+n_exp+n_mul+n_div
        with open(name, "w") as text_file:
            text_file.write("X"+str(layer)+"p=Matrix("+str(n_dim1p)+","+str(1)+","+"[")
            for i in range(n_id):
                text_file.write("X"+str(layer)+"["+str(i)+"],")
            for i in range(n_id,n_id+n_exp):
                text_file.write("-1+exp(X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp,n_id+n_exp+n_log):
                text_file.write("log(1+X"+str(layer)+"["+str(i)+"]),")
            for i in range(n_id+n_exp+n_log,n_id+n_exp+n_log+n_mul*2,2):
                text_file.write("X"+str(layer)+"["+str(i)+"]*"+"X"+str(layer)+"["+str(i+1)+"],")
            base=n_id+n_exp+n_log+n_mul*2
            for i in range(base,base+n_div*2,2):
                #text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+sb"+str(layer)+"L"+str(int((i-base)/2))+")")  
                text_file.write("X"+str(layer)+"["+str(i)+"]/"+"(X"+str(layer)+"["+str(i+1)+"]+1)")
                if(i!=n_id+n_exp+n_log+n_mul*2+n_div*2-2):
                    text_file.write(",")
            text_file.write("])")

        tmp = open(name, "r")
        tmpcode = tmp.read()
        exec(tmpcode, locals(), globals())
    
    name = "free_energy_density.dat"
    with open(name, "w") as text_file:
        text_file.write("fed=")
        for i in range(n_dim1p-1):
            text_file.write("X"+str(n_layer-1)+"p"+"["+str(i)+"]+")
        text_file.write("X"+str(n_layer-1)+"p"+"["+str(n_dim1p-1)+"]")
    tmp = open(name, "r")
    tmpcode = tmp.read()
    exec(tmpcode, locals(), globals())
    
    return fed
