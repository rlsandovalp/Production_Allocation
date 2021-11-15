import numpy as np
from scipy.optimize import minimize, nnls, fmin_slsqp
import matplotlib.pyplot as plt
import math

# plt.style.use(['science','nature'])

# FUNCTIONS OUR NOVEL DECONVOLUTION ALGORITHM

def ouropt_1(unknowns, def_operations, em_peaks_mean, um_peaks_mean, tipo, cv, ignorar, nEM):
    # Function to compute the loss function for a given value of cv. This function is used to obtain optimal values of x and MR.
    mini = 0
    for q, _ in enumerate(def_operations):
        if len(set(def_operations[q][0]).intersection(ignorar))+len(set(def_operations[q][1]).intersection(ignorar))>0: 
            continue
        Rij, rij = compute_ratios_our(0, 0, unknowns, nEM, em_peaks_mean, um_peaks_mean, q, def_operations,tipo)
        phi = (1/cv)*(1+rij/Rij)/(math.sqrt(2*(1+(rij/Rij)**2)))
        gam = math.sqrt(math.pi)*phi*math.exp(phi**2)*math.erf(phi)
        uno = math.log(1+(rij/Rij)**2)
        dos = math.log(1+gam)
        mini = mini + uno - dos
    return mini

def ouropt_2(cv, unknowns, def_operations, em_peaks_mean, um_peaks_mean, tipo, ignorar, nEM):
    # Function to compute the loss function for a given value of x, and MR. This function is used to obtain optimal values of cv.
    mini = 0
    ignorados = 0
    for q, _ in enumerate(def_operations):
        if len(set(def_operations[q][0]).intersection(ignorar))+len(set(def_operations[q][1]).intersection(ignorar))>0: 
            ignorados = ignorados + 1
            continue
        Rij, rij = compute_ratios_our(0, 0, unknowns, nEM, em_peaks_mean, um_peaks_mean, q, def_operations,tipo)
        phi = (1/cv)*(1+rij/Rij)/(math.sqrt(2*(1+(rij/Rij)**2)))
        gam = math.sqrt(math.pi)*phi*math.exp(phi**2)*math.erf(phi)
        uno = math.log(1+gam)
        mini = mini + uno
    mini = -mini + (len(def_operations)-ignorados)/(cv**2)
    return mini

def compute_ratios_our(Num, Den, unknowns, nEM, em_peaks_mean, um_peaks_mean, q, def_operations, tipo):
    # Function to read the experimental ratios and compute the analytical ratios for OUR objective function
    um_peaks_mean = np.reshape(um_peaks_mean, -1)
    if tipo[q] == '1/1':
        for em in range(nEM-1):
            Num = Num+unknowns[em]*unknowns[-nEM+1+em]*em_peaks_mean[em,def_operations[q][0][0]-1]
            Den = Den+unknowns[em]*unknowns[-nEM+1+em]*em_peaks_mean[em,def_operations[q][1][0]-1]
        Num = Num + unknowns[em+1]*em_peaks_mean[em+1,def_operations[q][0][0]-1]
        Den = Den + unknowns[em+1]*em_peaks_mean[em+1,def_operations[q][1][0]-1]
        Hi = um_peaks_mean[def_operations[q][0][0]-1]
        Hj = um_peaks_mean[def_operations[q][1][0]-1]
        Rij = Hi/Hj
    if tipo[q] == '1/2':
        for em in range(nEM-1):
            Num = Num+unknowns[em]*unknowns[-nEM+1+em]*em_peaks_mean[em,def_operations[q][0][0]-1]
            Den = Den+unknowns[em]*unknowns[-nEM+1+em]*(em_peaks_mean[em,def_operations[q][1][0]-1]+em_peaks_mean[em,def_operations[q][1][1]-1])
        Num = Num + (100-np.sum(unknowns[0:nEM-1]))*em_peaks_mean[em+1,def_operations[q][0][0]-1]
        Den = Den + (100-np.sum(unknowns[0:nEM-1]))*(em_peaks_mean[em+1,def_operations[q][1][0]-1]+em_peaks_mean[em+1,def_operations[q][1][1]-1])
        Hi = um_peaks_mean[def_operations[q][0][0]-1]
        Hj = um_peaks_mean[def_operations[q][1][0]-1]
        Hk = um_peaks_mean[def_operations[q][1][1]-1]
        Rij = Hi/(Hj+Hk)
    if tipo[q] == '1/3':
        for em in range(nEM-1):
            Num = Num+unknowns[em]*unknowns[-nEM+1+em]*em_peaks_mean[em,def_operations[q][0][0]-1]
            Den = Den+unknowns[em]*unknowns[-nEM+1+em]*(em_peaks_mean[em,def_operations[q][1][0]-1]+em_peaks_mean[em,def_operations[q][1][1]-1]+em_peaks_mean[em,def_operations[q][1][2]-1])
        Num = Num + (100-np.sum(unknowns[0:nEM-1]))*em_peaks_mean[em+1,def_operations[q][0][0]-1]
        Den = Den + (100-np.sum(unknowns[0:nEM-1]))*(em_peaks_mean[em+1,def_operations[q][1][0]-1]+em_peaks_mean[em+1,def_operations[q][1][1]-1]+em_peaks_mean[em,def_operations[q][1][2]-1])
        Hi = um_peaks_mean[def_operations[q][0][0]-1]
        Hj = um_peaks_mean[def_operations[q][1][0]-1]
        Hk = um_peaks_mean[def_operations[q][1][1]-1]
        Hl = um_peaks_mean[def_operations[q][1][2]-1]
        Rij = Hi/(Hj+Hk+Hl)
    if tipo[q] == '2/3':
        for em in range(nEM-1):
            Num = Num+unknowns[em]*unknowns[-nEM+1+em]*(em_peaks_mean[em,def_operations[q][0][0]-1]+em_peaks_mean[em,def_operations[q][0][1]-1])
            Den = Den+unknowns[em]*unknowns[-nEM+1+em]*(em_peaks_mean[em,def_operations[q][1][0]-1]+em_peaks_mean[em,def_operations[q][1][1]-1]+em_peaks_mean[em,def_operations[q][1][2]-1])
        Num = Num + (100-np.sum(unknowns[0:nEM-1]))*(em_peaks_mean[em+1,def_operations[q][0][0]-1]+em_peaks_mean[em+1,def_operations[q][0][1]-1])
        Den = Den + (100-np.sum(unknowns[0:nEM-1]))*(em_peaks_mean[em+1,def_operations[q][1][0]-1]+em_peaks_mean[em+1,def_operations[q][1][1]-1]+em_peaks_mean[em+1,def_operations[q][1][2]-1])
        Hi = um_peaks_mean[def_operations[q][0][0]-1]
        Hj = um_peaks_mean[def_operations[q][0][1]-1]
        Hk = um_peaks_mean[def_operations[q][1][0]-1]
        Hl = um_peaks_mean[def_operations[q][1][1]-1]
        Hm = um_peaks_mean[def_operations[q][1][2]-1]
        Rij = (Hi+Hj)/(Hk+Hl+Hm)
    rij=Num/Den
    return Rij, rij

def convert_operations(operations_path):
    # Function to convert the operations file to readable tables nratios, w_r
    with open(operations_path, 'r') as f: operations = f.read().splitlines()
    def_operations = []
    new_list = []
    n_p = []
    for q, operation in enumerate(operations):
        def_operations.append([])
        d=0
        def_operations[q].append([])
        multidigit = ''
        for character in operation:
            if character == '(':
                next
            elif character.isnumeric():
                multidigit += character
            elif character == '+' or character == '-' or character == ')':
                def_operations[q][d].append(int(multidigit))
                multidigit = ''
            elif character == '/':  
                d=1
                def_operations[q].append([])

    for h in def_operations:
        for i in h:
            for j in i:
                new_list.append(j)
    a = len(new_list)
    for i in range(max(new_list)):
        n_p.append(new_list.count(i+1))
    
    #Weight and type of each ratio (uncertainty propagation is computed with different formulas depending on the type of ratio)
    tipo = []
    for operation in def_operations:
        tipo.append(str(len(operation[0]))+'/'+str(len(operation[1])))
    return def_operations, tipo


# FUNCTIONS NOUVELLE IMPROVED DECONVOLUTION ALGORITHM

def preprocess_nouvelle(table,pp,max_cv_peaks,max_cv_samples):
    # print('- '*50)
    if pp == 1:
        #################################     Delete the peaks whose CV is larger than threshold value ##############################
        auxiliar_df = (table.std(level=0, ddof = 1)*100/table.mean(level=0)).max(axis = 0)
        ignorar = auxiliar_df[auxiliar_df>max_cv_peaks].index

        #################################     Identify repetitions causing large CV and delete them ##############################
        auxiliar_df = (table.std(level=0)*100/table.mean(level=0)).mean(axis = 1)
        revisar = auxiliar_df[auxiliar_df>max_cv_samples].index
        table.reset_index(inplace = True)
        for i in revisar:
            counter = np.argmax(np.mean(abs(table[table['Mix']==i].values[:,1:]-np.mean(table[table['Mix']==i].values[:,1:],axis = 0)), axis = 1))
            position_i = auxiliar_df.index.tolist().index(i)
            rep = table[table['Mix']==i].values.shape[0]
            table.drop(position_i*rep + counter, inplace = True)
        table.set_index('Mix', inplace = True)
    return(table,list(map(int, ignorar.tolist())))

def ratios(operations_path,dataset):
    # n_p has the number of occurrences of each peak
    # ratios results has the results of the ratios ... dah
    # w_p has the weight of the ratios
    # Def operations is a list that in each entry has the index of the peaks used for each ratio
    with open(operations_path, 'r') as f: operations = f.read().splitlines()
    dataset_access_format = "dataset['{}']"
    ratios_results = dataset.filter(['Mix'])
    def_operations = []
    new_list = []
    n_p = []
    for q, operation in enumerate(operations):
        def_operations.append([])
        d=0
        def_operations[q].append([])
        multidigit = ''
        for character in operation:
            if character == '(':
                next
            elif character.isnumeric():
                multidigit += character
            elif character == '+' or character == '-' or character == ')':
                def_operations[q][d].append(int(multidigit))
                multidigit = ''
            elif character == '/':  
                d=1
                def_operations[q].append([])
    for q, operation in enumerate(def_operations):
        numerator = ''
        denominator = ''
        for pos, x in enumerate(def_operations[q][0]):
            if pos == 0:
                numerator = numerator + dataset_access_format.format((x))
            elif pos != 0:
                numerator = numerator + '+'
                numerator = numerator + dataset_access_format.format((x))
        for pos, x in enumerate(def_operations[q][1]):
            if pos == 0:
                denominator = denominator + dataset_access_format.format((x))
            elif pos != 0:
                denominator = denominator + '+'
                denominator = denominator + dataset_access_format.format((x))
        ratios_results[operations[q]] = eval('('+numerator+')/('+denominator+')').values

    for h in def_operations:
        for i in h:
            for j in i:
                new_list.append(j)
    a = len(new_list)
    for i in range(max(new_list)):
        n_p.append(new_list.count(i+1))
    
    #Weight and type of each ratio (uncertainty propagation is computed with different formulas depending on the type of ratio)
    w_r=[]
    tipo = []
    for operation in def_operations:
        tipo.append(str(len(operation[0]))+'/'+str(len(operation[1])))
    for q, tip in enumerate(tipo):
        a = len(def_operations[q][0]+def_operations[q][1])
        if tip == '1/1':
            w_r.append(a/(n_p[def_operations[q][0][0]-1]+n_p[def_operations[q][1][0]-1]))
        elif tip == '1/2':
            w_r.append(a/(n_p[def_operations[q][0][0]-1]+n_p[def_operations[q][1][0]-1]+n_p[def_operations[q][1][1]-1]))
        elif tip == '1/3':
            w_r.append(a/(n_p[def_operations[q][0][0]-1]+n_p[def_operations[q][1][0]-1]+n_p[def_operations[q][1][1]-1]+n_p[def_operations[q][1][2]-1]))
        elif tip == '2/1':
            w_r.append(3/(n_p[def_operations[q][0][0]-1]+n_p[def_operations[q][0][1]-1]+n_p[def_operations[q][1][0]-1]))
        elif tip == '2/3':
            w_r.append(5/(n_p[def_operations[q][0][0]-1]+n_p[def_operations[q][0][1]-1]+n_p[def_operations[q][1][0]-1]+n_p[def_operations[q][1][1]-1]+n_p[def_operations[q][1][2]-1]))
    return ratios_results, w_r, tipo, def_operations

def model(MR, Mix, nratios, proportions_values, em_peaks_mean_values, def_operations, tipo, EM): 
    # Evaluates the ratios mixture model for the unkwnown mixtures for given values of MR
    Mix_ratios_modeled=np.zeros((Mix,nratios))
    for sin in range(Mix):
        for q, _ in enumerate(def_operations):
            Num, Den = 0, 0
            if tipo[q] == '1/1':
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
                Num = Num + proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den + proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
            elif (tipo[q] == '1/2') or (tipo[q] == '1/2a') or (tipo[q] == '1/2b'):
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1])
                Num = Num+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1])
            elif (tipo[q] == '2/1') or (tipo[q] == '2/1a') or (tipo[q] == '2/1b'):
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                    Den = Den+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
                Num = Num+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
                Den = Den+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
            elif (tipo[q] == '1/3') or (tipo[q] == '1/3a') or (tipo[q] == '1/3b') or (tipo[q] == '1/3c'):
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
                Num = Num+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
            elif (tipo[q] == '2/3') or (tipo[q] == '2/3a') or (tipo[q] == '2/3b') or (tipo[q] == '2/3c') or (tipo[q] == '2/3d') or (tipo[q] == '2/3e') or (tipo[q] == '2/3f') or (tipo[q] == '2/3g') or (tipo[q] == '2/3h') or (tipo[q] == '2/3i') or (tipo[q] == '2/3j') or (tipo[q] == '2/3k') or (tipo[q] == '2/3l'):
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
                Num = Num+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
            Mix_ratios_modeled[sin,q]=Num/Den
    return Mix_ratios_modeled

def improved_loss_function_1(MR, Mix, nratios, proportions_values, em_peaks_mean_values, def_operations, tipo, nEM, syn_mix_ratios_mean, ignorar):
    # Function to compute the loss function for a given value of MR. This function is used to obtain optimal values of MR.
    Mix_ratios_modeled = np.zeros((Mix,nratios))
    for sin in range(Mix):
        for q, _ in enumerate(def_operations):
            if len(set(def_operations[q][0]).intersection(ignorar))+len(set(def_operations[q][1]).intersection(ignorar))>0: continue
            Num, Den = 0, 0
            if tipo[q] == '1/1':
                for em in range(nEM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
                Num = Num + proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den + proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
            elif tipo[q] == '1/2':
                for em in range(nEM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1])
                Num = Num+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1])
            elif tipo[q] == '2/1':
                for em in range(nEM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                    Den = Den+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
                Num = Num+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
                Den = Den+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
            elif tipo[q] == '1/3':
                for em in range(nEM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
                Num = Num+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
            elif tipo[q] == '2/3':
                for em in range(nEM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
                Num = Num+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
            Mix_ratios_modeled[sin,q]=Num/Den
    mini = np.log(1+(1/syn_mix_ratios_mean*(Mix_ratios_modeled-syn_mix_ratios_mean))**2)
    return mini.sum()

def improved_loss_function_2(xi, nratios, em_peaks_mean_values, tipo, nEM, MR, def_operations, UM_ratios, ignorar):
    # Function to compute the loss function for a given value of x. This function is used to obtain optimal values of x.
    global mini
    CR=np.zeros(nratios)
    for q, _ in enumerate(def_operations):
        if len(set(def_operations[q][0]).intersection(ignorar))+len(set(def_operations[q][1]).intersection(ignorar))>0: continue
        Num, Den = 0, 0
        if tipo[q] == '1/1':
            for em in range(nEM-1):
                Num = Num+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                Den = Den+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
            Num = Num + (100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
            Den = Den + (100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
        elif (tipo[q] == '1/2') or (tipo[q] == '1/2a') or (tipo[q] == '1/2b'):
            for em in range(nEM-1):
                Num = Num+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                Den = Den+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1])
            Num = Num+(100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
            Den = Den+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1])
        elif (tipo[q] == '2/1') or (tipo[q] == '2/1a') or (tipo[q] == '2/1b'):
            for em in range(nEM-1):
                Num = Num+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                Den = Den+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
            Num = Num+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
            Den = Den+(100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
        elif (tipo[q] == '1/3') or (tipo[q] == '1/3a') or (tipo[q] == '1/3b') or (tipo[q] == '1/3c'):
            for em in range(nEM-1):
                Num = Num+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                Den = Den+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
            Num = Num+(100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
            Den = Den+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
        elif (tipo[q] == '2/3') or (tipo[q] == '2/3a') or (tipo[q] == '2/3b') or (tipo[q] == '2/3c') or (tipo[q] == '2/3d') or (tipo[q] == '2/3e') or (tipo[q] == '2/3f') or (tipo[q] == '2/3g') or (tipo[q] == '2/3h') or (tipo[q] == '2/3i') or (tipo[q] == '2/3j') or (tipo[q] == '2/3k') or (tipo[q] == '2/3l'):
            for em in range(nEM-1):
                Num = Num+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                Den = Den+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
            Num = Num+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
            Den = Den+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
        CR[q]=Num/Den
    mini=np.log(1+(1/UM_ratios*(UM_ratios-CR))**2)
    return mini.sum()


# FUNCTIONS ORIGINAL NOUVELLE DECONVOLUTION ALGORITHM

def ratios_sd(operations_path, def_operations, mix_ratios_mean, syn_mix_peaks_mean, syn_mix_peaks_variance, tipo):
    # Compute standard deviation of ratios following a Taylor expansion
    with open(operations_path, 'r') as f: operations = f.read().splitlines()
    mix_ratios_var = mix_ratios_mean.filter(['Mix'])
    syn_mix_peaks_mean_values = syn_mix_peaks_mean.values
    syn_mix_peaks_variance_values = syn_mix_peaks_variance.values
    for q, operation in enumerate(operations):
        if tipo[q] == '1/1':
            hi = syn_mix_peaks_mean_values[:,def_operations[q][0][0]-1]
            hj = syn_mix_peaks_mean_values[:,def_operations[q][1][0]-1]
            si = syn_mix_peaks_variance_values[:,def_operations[q][0][0]-1]
            sj = syn_mix_peaks_variance_values[:,def_operations[q][1][0]-1]
            mix_ratios_var[operation] = (1/(hj**2))*(si+((hi/hj)**2)*sj)
        elif tipo[q] == '1/2':
            hi = syn_mix_peaks_mean_values[:,def_operations[q][0][0]-1]
            hj = syn_mix_peaks_mean_values[:,def_operations[q][1][0]-1]
            hk = syn_mix_peaks_mean_values[:,def_operations[q][1][1]-1]
            si = syn_mix_peaks_variance_values[:,def_operations[q][0][0]-1]
            sj = syn_mix_peaks_variance_values[:,def_operations[q][1][0]-1]
            sk = syn_mix_peaks_variance_values[:,def_operations[q][1][1]-1]
            fr = 1/((hj+hk)**2)
            mix_ratios_var[operation] = (1/(hj+hk)**2)*(si+hi**2*(sj+sk)/(hj+hk)**2)
        elif tipo[q] == '1/3':
            hi = syn_mix_peaks_mean_values[:,def_operations[q][0][0]-1]
            hj = syn_mix_peaks_mean_values[:,def_operations[q][1][0]-1]
            hk = syn_mix_peaks_mean_values[:,def_operations[q][1][1]-1]
            hl = syn_mix_peaks_mean_values[:,def_operations[q][1][2]-1]
            si = syn_mix_peaks_variance_values[:,def_operations[q][0][0]-1]
            sj = syn_mix_peaks_variance_values[:,def_operations[q][1][0]-1]
            sk = syn_mix_peaks_variance_values[:,def_operations[q][1][1]-1]
            sl = syn_mix_peaks_variance_values[:,def_operations[q][1][2]-1]
            fr = 1/((hj+hk+hl)**2)
            mix_ratios_var[operation] = (fr*(si+hi**2*fr*(sj+sk+sl)))**0.5
        elif tipo[q] == '2/1':
            hi = syn_mix_peaks_mean_values[:,def_operations[q][0][0]-1]
            hj = syn_mix_peaks_mean_values[:,def_operations[q][0][1]-1]
            hk = syn_mix_peaks_mean_values[:,def_operations[q][1][0]-1]
            si = syn_mix_peaks_variance_values[:,def_operations[q][0][0]-1]
            sj = syn_mix_peaks_variance_values[:,def_operations[q][0][1]-1]
            sk = syn_mix_peaks_variance_values[:,def_operations[q][1][0]-1]
            mix_ratios_var[operation] = (1/(hk**2)*(si+sj+sk*(hi+hj)**2))**0.5
        elif tipo[q] == '2/3':
            hi = syn_mix_peaks_mean_values[:,def_operations[q][0][0]-1]
            hj = syn_mix_peaks_mean_values[:,def_operations[q][0][1]-1]
            hk = syn_mix_peaks_mean_values[:,def_operations[q][1][0]-1]
            hl = syn_mix_peaks_mean_values[:,def_operations[q][1][1]-1]
            hm = syn_mix_peaks_mean_values[:,def_operations[q][1][2]-1]
            si = syn_mix_peaks_variance_values[:,def_operations[q][0][0]-1]
            sj = syn_mix_peaks_variance_values[:,def_operations[q][0][1]-1]
            sk = syn_mix_peaks_variance_values[:,def_operations[q][1][0]-1]
            sl = syn_mix_peaks_variance_values[:,def_operations[q][1][1]-1]
            sm = syn_mix_peaks_variance_values[:,def_operations[q][1][2]-1]
            fr = 1/((hk+hl+hm)**2)
            mix_ratios_var[operation] = (fr*(si+sj+(hi+hj)**2*fr*(sk+sl+sm)))**0.5        
    return mix_ratios_var.values**0.5

def loss_function_1(MR, Mix, nratios, proportions_values, em_peaks_mean_values, def_operations, tipo, EM, Mix_Q, mix_ratios_mean, ignorar): 
    # Function to compute the loss function for a given value of MR. This function is used to obtain optimal values of MR.
    Mix_ratios_modeled = np.zeros((Mix,nratios))
    for sin in range(Mix):
        for q, _ in enumerate(def_operations):
            if len(set(def_operations[q][0]).intersection(ignorar))+len(set(def_operations[q][1]).intersection(ignorar))>0: continue
            Num, Den = 0, 0
            if tipo[q] == '1/1':
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
                Num = Num + proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den + proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
            elif tipo[q] == '1/2':
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1])
                Num = Num+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1])
            elif tipo[q] == '2/1':
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                    Den = Den+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
                Num = Num+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
                Den = Den+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
            elif tipo[q] == '1/3':
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
                Num = Num+proportions_values[sin,em+1]*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
            elif tipo[q] == '2/3':
                for em in range(EM-1):
                    Num = Num+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                    Den = Den+MR[em]*proportions_values[sin,em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
                Num = Num+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
                Den = Den+proportions_values[sin,em+1]*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
            Mix_ratios_modeled[sin,q]=Num/Den
    mini=np.log(1+0.5*(Mix_Q*(Mix_ratios_modeled-mix_ratios_mean))**2)
    return mini.sum()

def loss_function_2(xi, nratios, em_peaks_mean_values, tipo, EM, MR, def_operations, Mix_Q, UM_ratios, ignorar):
    # Function to compute the loss function for a given value of x. This function is used to obtain optimal values of x.
    CR = np.zeros(nratios)
    for q, _ in enumerate(def_operations):
        if len(set(def_operations[q][0]).intersection(ignorar))+len(set(def_operations[q][1]).intersection(ignorar))>0: continue
        Num, Den = 0, 0
        if tipo[q] == '1/1':
            for em in range(EM-1):
                Num = Num+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                Den = Den+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
            Num = Num + (100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
            Den = Den + (100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
        elif (tipo[q] == '1/2') or (tipo[q] == '1/2a') or (tipo[q] == '1/2b'):
            for em in range(EM-1):
                Num = Num+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                Den = Den+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1])
            Num = Num+(100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
            Den = Den+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1])
        elif (tipo[q] == '2/1') or (tipo[q] == '2/1a') or (tipo[q] == '2/1b'):
            for em in range(EM-1):
                Num = Num+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                Den = Den+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][1][0]-1]
            Num = Num+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
            Den = Den+(100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][1][0]-1]
        elif (tipo[q] == '1/3') or (tipo[q] == '1/3a') or (tipo[q] == '1/3b') or (tipo[q] == '1/3c'):
            for em in range(EM-1):
                Num = Num+MR[em]*xi[em]*em_peaks_mean_values[em,def_operations[q][0][0]-1]
                Den = Den+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
            Num = Num+(100-np.sum(xi))*em_peaks_mean_values[em+1,def_operations[q][0][0]-1]
            Den = Den+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
        elif (tipo[q] == '2/3') or (tipo[q] == '2/3a') or (tipo[q] == '2/3b') or (tipo[q] == '2/3c') or (tipo[q] == '2/3d') or (tipo[q] == '2/3e') or (tipo[q] == '2/3f') or (tipo[q] == '2/3g') or (tipo[q] == '2/3h') or (tipo[q] == '2/3i') or (tipo[q] == '2/3j') or (tipo[q] == '2/3k') or (tipo[q] == '2/3l'):
            for em in range(EM-1):
                Num = Num+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][0][0]-1]+em_peaks_mean_values[em,def_operations[q][0][1]-1])
                Den = Den+MR[em]*xi[em]*(em_peaks_mean_values[em,def_operations[q][1][0]-1]+em_peaks_mean_values[em,def_operations[q][1][1]-1]+em_peaks_mean_values[em,def_operations[q][1][2]-1])
            Num = Num+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][0][0]-1]+em_peaks_mean_values[em+1,def_operations[q][0][1]-1])
            Den = Den+(100-np.sum(xi))*(em_peaks_mean_values[em+1,def_operations[q][1][0]-1]+em_peaks_mean_values[em+1,def_operations[q][1][1]-1]+em_peaks_mean_values[em+1,def_operations[q][1][2]-1])
        CR[q]=Num/Den
    mini=np.log(1+0.5*(Mix_Q*(UM_ratios-CR))**2)
    return mini.sum()


# f_normalize normalize a given set of peaks based on the parameter normalize and separate the end-members from the mixture
def f_normalize(normalize,peaks_averaged,end_members):
    if normalize == 1:                                
        peaks_normalized = peaks_averaged/np.mean(peaks_averaged, axis=0)
    else:
        peaks_normalized = peaks_averaged
    return np.transpose(peaks_normalized[0:end_members,:]), np.transpose(peaks_normalized[end_members:,:])

# f_normalize normalize for ALS
def f_normalize_als(normalize,b):
    if normalize == 1:
        for i in range(b.shape[0]):                                
            b[i,:] = b[i,:]/np.max(b[i,:])
    else:
        b = b
    return b

# f_calcula performs least squares and save the info in the selected lists
def f_calcula(p_emr,p_mr,saving,end_members):
    calcula = np.linalg.lstsq(p_emr, p_mr)[0]
    # np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(p_emr),p_emr)),np.transpose(p_emr)),p_mr)
    for l in range(end_members):
        saving[l].append(calcula[l])
    return(saving)

# f_calcula constrained and bounded
def f_calcula_opt(p_emr,p_mr,saving,end_members,x0,residuals,eq_cond,bounds):
    A = p_emr
    b = p_mr
    calcula = fmin_slsqp(residuals, x0, bounds = bounds, eqcons = [eq_cond], args = (A, b), iprint = -1)
    for l in range(end_members):
        saving[l].append(calcula[l])
    return(saving)

# preprocess delete rows with high average cv and repetitions causing large cv
def preprocess(table,pp,max_cv_peaks,max_cv_samples):
    if pp == 1:
        #################################     Delete the peaks whose CV is larger than threshold value ##############################
        auxiliar_df = ((table.std(level=0, ddof = 1))*100/table.mean(level=0)).max(axis = 0)
        matar = auxiliar_df[auxiliar_df>max_cv_peaks].index
        for i in matar: 
            table.drop(i, axis = 1, inplace = True)

        #################################     Identify repetitions causing large CV and delete them ##############################
        auxiliar_df = ((table.std(level=0))*100/table.mean(level=0)).mean(axis = 1)
        revisar = auxiliar_df[auxiliar_df>max_cv_samples].index
        table.reset_index(inplace = True)
        for i in revisar:
            counter = np.argmax(np.mean(abs(table[table['Mix']==i].values[:,1:]-np.mean(table[table['Mix']==i].values[:,1:],axis = 0)), axis = 1))
            position_i = auxiliar_df.index.tolist().index(i)
            rep = table[table['Mix']==i].values.shape[0]
            table.drop(position_i*rep + counter, inplace = True)
        table.set_index('Mix')
    return(table)

# preprocess nouvelle
def preprocess_nouvelle(table,pp,max_cv_peaks,max_cv_samples):
    # print('- '*50)
    if pp == 1:
        #################################     Delete the peaks whose CV is larger than threshold value ##############################
        auxiliar_df = (table.std(level=0, ddof = 1)*100/table.mean(level=0)).max(axis = 0)
        ignorar = auxiliar_df[auxiliar_df>max_cv_peaks].index

        #################################     Identify repetitions causing large CV and delete them ##############################
        auxiliar_df = (table.std(level=0)*100/table.mean(level=0)).mean(axis = 1)
        revisar = auxiliar_df[auxiliar_df>max_cv_samples].index
        table.reset_index(inplace = True)
        for i in revisar:
            counter = np.argmax(np.mean(abs(table[table['Mix']==i].values[:,1:]-np.mean(table[table['Mix']==i].values[:,1:],axis = 0)), axis = 1))
            position_i = auxiliar_df.index.tolist().index(i)
            rep = table[table['Mix']==i].values.shape[0]
            table.drop(position_i*rep + counter, inplace = True)
        table.set_index('Mix', inplace = True)
    return(table,list(map(int, ignorar.tolist())))

# preprocess nouvelle
def preprocess_our(table,pp,max_cv_samples, max_cv_peaks):
    # print('- '*50)
    if pp == 1:
        #################################     Delete the peaks whose CV is larger than threshold value ##############################
        auxiliar_df = ((table.std(level=0, ddof = 1))*100/table.mean(level=0)).max(axis = 0)
        ignorar = auxiliar_df[auxiliar_df>max_cv_peaks].index

        #################################     Identify repetitions causing large CV and delete them ##############################
        auxiliar_df = ((table.std(level=0))*100/table.mean(level=0)).mean(axis = 1)
        revisar = auxiliar_df[auxiliar_df>max_cv_samples].index
        table.reset_index(inplace = True)
        for i in revisar:
            counter = np.argmax(np.mean(abs(table[table['Mix']==i].values[:,1:]-np.mean(table[table['Mix']==i].values[:,1:],axis = 0)), axis = 1))
            position_i = auxiliar_df.index.tolist().index(i)
            rep = table[table['Mix']==i].values.shape[0]
            table.drop(position_i*rep + counter, inplace = True)
        table.set_index('Mix', inplace = True)
    return(table,list(map(int, ignorar)))

# Computes the best approximation of X based on a L1 norm
def L1_norm(beta,p_emr,p_mr):
    return np.linalg.norm(np.matmul(p_emr,beta)-p_mr, ord=1)

# Minimizes Y-beta*X and save tha info in the selected lists (I must try to constrain)
def f_calcula_l1(p_emr, p_mr, saving,end_members):
    beta = np.ones(end_members)*(1/end_members)
    res = minimize(L1_norm,beta, args=(p_emr,p_mr))
    calcula = res.x
    for l in range(end_members):
        saving[l].append(calcula[l])
    return(saving)

# Minimizes Y-beta*X and save tha info in the selected lists (I must try to constrain)
def f_calcula_l1_cons(p_emr, p_mr, saving,end_members):
    beta = np.ones(end_members)*(1/end_members)
    bnds = ((0,1),(0,1),(0,1))
    cons = ({'type': 'ineq', 'fun': lambda beta: np.sum(beta)-0.95})
    res = minimize(L1_norm,beta, args=(p_emr,p_mr), bounds = bnds, constraints = cons)
    calcula = res.x
    for l in range(end_members):
        saving[l].append(calcula[l])
    return(saving)

# Sort a 1D array
def bogosort(x):
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    return x


# Plot confidence intervals
def plot_results(X_todos,conf_int_inf,conf_int_sup,real_results):
    factors = X_todos.shape[0]
    mixtures = X_todos.shape[1]
    r = np.zeros((factors,mixtures))
    plt.figure(figsize=(mixtures+4, 4))
    barWidth = 0.3
    yer = (conf_int_sup-conf_int_inf)/2
    for factor in range(factors):
        if factor == 0: 
            r[factor,:] = np.arange(mixtures)
        else:
            r[factor,:] = [x + barWidth for x in r[factor-1,:]]
        plt.bar(r[factor,:], X_todos[factor,:]*100, width = barWidth, edgecolor = 'black', yerr=yer[factor,:], capsize=7, label='EM'+str(factor+1))
        if factor == 0:
            plt.scatter(r[factor,:], real_results[:,factor], color = 'black', zorder=2, label = 'True values')
        else:
            plt.scatter(r[factor,:], real_results[:,factor], color = 'black', zorder=2)
    plt.xticks([r + barWidth for r in range(mixtures)], ['M'+str(mixture + 1) for mixture in range(mixtures)])
    plt.ylabel('Mass Fractions [\%]')
    plt.legend()
    plt.show()