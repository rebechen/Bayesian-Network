#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def epd(factor):
    factor2=factor.copy()
    factor2.iloc[:,0]=0
    factor2['prob']=1-factor['prob']
    frames=[factor,factor2]
    result=pd.concat(frames)
    return result


E=np.array([1,0.0003]).reshape(1,2)
E=pd.DataFrame(data=E,columns=['E','prob'])
E=epd(E)


B=np.array([1,0.0001]).reshape(1,2)
B=pd.DataFrame(data=B,columns=['B','prob'])
B=epd(B)

WA=np.array([1,0,0.4,
            1,1,0.8]).reshape(2,3)
WA=pd.DataFrame(data=WA,columns=['W','A','prob'])
WA=epd(WA)


GA=np.array([1,0,0.04,
            1,1,0.4]).reshape(2,3)
GA=pd.DataFrame(data=GA,columns=['G','A','prob'])
GA=epd(GA)


ABE=np.array([1,0,0,0.01,
            1,0,1,0.2,
            1,1,0,0.95,
            1,1,1,0.96]).reshape(4,4)
ABE=pd.DataFrame(data=ABE,columns=['A','B','E','prob'])
ABE=epd(ABE)



# # Restrict

# In[ ]:


def restrict(factor,variable,value):
    return factor.loc[factor[variable] == value].loc[:, factor.columns != variable]


# # Sumout

# In[ ]:


def sumout(factor,variable):
    print("Sumout",variable, "from:")
    print(factor)
    variables_list = list(factor.columns.values)
    variables_list.remove(variable)
    variables_list.remove("prob")
    print("Result: ")
    result = factor.groupby(variables_list).sum().reset_index()
    print(result.drop([variable],axis=1))
    print("---------------")
    return result.drop([variable],axis=1)


# # Multiply

# In[ ]:


def product_factor(factor1, factor2):
    print("Multiplying:")
    print(factor1)
    print(factor2)
    def repeat(df,var):
        copy1 = df.copy()
        copy1[var] = 1
        copy2 = df.copy()
        copy2[var] = 0
        return pd.concat([copy1, copy2])
    df1 = factor1.copy()
    df2 = factor2.copy()
    #rename column prob
    df1["prob1"] = df1["prob"]
    df1=df1.drop(["prob"], axis=1)
    df2["prob2"] = df2["prob"]
    df2 = df2.drop(["prob"], axis=1)
    

    common_columns = set(df1.columns).intersection(set(df2.columns))
    aa = []
    for item in common_columns:
        aa.append(item)
    
    #if common columns exist
    if aa != []:
        result = df1.merge(df2)
        result["prob"] = result["prob1"]*result["prob2"]
        print("Result:")
        result = result.drop(['prob1', 'prob2'], axis=1)
        print(result)
        print("---------------")
        return result
    
    #if common columns do not exist
    elif aa == []:
        df1['tmp'] = 1
        df2['tmp'] = 1

        df = pd.merge(df1, df2, on=['tmp'])
        df = df.drop('tmp', axis=1)
        df["prob"] = df["prob1"] *df["prob2"] 
        df = df.drop("prob1", axis=1)
        df = df.drop("prob2", axis=1)
        print("Result:")
        print(df)
        print("---------------")
        return df
    
        


# In[ ]:





# # Normalize

# In[ ]:


def normalize(factor):
    print("Normalize:")
    print(factor)
    df = factor.copy()
    total = sum(df["prob"])
    df["prob"] = round(df["prob"]/total,8)
    print("Result:")
    print(df)
    print("---------------")
    return df


# In[ ]:





# In[ ]:


def inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars): #"AB", "AS"
    #iterlate all hidden variables
    for var in ordered_hidden_var_list:
        cal_list =[]
        new_factor_list =[]
        #multiply all the factors involve the hidden variables then sumout
        for factor in factor_list:
            if var in list(factor.columns):
                cal_list.append(factor) 
            else:
                new_factor_list.append(factor)
        while len(cal_list) > 1:
            cal_list = [product_factor(cal_list[0],cal_list[1])] + cal_list[2:]
        new_factor_list = [sumout(cal_list[0],var)]+new_factor_list    
        factor_list = new_factor_list      
    while len(factor_list) >1:
        factor_list = [product_factor(factor_list[0],factor_list[1])] + factor_list[2:]
    result = factor_list[0]
    if evidence_vars != []:
        for item in evidence_vars:
            result = restrict(result,item[0],item[1])
        result = normalize(result)
    elif evidence_vars == []:
        result = normalize(factor_list[0])


# In[ ]:





# In[ ]:


factor_list=[]
factor_list.append(E)
factor_list.append(B)
factor_list.append(WA)
factor_list.append(ABE)
factor_list.append(GA)
ordered_hidden_var_list = ["A","B","E","G"]
evidence_vars =[]
query_variables=["W"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


factor_list=[]
factor_list.append(E)
factor_list.append(B)
factor_list.append(WA)
factor_list.append(ABE)
factor_list.append(GA)
ordered_hidden_var_list = ["A","B","E","W"]
evidence_vars =[]
query_variables=["G"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:





# In[ ]:


#p(w) = 0.40406


# In[ ]:


#part2 check p(G=1|W =1) and p(G = 1)


# In[ ]:


ordered_hidden_var_list = ["A","B","E"]
evidence_vars =[["W",1.0]]
query_variables=["G"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


ordered_hidden_var_list = ["A","B","E"]
evidence_vars =[["W",0]]
query_variables=["G"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


ordered_hidden_var_list = ["B","E"]
evidence_vars =[["W",1],["G",0]]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# # p(G=1|W =1) =  0.408039

# In[ ]:


ordered_hidden_var_list = ["A","B","E","W"]
evidence_vars =[]
query_variables=["G"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(G=1|W =0)  = 0.401363


# In[ ]:


#p(G=1|W =1)  = 0.408039


# In[ ]:


#3A  P(A) ?=P(A|W)


# In[ ]:


ordered_hidden_var_list = ["B","E","G","W"]
evidence_vars =[]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(A=1) = 0.010151


# In[ ]:


ordered_hidden_var_list = ["B","E","G"]
evidence_vars =[["W",1]]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#P(A=1|W=1) = 0.020098


# In[ ]:


ordered_hidden_var_list = ["B","E","G"]
evidence_vars =[["W",0]]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#P(A=1|W=1) = 0.020098


# In[ ]:


#3B
#p(A = 1|W=1) ?= p(A = 1|W=1, G = 1) 

#P(A=1|W=1) = 0.020098
#P(A=1|G=1) = 0.020098
#p(A = 1|W=1, G = 1) =0.039404


# In[ ]:


ordered_hidden_var_list = ["B","E"]
evidence_vars =[["G",1],["W",1]]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(A = 1|W=1, G = 1) =0.039404


# In[ ]:


#3C

#P(A=1|G=1) ? = P(A=1|W=1,G=1)

#P(A=1|G=1) = 0.020098
#p(A=1|W=1, G=1) =0.039404


# In[ ]:


ordered_hidden_var_list = ["B","E","W"]
evidence_vars =[["G",1]]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(A=1|W=0, G=0) 
ordered_hidden_var_list = ["B","E"]
evidence_vars =[["G",0],["W",0]]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(A=1|W=0, G=0) = 0.001138


# In[ ]:


#p(A=1|W=0, G=0) 
ordered_hidden_var_list = ["B","E"]
evidence_vars =[["G",1],["W",0]]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(A=1|W=0, G=1) = 0.00679


# In[ ]:


#p(A=1|W=0, G=0) 
ordered_hidden_var_list = ["B","E"]
evidence_vars =[["G",0],["W",0]]
query_variables=["A"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(A=1|W=0, G=0)=0.00679


# In[ ]:


#4A
#P(B = 1) ?= p(B=1|W=1) 
#P(B)
ordered_hidden_var_list = ["A","E","G","W"]
evidence_vars =[]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#4A
#P(B) = 0.0001
#P(B = 1) ?= p(B=1|W=1)  
# = 0.000193
#NO
ordered_hidden_var_list = ["A","E","G"]
evidence_vars =[["W",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#4A
#P(B = 1) ?= p(B=1|G=1) 
#p(B=1|G=1) = 0.000193
#P(B) = 0.0001
#NO
ordered_hidden_var_list = ["A","E","W"]
evidence_vars =[["G",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#4A
#P(B = 1) ?= p(B=1|A=1) 
#p(B=1|A=1) = 0.009359
#P(B) = 0.0001
#NO
ordered_hidden_var_list = ["E","G","W"]
evidence_vars =[["A",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:





# In[ ]:


#4B 
#check
#P(B=1|W=1) ?= P(B=1|G=1,W=1)
#P(B=1|W=1) ?= P(B=1|A=1,W=1)
#P(B=1|W=1)  =  0.000193
ordered_hidden_var_list = ["A","E","G"]
evidence_vars =[["W",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#4B part1
#check
#P(B=1|W=1) ?= P(B=1|G=1,W=1)
#P(B=1|W=1)  =  0.000193
# P(B=1|G=1,W=1) = 0.000374
ordered_hidden_var_list = ["A","E"]
evidence_vars =[["W",1],["G",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#4B  -part2
#check
#P(B=1|W=1) ?= P(B=1|A=1,W=1)
#P(B=1|W=1)  =  0.000193
#P(B=1|A=1,W=1) = 0.009359
ordered_hidden_var_list = ["E","G"]
evidence_vars =[["A",1],["W",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#4c
#check
#P(B=1|G=1) ?= P(B=1|G=1,W=1)
#P(B=1|G=1) ?= P(B=1|G=1,A=1)
#P(B=1|G=1)  =  0.000193
ordered_hidden_var_list = ["A","E","W"]
evidence_vars =[["G",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#4c part1
#check
#P(B=1|G=1) ?= P(B=1|G=1,W=1)
#P(B=1|G=1)  =  0.000193
# P(B=1|G=1,W=1) = 0.000374
ordered_hidden_var_list = ["A","E"]
evidence_vars =[["W",1],["G",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#4c  -part2
#check
#P(B=1|G=1) ?= P(B=1|A=1,G=1)
#P(B=1|G=1) = 0.000193
#P(B=1|G=1,A=1) = 0.009359
ordered_hidden_var_list = ["E","W"]
evidence_vars =[["A",1],["G",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#4d
#check
#P(B=1|A=1) ?= P(B=1|A=1,W=1)
#P(B=1|A=1) ?= P(B=1|A=1,G=1)
#P(B=1|A=1)  = 0.009359
ordered_hidden_var_list = ["E","G","W"]
evidence_vars =[["A",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#P(B=1|A=0)  = 0.000005
ordered_hidden_var_list = ["E","G","W"]
evidence_vars =[["A",0]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#4d part1
#check
#P(B=1|A=1) ?= P(B=1|A=1,W=1)
#P(B=1|A=1)  = 0.009359
#P(B=1|A=1,W=1) = 0.009359
ordered_hidden_var_list = ["E","G"]
evidence_vars =[["A",1],["W",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#4d  -part2
#check
#P(B=1|A=1) ?= P(B=1|A=1,G=1)
#P(B=1|A=1)  = 0.009359
#P(B=1|A=1,G=1) = 0.009359
ordered_hidden_var_list = ["E","W"]
evidence_vars =[["A",1],["G",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#checked
#P(B=1|A=1) = P(B=1|A=1,W=1)
#also check 
#P(B=1|A=1) ?= P(B=1|A=1,W=0)
#P(B=1|A=0) ?= P(B=1|A=0,W=0)
#P(B=1|A=0) ?= P(B=1|A=0,W=1)

#checked
#P(B=1|A=1) = P(B=1|A=1,G=1)
#also check
#P(B=1|A=1) = P(B=1|A=1,G=0)
#P(B=1|A=0) = P(B=1|A=0,G=1)
#P(B=1|A=0) = P(B=1|A=0,G=0)


# In[ ]:


#P(B=1|A=1) ?= P(B=1|A=1,W=0)
#P(B=1|A=1)  = 0.009359
#P(B=1|A=1,W=0) = 0.009359
ordered_hidden_var_list = ["E","G"]
evidence_vars =[["A",1],["W",0]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#P(B=1|A=0)  = 0.000005
#P(B=1|A=0) ?= P(B=1|A=0,W=0)
#P(B=1|A=0,W=0) = 0.000005
ordered_hidden_var_list = ["E","G"]
evidence_vars =[["A",0],["W",0]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#P(B=1|A=0)  = 0.000005
#P(B=1|A=0) ?= P(B=1|A=0,W=1)
#P(B=1|A=0,W=1) = 0.000005
ordered_hidden_var_list = ["E","G"]
evidence_vars =[["A",0],["W",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#also check
#P(B=1|A=1) = P(B=1|A=1,G=0)
#P(B=1|A=1)  = 0.009359
#P(B=1|A=1,G=0) = 0.009359
ordered_hidden_var_list = ["E","W"]
evidence_vars =[["A",1],["G",0]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#P(B=1|A=0) = P(B=1|A=0,G=1)
# P(B=1|A=0) = 0.000005
#P(B=1|A=0,G=1) = 0.000005
ordered_hidden_var_list = ["E","W"]
evidence_vars =[["A",0],["G",1]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#P(B=1|A=0) = P(B=1|A=0,G=0)
# P(B=1|A=0) = 0.000005
#P(B=1|A=0,G=0) = 0.000005
ordered_hidden_var_list = ["E","W"]
evidence_vars =[["A",0],["G",0]]
query_variables=["B"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5A 
#P(E=1|A=1) ? = P(E=1|A=1,W=1)
#P(E=1|A=1) ? = P(E=1|A=1,G=1)
#P(E=1|A=1) ? = P(E=1|A=1,WE=1)


# In[ ]:


#5A 
#P(E=1|A=1)  = 0.005913
ordered_hidden_var_list = ["B","G","W"]
evidence_vars =[["A",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#5A 
#P(E=1|A=1)  = 0.005913
#P(E=1|A=1,W=1) = 0.005913
#then also need to check 
#P(E=1|A=1) ? = P(E=1|A=1,W=0)
#P(E=1|A=0) ? = P(E=1|A=0,W=1)
#P(E=1|A=0) ? = P(E=1|A=0,W=0)
ordered_hidden_var_list = ["B","G"]
evidence_vars =[["A",1],["W",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5A 
#P(E=1|A=1)  = 0.005913
#then also need to check 
#P(E=1|A=1) = P(E=1|A=1,W=0) = 0.005913
ordered_hidden_var_list = ["B","G"]
evidence_vars =[["A",1],["W",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5A 
#P(E=1|A=0) = 0.000242
ordered_hidden_var_list = ["B","G","W"]
evidence_vars =[["A",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#P(G|W) 
ordered_hidden_var_list = ["A","B","E"]
evidence_vars =[["W",0]]
query_variables=["G"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:





# In[ ]:


#5A 
#P(E=1|A=0) = 0.000242
#then also need to check 
#P(E=1|A=0)  = P(E=1|A=0,W=1) = 0.000242
ordered_hidden_var_list = ["B","G"]
evidence_vars =[["A",0],["W",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5A 
#P(E=1|A=0)  = 0.000242
#P(E=1|A=0) = P(E=1|A=0,W=0) not equal 
ordered_hidden_var_list = ["B","G"]
evidence_vars =[["A",0],["W",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5A- part2
#P(E=1|A=1) = 0.005913
#P(E=1|A=1) = P(E=1|A=1,G=1)  = 0.005913
#also need to check 
#P(E=1|A=1) = P(E=1|A=1,G=0)
#P(E=1|A=0) = P(E=1|A=0,G=1)
#P(E=1|A=0) = P(E=1|A=0,G=0)
ordered_hidden_var_list = ["B","W"]
evidence_vars =[["A",1],["G",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#5A- part2
#P(E=1|A=1) = 0.005913
#also need to check 
#P(E=1|A=1) = P(E=1|A=1,G=0) = 0.005913
ordered_hidden_var_list = ["B","W"]
evidence_vars =[["A",1],["G",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:





# In[ ]:


#5A- part2
#P(E=1|A=0)  = 0.000242
#also need to check 
#P(E=1|A=0) = P(E=1|A=0,G=1) = 0.000242
ordered_hidden_var_list = ["B","W"]
evidence_vars =[["A",0],["G",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#5A- part2
#P(E=1|A=0)  = 0.000242
#also need to check 
#P(E=1|A=0) = P(E=1|A=0,G=0) = 0.000242
ordered_hidden_var_list = ["B","W"]
evidence_vars =[["A",0],["G",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5A- part3
#P(E=1|A=1) = 0.005913
#also need to check 
#P(E=1|A=1) = P(E=1|A=1,B=1) = 0.000303
ordered_hidden_var_list = ["G","W"]
evidence_vars =[["A",1],["B",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#5B part1
#check p(E = 1|A = 1,B = 1) = p(E = 1|A = 1,B = 1, G = 1)
#p(E = 1|A = 1,B = 1) = 0.000303

ordered_hidden_var_list = ["G","W"]
evidence_vars =[["A",1],["B",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5B part1
#check p(E = 1|A = 1,B = 1) = p(E = 1|A = 1,B = 1, G = 1)
#p(E = 1|A = 1,B = 1) = 0.000303
# p(E = 1|A = 1,B = 1, G = 1) = 0.000303
ordered_hidden_var_list = ["W"]
evidence_vars =[["A",1],["B",1],["G",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#5B part1
#check p(E = 1|A = 1,B = 1) = p(E = 1|A = 1,B = 1, G = 1)
#p(E = 1|A = 1,B = 1) = 0.000303
#also check 
#p(E = 1|A = 1,B = 1) = p(E = 1|A = 1,B = 1, G = 0)

#p(E = 1|A = 1,B = 0) = p(E = 1|A = 1,B = 0, G = 0)
#p(E = 1|A = 1,B = 0) = p(E = 1|A = 1,B = 0, G = 1)

#p(E = 1|A = 0,B = 1) = p(E = 1|A = 0,B = 1, G = 0)
#p(E = 1|A = 0,B = 1) = p(E = 1|A = 0,B = 1, G = 1)


#p(E = 1|A = 0,B = 0) = p(E = 1|A = 0,B = 0, G = 0)
#p(E = 1|A = 0,B = 0) = p(E = 1|A = 0,B = 0, G = 1)



# In[ ]:


#p(E = 1|A = 1,B = 1) = 0.000303
#also need to calculate 
#p(E = 1|A = 0,B = 0) = 0.000242
#p(E = 1|A = 1,B = 0) = 0.005966
#p(E = 1|A = 0,B = 1) = 0.00024


# In[ ]:


#ordered_hidden_var_list = ["G","W"]
#evidence_vars =[["A",0],["B",1]]
#query_variables=["E"]
#inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5B part1
#check p(E = 1|A = 1,B = 1) = p(E = 1|A = 1,B = 1, G = 1)
#p(E = 1|A = 1,B = 1) = 0.000303
#also check 
#p(E = 1|A = 1,B = 1) = p(E = 1|A = 1,B = 1, G = 0) = 0.000303
ordered_hidden_var_list = ["W"]
evidence_vars =[["A",1],["B",1],["G",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#p(E = 1|A = 1,B = 0) = p(E = 1|A = 1,B = 0, G = 0) = 0.005966
#p(E = 1|A = 1,B = 0) = 0.005966
#p(E = 1|A = 1,B = 0, G = 0) = 0.005966
ordered_hidden_var_list = ["W"]
evidence_vars =[["A",1],["B",0],["G",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(E = 1|A = 1,B = 0) = 0.005966
#p(E = 1|A = 1,B = 0) = p(E = 1|A = 1,B = 0, G = 1) = 0.005966
ordered_hidden_var_list = ["W"]
evidence_vars =[["A",1],["B",0],["G",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#p(E = 1|A = 0,B = 1) = p(E = 1|A = 0,B = 1, G = 0)
#p(E = 1|A = 0,B = 1) = 0.00024
#p(E = 1|A = 0,B = 1, G = 0) = 0.00024
ordered_hidden_var_list = ["W"]
evidence_vars =[["A",0],["B",1],["G",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#p(E = 1|A = 0,B = 1) = p(E = 1|A = 0,B = 1, G = 1)
#p(E = 1|A = 0,B = 1) = 0.00024
#p(E = 1|A = 0,B = 1, G = 1) = 0.00024
ordered_hidden_var_list = ["W"]
evidence_vars =[["A",0],["B",1],["G",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#p(E = 1|A = 0,B = 0) = p(E = 1|A = 0,B = 0, G = 0)
#p(E = 1|A = 0,B = 0)  = 0.000242
#p(E = 1|A = 0,B = 0, G = 0)
ordered_hidden_var_list = ["W"]
evidence_vars =[["A",0],["B",0],["G",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(E = 1|A = 0,B = 0) = p(E = 1|A = 0,B = 0, G = 1)
#p(E = 1|A = 0,B = 0)  = 0.000242
#p(E = 1|A = 0,B = 0, G = 1) = 0.000242
ordered_hidden_var_list = ["W"]
evidence_vars =[["A",0],["B",0],["G",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5B 
#check p(E=1|A=1,B=1) ?=  p(E=1|A=1,B=1,W=1) = 0.000303
#p(E = 1|A = 1,B = 1) = 0.000303
#p(E=1|A=1,B=1,W=1) = 0.000303
#also need check 
ordered_hidden_var_list = ["G"]
evidence_vars =[["A",1],["B",1],["W",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#5B 
#also need to check 
#p(E = 1|A = 1,B = 1) = 0.000303
#also need to calculate 
#p(E = 1|A = 0,B = 0) = 0.000242
#p(E = 1|A = 1,B = 0) = 0.005966
#p(E = 1|A = 0,B = 1) = 0.00024


# In[ ]:


#5B 
#check p(E=1|A=1,B=1) ?=  p(E=1|A=1,B=1,W=1) = 0.000303
#p(E = 1|A = 1,B = 1) = 0.000303
#p(E=1|A=1,B=1,W=1) = 0.000303
#also need check 
#p(E=1|A=1,B=1) ?=  p(E=1|A=1,B=1,W=0) 


#p(E=1|A=1,B=0) ?=  p(E=1|A=1,B=0,W=1) 
#p(E=1|A=1,B=0) ?=  p(E=1|A=1,B=0,W=0) 

#p(E=1|A=0,B=1) ?=  p(E=1|A=0,B=1,W=1) 
#p(E=1|A=0,B=1) ?=  p(E=1|A=0,B=1,W=0) 


#p(E=1|A=0,B=0) ?=  p(E=1|A=0,B=0,W=1) 
#p(E=1|A=0,B=0) ?=  p(E=1|A=0,B=0,W=0) 


# In[ ]:


#5B 
#also need to check 
#p(E = 1|A = 1,B = 1) = 0.000303
#also need check 
#p(E=1|A=1,B=1) ?=  p(E=1|A=1,B=1,W=0)  = 0.000303
ordered_hidden_var_list = ["G"]
evidence_vars =[["A",1],["B",1],["W",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(E=1|A=1,B=0) ?=  p(E=1|A=1,B=0,W=1)  = 0.005966
#p(E = 1|A = 1,B = 0) = 0.005966
ordered_hidden_var_list = ["G"]
evidence_vars =[["A",1],["B",0],["W",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:


#p(E=1|A=1,B=0) ?=  p(E=1|A=1,B=0,W=0)  = 0.005966
#p(E = 1|A = 1,B = 0) = 0.005966
ordered_hidden_var_list = ["G"]
evidence_vars =[["A",1],["B",0],["W",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#p(E=1|A=0,B=1) ?=  p(E=1|A=0,B=1,W=1)  = 0.00024
#p(E = 1|A = 0,B = 1) = 0.00024
ordered_hidden_var_list = ["G"]
evidence_vars =[["A",0],["B",1],["W",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#p(E=1|A=0,B=1) ?=  p(E=1|A=0,B=1,W=0)  = 0.00024
#p(E = 1|A = 0,B = 1) = 0.00024
ordered_hidden_var_list = ["G"]
evidence_vars =[["A",0],["B",1],["W",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#p(E=1|A=0,B=0) ?=  p(E=1|A=0,B=0,W=1) = 0.000242
#p(E = 1|A = 0,B = 0) = 0.000242
ordered_hidden_var_list = ["G"]
evidence_vars =[["A",0],["B",0],["W",1]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#p(E=1|A=0,B=0) ?=  p(E=1|A=0,B=0,W=0) = 0.000242
#p(E = 1|A = 0,B = 0) = 0.000242
ordered_hidden_var_list = ["G"]
evidence_vars =[["A",0],["B",0],["W",0]]
query_variables=["E"]
inference(factor_list, query_variables, ordered_hidden_var_list, evidence_vars)


# In[ ]:





# In[ ]:


#also check
#P(B=1|A=0) = 0.000005 (next cell)
# run in this order
#P(B=1|A=0)?=p(B=1|A=0,W=1,G=1) = 0.000005
#P(B=1|A=0)?=p(B=1|A=0,W=0,G=1) = 0.000005
#P(B=1|A=0)?=p(B=1|A=0,W=0,G=0) = 0.000005
#P(B=1|A=0)?=p(B=1|A=0,W=1,G=1) = 0.000005

#and 
#P(B=1|A=1)  = 0.009359 (from pervious cell)
#P(B=1|A=1)?=p(B=1|A=1,W=1,G=1) = 0.009359
#P(B=1|A=1)?=p(B=1|A=1,W=0,G=1) = 0.009359
#P(B=1|A=1)?=p(B=1|A=1,W=0,G=0) = 0.009359 
#P(B=1|A=1)?=p(B=1|A=1,W=1,G=1) = 0.009359

