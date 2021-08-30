import pandas as pd
import numpy as np


 
def LoadArabic (filename):
    
    file=pd.read_csv(filename,header=None,sep=" " )
  
    current_mts=[]

    data_size=0
    max_len=0
    num_of_attr=13

    data=[]
    labels=[] 
    last_mts_id=file.iloc[0,0]
   
    for i in range (len(file)):    
        mts_id=file.iloc[i,0]
   

        mts_num=file.iloc[i,1]
   
        max_len=max(mts_num,max_len)
       
      
        mts_nums=file.iloc[i,3:3+num_of_attr].values
        
        if len(mts_nums) < num_of_attr:
            print ('less attributes...i',i)
      

         

        if   mts_id != last_mts_id:
            if current_mts :
                data.append(current_mts)
                labels.append(mts_label)
                current_mts=[]
                data_size+=1
                last_mts_id=mts_id

        mts_label=file.iloc[i,2]
        current_mts.append(mts_nums)
         

    X=np.zeros((data_size,max_len,num_of_attr))
    Y=np.zeros((data_size,1))

    for i  in range(len(data)):

        X[i,0:len(data[i]),:]=data[i]
        Y[i]=labels[i]
    
    return X,Y

