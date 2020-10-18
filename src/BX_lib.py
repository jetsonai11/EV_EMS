### Import necessary python modules
import numpy as np
import warnings
import csv
warnings.simplefilter('ignore')

def indx_list_search_s2(s,ss,gap_s,reso_s,reso_a):    
    i_list=np.ones(len(s))    
    for i in range(len(s)):
        for j in range(len(ss[0,])):            
            if np.abs(s[i]-ss[i,j])<=gap_s[i]:
                i_list[i]=j                
                break
    indx = int(i_list[0]*reso_a + i_list[1]*reso_s*reso_a)
    indx_vec=np.zeros(reso_a)
    cnt=0
    for i in range(indx,indx+reso_a):        
        indx_vec[cnt]=i
        cnt=cnt+1
    indx_vec=np.array(indx_vec,int)    
    return indx_vec

def indx_list_search_s3(s,ss,gap_s,reso_s,reso_a):    
    i_list=np.ones(len(s))    
    for i in range(len(s)):
        for j in range(len(ss[0,])):            
            if np.abs(s[i]-ss[i,j])<=gap_s[i]:
                i_list[i]=j                
                break
    indx = int( i_list[0]*reso_a + i_list[1]*reso_s*reso_a + i_list[2]*reso_a*reso_s**2)
    indx_vec=np.zeros(reso_a)
    cnt=0
    for i in range(indx,indx+reso_a):        
        indx_vec[cnt]=i
        cnt=cnt+1
    indx_vec=np.array(indx_vec,int)    
    return indx_vec

def indx_search(a,aa,gap_a):
    i_a = len(aa)/2
    for j in range(len(aa)):
        if np.abs(a-aa[j])<=gap_a:
            i_a=j
            break
    indx = i_a
    return indx
