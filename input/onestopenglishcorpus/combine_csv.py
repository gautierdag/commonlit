import os
import glob
import numpy as np
import pandas as pd


path = "C:/Users/Maunish Dave/Desktop/doing boring stuff with python/Data Science Notepad/OneStopEnglishCorpus-master/Texts-Together-OneCSVperFile"
extension = 'csv'
os.chdir(path)
files = glob.glob('*.{}'.format(extension))

Elementary = list()
Intermediate = list()
Advanced = list()

for i,file in enumerate(files):
    l = pd.read_csv(f"{path}/{file}",encoding='cp1252')
    if l.shape[0] > 0:
        print(i)
        Elementary.append(l.iloc[:,0].to_list())
        Intermediate.append(l.iloc[:,1].to_list())
        Advanced.append(l.iloc[:,2].to_list())


df = pd.DataFrame({'Elementary':Elementary,'Intermediate':Intermediate,'Advanced':Advanced})

df.to_csv('all_data.csv',index=False)

print(df.shape)


