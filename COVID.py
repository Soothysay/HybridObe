def encode (df):
  # Changes a Given Pandas DataFrame df to match COVID-19 Scenario
  import pandas as pd
  import random
  wl=df['Weight'].to_list()
  for i in range(len(wl)):
     rand=random.uniform(0,4.455)
     wl[i]=wl[i]+rand
  df['Weight'] = wl
  return df