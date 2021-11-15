import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
df=pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
#Preprocessing the Synthetic data
#Rounding of synthetic data points
df['Age']=df['Age'].round(0)
df['FCVC']=df['FCVC'].round(0)
df['NCP']=df['NCP'].round(0)
df['CH2O']=df['CH2O'].round(0)
df['FAF']=df['FAF'].round(0)
df['TUE']=df['TUE'].round(0)
#Encoding Categorical columns
df = pd.get_dummies(df,prefix=['Gender'], columns = ['Gender'], drop_first=True)
df = pd.get_dummies(df,prefix=['family_history_with_overweight_'], columns = ['family_history_with_overweight'], drop_first=True)
df = pd.get_dummies(df,prefix=['FAVC'], columns = ['FAVC'], drop_first=True)
df = pd.get_dummies(df,prefix=['FCVC'], columns = ['FCVC'], drop_first=False)
df = pd.get_dummies(df,prefix=['NCP'], columns = ['NCP'], drop_first=False)
df = pd.get_dummies(df,prefix=['CAEC'], columns = ['CAEC'], drop_first=False)
df = pd.get_dummies(df,prefix=['SMOKE'], columns = ['SMOKE'], drop_first=True)
df = pd.get_dummies(df,prefix=['CH2O'], columns = ['CH2O'], drop_first=False)
df = pd.get_dummies(df,prefix=['SCC'], columns = ['SCC'], drop_first=True)
df = pd.get_dummies(df,prefix=['FAF'], columns = ['FAF'], drop_first=False)
df = pd.get_dummies(df,prefix=['CALC'], columns = ['CALC'], drop_first=False)
df = pd.get_dummies(df,prefix=['MTRANS'], columns = ['MTRANS'], drop_first=False)
#Target Column Wrong
del df['NObeyesdad']
df['BMI']=df['Weight']/(df['Height']*df['Height'])
df.loc[df['BMI']<18.50,'OBESITY']=1
df.loc[(df['BMI']>=18.50)&(df['BMI']<25),'OBESITY']=2
df.loc[(df['BMI']>=25)&(df['BMI']<30),'OBESITY']=3
df.loc[(df['BMI']>=30)&(df['BMI']<35),'OBESITY']=4
df.loc[(df['BMI']>=35)&(df['BMI']<40),'OBESITY']=5
df.loc[df['BMI']>=40,'OBESITY']=6
del df['BMI']
# Creating 1 Dataframe as MinMax Scaler
df1=df.copy()
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df1[['Age','Height','Weight']]=scaler.fit_transform(df1[['Age','Height','Weight']])
# Creating 1 dataframe with maximum absolute Scaler
from sklearn.preprocessing import MaxAbsScaler
df2=df.copy()
scaler1=MaxAbsScaler()
df2[['Age','Height','Weight']]=scaler1.fit_transform(df2[['Age','Height','Weight']])
# Transforming by z-score
df3=df.copy()
from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
df3[['Age','Height','Weight']]=scaler2.fit_transform(df3[['Age','Height','Weight']])
# Scaling with Inter Quartile Range
df4=df.copy()
from sklearn.preprocessing import RobustScaler
scaler3 = RobustScaler()
df4[['Age','Height','Weight']]=scaler3.fit_transform(df4[['Age','Height','Weight']])

#Taking any 1 data to run best model tuner
target=df1['OBESITY']
df1.pop('OBESITY')  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1, target,
                                                    train_size=0.80, test_size=0.20)         

exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_union(
            FunctionTransformer(copy),
            make_union(
                FunctionTransformer(copy),
                make_union(
                    FunctionTransformer(copy),
                    FunctionTransformer(copy)
                )
            )
        )
    ),
    SelectFwe(score_func=f_classif, alpha=0.049),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=1.0, min_samples_leaf=2, min_samples_split=5, n_estimators=100)
)
exported_pipeline.fit(X_train,y_train)
results = exported_pipeline.predict(X_test)

#Comparing results
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, results))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,results))
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test,results, pos_label=6)
print(metrics.auc(fpr, tpr))
from sklearn.metrics import precision_score
print(precision_score(y_test,results,average='macro'))
#from tpot import TPOTClassifier
#tpot = TPOTClassifier(verbosity=2,n_jobs=-3,warm_start=True,periodic_checkpoint_folder=r'C:\Users\Akash\Desktop\Capstone Project\pipes')
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#tpot.export('tpot_code_pipeline.py')
#print('a')