import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
#from tpot import TPOTClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
import random
import warnings
from statistics import mean
warnings.filterwarnings('ignore')
df=pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
#df_x=pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
#random.seed(23)
#Preprocessing the Synthetic data
#Rounding of synthetic data points
#df['Age']=df['Age'].round(0)
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
df3=df.copy()
from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
df3[['Age','Height','Weight']]=scaler2.fit_transform(df3[['Age','Height','Weight']])
target=df3['OBESITY']
df3.pop('OBESITY')
df3.pop('CALC_Always')  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df3, target,
                                                    train_size=0.75, test_size=0.25,shuffle=True)         
#tpot = TPOTClassifier(verbosity=2, n_jobs=-2,warm_start=True,periodic_checkpoint_folder=r'C:\Users\Akash\Desktop\Capstone Project\pipe')
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#tpot.export('tpot_pipeline.py')
exported_pipeline = make_pipeline(
    SelectFromModel(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.1, n_estimators=100), threshold=0.05),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.4, n_estimators=100), step=0.6000000000000001),
    MaxAbsScaler(),
    StackingEstimator(estimator=MLPClassifier(alpha=0.1, learning_rate_init=0.5)),
    RobustScaler(),
    XGBClassifier(learning_rate=0.5, max_depth=7, min_child_weight=1, n_estimators=100, n_jobs=1, subsample=0.8500000000000001, verbosity=0)
)
exported_pipeline.fit(X_train,y_train)
results = exported_pipeline.predict(X_test)

#Comparing results
from sklearn.metrics import recall_score
print(recall_score(y_test, results, average= 'weighted'))
from sklearn.metrics import precision_score
print(precision_score(y_test, results, average= 'weighted'))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, results))
from sklearn.metrics import f1_score
print(f1_score(y_test, results, average= 'weighted'))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,results))
cfm=confusion_matrix(y_test,results)
avacc=(np.sum(cfm[i,i] for i in range(6)))/(np.sum(cfm))
print('Average Accuracy=')
print(avacc)
from sklearn.metrics import classification_report
print(classification_report(y_test, results,digits=6, target_names=['Class 1', 'Class 2', 'Class 3','Class 4','Class 5','Class 6']))
def indimetric(cfm):
    import numpy as np
    inprec=[]
    inacc=[]
    inrec=[]
    tp=[]
    fp=[]
    tn=[]
    fn=[]
    for i in range(len(cfm)):
        tp.append(cfm[i,i])
        tn.append(np.sum(cfm[(i+1):,(i+1):])+np.sum(cfm[0:i,0:i])+np.sum(cfm[0:i,(i+1):]))
        fp.append(np.sum(cfm[0:i,i])+np.sum(cfm[(i+1):,i]))
        fn.append(np.sum(cfm[i,0:i])+np.sum(cfm[i,(i+1):]))
        inprec.append((tp[i])/(tp[i]+fp[i]))
        inacc.append(((tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i]))*100)
        inrec.append((tp[i])/(tp[i]+fn[i]))
    #print (tp,tn,fp,fn)
    #print(inprec)
    return tp,tn,fp,fn,inacc,inprec,inrec
tp,tn,fp,fn,accuracy,precision,recall=indimetric(cfm)
def plotgraph(data):
    import plotly.graph_objects as go
    from plotly.offline import plot
    x=['Underweight','Normal','Overweight','Obesity I','Obesity II','Obesity III']
    fig = go.Figure([go.Bar(x=x, y=data)])
    fig.show()
    plot(fig)
#plotgraph(accuracy)
#plotgraph(precision)
#plotgraph(recall)
print(sum(precision)/6)
print(sum(recall)/6)