
# coding: utf-8

# In[2]:


import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Imputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


# In[3]:


os.listdir()


# In[4]:


with open('HomeCredit_columns_description.csv', 'r') as file:
    content = file.read()
    print (content)
del content


# In[3]:


dfTrain = pd.read_csv('application_train.csv')


# In[4]:


dfTest = pd.read_csv('application_test.csv')


# In[7]:


print('Training data shape: ', dfTrain.shape)
dfTrain.head()


# In[8]:


print('Test data shape: ', dfTest.shape)
dfTest.head()


# In[9]:


qdTrain = dfTrain.shape[0]
qdTest = dfTest.shape[0]
qdTotal = qdTrain + qdTest
qTrPer = (qdTrain*100)/qdTotal
qTePer = (qdTest*100)/qdTotal


# In[10]:


print('Total data: ', qdTotal)
print('Percentage for training: {}'.format(qTrPer))
print('Percentage for testing: {}'.format(qTePer))
print('Number of features training: ', dfTrain.shape[1])
print('Number of features test: ', dfTest.shape[1])


# In[11]:


dfTrain['TARGET'].value_counts()


# In[12]:


dfTrain['TARGET'].astype(int).plot.hist()


# In[13]:


missingVal = dfTrain.isnull().sum()
MVpercent = 100 * dfTrain.isnull().sum() / len(dfTrain)
dfMissingValues = pd.concat([missingVal, MVpercent], axis=1)
dfMissingValues = dfMissingValues.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
dfMissingValues = dfMissingValues[dfMissingValues.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(3)


# In[14]:


dfMissingValues.head(25)


# In[15]:


dfMissingValues['% of Total Values'].astype(int).plot.hist()


# In[16]:


x = dfTrain.dtypes.value_counts()
idX2 = 0
ls = ['float64','int64', 'object']
for idx in range(len(x)):
    per = (x[idx]*100)/x.sum()
    print('{} has a percentage of {:0.2f}'.format(ls[idX2], per))
    idX2 += 1


# #### Cuales son las columnas tipo object

# In[17]:


dfTrain.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# Para evitar aumento innecesario de la dimensionalidad, para las 'object' que solo son dos unicas cosas usaremos label encoder (0,1) y OneHot para las que son mayores a 2 para evitar darle mas peso a una forma de la varible que a otra usando label encoder

# In[5]:


labelEnc = LabelEncoder()
for column in dfTrain:
    if dfTrain[column].dtype == 'object':
        if len(list(dfTrain[column].unique())) <= 2:
            labelEnc.fit(dfTrain[column])
            dfTrain[column] = labelEnc.transform(dfTrain[column])
            dfTest[column] = labelEnc.transform(dfTest[column])


# In[6]:


dfTrain = pd.get_dummies(dfTrain)
dfTest = pd.get_dummies(dfTest)


# In[20]:


print('Number of features training: ', dfTrain.shape[1])
print('Number of features test: ', dfTest.shape[1])


# Hacer One Hot ha creado mas columnas y ha desalineado los dos df, no pueden estar así, entonces toca alinearlos

# In[7]:


targetLabel = dfTrain['TARGET'] #Guardamos el label para mas adelante


# In[8]:


dfTrain, dfTest = dfTrain.align(dfTest, join = 'inner', axis = 1) #Los unimos por el axis=1 por las columnas


# In[9]:


dfTrain['TARGET'] = targetLabel #Volvemos a añadir la columna del target


# In[24]:


print('Number of features training: ', dfTrain.shape[1])
print('Number of features test: ', dfTest.shape[1])


# Los datos poseen columnas de información sobre días, y puede ser que estos días tengan fechas curiosas por error

# In[25]:


#Las columnas seran: DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION
(dfTrain['DAYS_BIRTH'] / -365).describe()


# Como se puede ver, los datos (pasados desde negativos) tienen una descripcion normal, con una minina edad de 20 años y una maxima de 69.
# Con lo que está en unos rangos normales
# 
# Siguiente columna

# In[26]:


(dfTrain['DAYS_EMPLOYED']).describe()


# Y ahora como que algo no cuadra, todos los datos van en un buen rango excepto el maximo, porque no tiene ningun sentido, si los días se cuentan en negativo hacia atras porque esta positivo, y ademas ese numero es gigante, veamos cuanto es en años (365243)

# In[27]:


365243/365


# Mas de mil años, algo no funciona con ese valor, entonces veamos que tal se ven en grafica

# In[28]:


dfTrain['DAYS_EMPLOYED'].plot.hist();
plt.xlabel('Days Employment');


# In[29]:


raros = dfTrain[dfTrain['DAYS_EMPLOYED'] > 0]
print('Hay %d dias que no cuadran' % len(raros))


# No son poquitos los que justamente andan entre el 75% y el 100%, algo hay que hacer

# In[10]:


dfTrain['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)


# Veamos en test que tal

# In[31]:


dfTest['DAYS_EMPLOYED'].plot.hist();
plt.xlabel('Days Employment');


# O mira de nuevo los raros, vamos a hacerles lo mismo

# In[11]:


dfTest['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)


# In[33]:


dfTrain['DAYS_EMPLOYED'].plot.hist();
plt.xlabel('Days Employment');


# Mira que bien que se ve ahora

# In[34]:


(dfTrain['DAYS_REGISTRATION']).describe()


# In[35]:


dfTrain['DAYS_REGISTRATION'].plot.hist();
plt.xlabel('Days Registry');


# In[36]:


(dfTrain['OWN_CAR_AGE']).describe()


# In[37]:


dfTrain['OWN_CAR_AGE'].head(25)


# In[38]:


dfTrain['OWN_CAR_AGE'].plot.hist();
plt.xlabel('Age of client car');


# Ahora que nos encargamos de las columnas tipo 'object' y de las cosas raras miremos correlaciones y luego creamos algunas variables pensando en el problema

# In[39]:


correlations = dfTrain.corr()['TARGET'].sort_values()


# En correlations mostramos todas las correlaciones, pero vamos a ver solamente las 10 mas positivas y negativas

# In[40]:


print('Correlaciones mas positivas:\n', correlations.tail(11))
print('\nCorrelaciones mas negativas:\n', correlations.head(10))


# Para efectos de Feature Engineering se van a añadir las siguientes variables:
# 
# CREDITINC_PERCENT: cuanto porcentaje representa el credito en los ingresos del cliente
# 
# ANNINC_PERCENT: anualmente cuando porcentaje es el credito de los ingresos del cliente
# 
# EMPLOYMENTDAYS_PERCENT: por cuanto tiempo en la vida total del cliente a trabajado
# 
# MORE_2_CHILDREN: si tiene mas de dos hijos, por el tema de costos elevados
# 
# MEAN_EXT_SOURCE: promedio de las tres referencias externas
# 
# DISTANCE_EXTS_COM: distancia al promedio de MEAN_EXT_SOURCE que tienen los valores positivos en dfTtrain

# Añadir variables a train

# In[12]:


dfTrain['CREDITINC_PERCENT'] = dfTrain['AMT_CREDIT'] / dfTrain['AMT_INCOME_TOTAL']
dfTrain['ANNINC_PERCENT'] = dfTrain['AMT_ANNUITY'] / dfTrain['AMT_INCOME_TOTAL']
dfTrain['EMPLOYMENTDAYS_PERCENT'] = dfTrain['DAYS_EMPLOYED'] / dfTrain['DAYS_BIRTH']
dfTrain['MORE_2_CHILDREN'] = dfTrain['CNT_CHILDREN'].map(lambda x: 1 if x > 1 else 0)
dfTrain['SUM_EXT_SOURCE'] = dfTrain['EXT_SOURCE_1']+dfTrain['EXT_SOURCE_2']+dfTrain['EXT_SOURCE_3']
dfTrain['MEAN_EXT_SOURCES'] = dfTrain['SUM_EXT_SOURCE']/3


# In[13]:


groupsTarget = dfTrain['MEAN_EXT_SOURCES'].groupby(dfTrain['TARGET'])
groupsTarget.mean()


# In[14]:


disMeanExt = groupsTarget.mean()[0]
disMeanExt


# In[15]:


dfTrain['DISTANCE_EXTS_COM'] = disMeanExt - dfTrain['MEAN_EXT_SOURCES']


# In[70]:


dfTrain.head()


# Añadir variables a test

# In[16]:


dfTest['CREDITINC_PERCENT'] = dfTest['AMT_CREDIT'] / dfTest['AMT_INCOME_TOTAL']
dfTest['ANNINC_PERCENT'] = dfTest['AMT_ANNUITY'] / dfTest['AMT_INCOME_TOTAL']
dfTest['EMPLOYMENTDAYS_PERCENT'] = dfTest['DAYS_EMPLOYED'] / dfTest['DAYS_BIRTH']
dfTest['MORE_2_CHILDREN'] = dfTest['CNT_CHILDREN'].map(lambda x: 1 if x > 1 else 0)
dfTest['SUM_EXT_SOURCE'] = dfTest['EXT_SOURCE_1']+dfTest['EXT_SOURCE_2']+dfTest['EXT_SOURCE_3']
dfTest['MEAN_EXT_SOURCES'] = dfTest['SUM_EXT_SOURCE']/3
dfTest['DISTANCE_EXTS_COM'] = disMeanExt - dfTest['MEAN_EXT_SOURCES']


# In[72]:


dfTest.head()


# ### Modelos

# In[17]:


# Preparamos dfTrain:
dfTrain = dfTrain.drop('TARGET', axis=1) #Recordad Target as targetLabel

nanReplacer = Imputer(strategy='median')

nanReplacer.fit(dfTrain)

train1 = nanReplacer.transform(dfTrain)
testF = nanReplacer.transform(dfTest)

XTrain, XVal, ytrain, yval = train_test_split(train1, targetLabel, test_size=0.3)


# #### Decision Tree

# In[104]:


dt = Pipeline(
    [
        #('scaler', MinMaxScaler()),
        ('reg', DecisionTreeClassifier(criterion='entropy'))
    ]
)
dt.fit(XTrain, ytrain)


# In[105]:


train_score = f1_score(ytrain, dt.predict(XTrain))
train_score


# In[106]:


val_score = f1_score(yval, dt.predict(XVal))
val_score


# In[107]:


train_score_acc = accuracy_score(ytrain, dt.predict(XTrain))
val_score_acc = accuracy_score(yval, dt.predict(XVal))
print(train_score_acc)
print(val_score_acc)


# #### Random Forest

# In[109]:


rf = Pipeline(
    [
        #('scaler', MinMaxScaler()),
        ('reg', RandomForestClassifier(criterion='entropy'))
    ]
)
rf.fit(XTrain, ytrain)


# In[110]:


train_score = f1_score(ytrain, rf.predict(XTrain))
train_score


# In[111]:


val_score = f1_score(yval, rf.predict(XVal))
val_score


# In[112]:


train_score_acc = accuracy_score(ytrain, rf.predict(XTrain))
val_score_acc = accuracy_score(yval, rf.predict(XVal))
print(train_score_acc)
print(val_score_acc)


# In[117]:


# Listas para guardar el score
train_scores = []
val_scores = []
param_range = [100, 200, 300, 400, 500]
for f in param_range:
    
    clf = RandomForestClassifier(n_estimators=f ,criterion='entropy')
    
    clf.fit(XTrain, ytrain)
    y_train_pred = clf.predict(XTrain)
    y_val_pred = clf.predict(XVal)
    
    train_score = accuracy_score(ytrain, y_train_pred)
    val_score = accuracy_score(yval, y_val_pred)
    
    train_scores.append(train_score)
    val_scores.append(val_score)
    print (f)
    

# plot the scores along the diffent values of the hiper parameter
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
ax.plot(
    param_range, train_scores, color='blue',
    marker='o', linestyle='-', markersize=5, 
    label='Training data'
)

ax.plot(
    param_range, val_scores, color='green',
    marker='s', linestyle='--', markersize=5, 
    label='Validation data'
)

ax.set_xscale('log')
ax.set_xlabel('Hiperparameter n_estimators')
ax.set_ylabel('Accuracy')
ax.set_ylim(min(min(train_scores, val_scores)) - 0.1, 1)
ax.grid(linestyle='--')
ax.legend(loc='lower right')


# #### SVC

# In[ ]:


sv = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('reg', LinearSVC())
    ]
)
sv.fit(XTrain, ytrain)


# In[ ]:


train_score = f1_score(ytrain, sv.predict(XTrain))
train_score


# In[ ]:


val_score = f1_score(yval, sv.predict(XVal))
val_score


# In[ ]:


train_score_acc = accuracy_score(ytrain, sv.predict(XTrain))
val_score_acc = accuracy_score(yval, sv.predict(XVal))
print(train_score_acc)
print(val_score_acc)

