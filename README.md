# EXNO:4-DS
```
Name:Kavi Keerthana R
Reg No:212222100022
```
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
```
```
df=pd.read_csv("/content/bmi (1).csv")
df1=pd.read_csv("/content/bmi (1).csv")
df2=pd.read_csv("/content/bmi (1).csv")
df3=pd.read_csv("/content/bmi (1).csv")
df4=pd.read_csv("/content/bmi (1).csv")
```
```
df.head()
```
![Screenshot 2024-04-16 160612](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/b679364c-c039-40bf-8969-2ce47e4cee22)
```
df.dropna()
```
![Screenshot 2024-04-16 160729](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/21e2db4c-16f2-408f-9436-083db5f334ba)
```
max_vals = np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![Screenshot 2024-04-16 160814](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/0be461e2-d30f-4344-b946-ba1890a06fd8)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![Screenshot 2024-04-16 160849](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/7237e9e5-ae88-494a-a907-5ff2da9316b2)
```
from sklearn.preprocessing import MinMaxScaler
```
```
scaler = MinMaxScaler()
df[['Height','Weihgt']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-04-16 160943](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/ff65fe24-96bd-408b-83a9-6b193b352f86)
```
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
![Screenshot 2024-04-16 161036](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/7eaea797-2848-402b-a1f9-e5b192a0bbcc)
```
from sklearn.preprocessing import MaxAbsScaler
scaler =  MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![Screenshot 2024-04-16 201013](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/129987b2-931b-49e9-b264-3c70ea9733b5)
```
from sklearn.preprocessing import RobustScaler
scaler =  RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![Screenshot 2024-04-16 201139](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/ccd25eea-502a-4129-9696-3167f5ffd99b)
```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
data=pd.read_csv('/content/income(1) (1).csv',na_values=[' ?'])
data
```
![Screenshot 2024-04-16 201253](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/b3aedff6-9a3a-4b06-be82-5d636c099003)
```
data.isnull().sum()
```
![Screenshot 2024-04-16 201330](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/80ef2909-b382-43d0-b707-310ed6279f23)
```
missing = data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-04-16 201418](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/c970672c-0be4-4998-bc52-80277975d1aa)
```
data2 = data.dropna(axis=0)
data2
```
![Screenshot 2024-04-16 201450](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/66ebe610-256a-45d8-9a5b-a736f7b13917)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-04-16 201545](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/2c04a96a-4693-4370-ba3c-515c8ea3e94d)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-04-16 201622](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/a46fc0d9-cc3f-4ec8-a4a9-8fd77ba8991b)
```
data2
```
![Screenshot 2024-04-16 201649](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/c8444a71-6129-4bc0-8deb-c9cc61edc15b)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![Screenshot 2024-04-16 201731](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/9a843098-fb65-46f2-8244-aa83d5881af1)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-04-16 201801](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/3dafffd5-d2b0-42cc-b69c-35cd20e4ce04)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-04-16 201830](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/5354b558-8f78-4fac-9918-d5a62183eb73)
```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-04-16 201856](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/b5403ca0-ccbb-4d05-863c-7060548526c7)
```
x=new_data[features].values
print(x)
```
![Screenshot 2024-04-16 201924](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/de691ffc-a1b2-42e4-b5a7-dc23f0fdfb50)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
```
```
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]

}
```
```
df=pd.DataFrame(data)
df
```
![Screenshot 2024-04-16 202141](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/64228545-2aa1-401f-a857-cd6f4a6f3303)
```
X=df[['Feature1','Feature3']]
y=df[['Target']]
```
```
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
```
![Screenshot 2024-04-16 202218](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/90910896-bab7-46e1-adb5-65d5e65fe0c6)
```
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-04-16 202255](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/6f74dd65-29d1-4c44-a8dd-a10a742ad9a4)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
```
```
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-04-16 202347](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/1696d545-34ff-4eb0-8c09-78f592c259b4)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/422ee4a0-e291-426a-ad36-069994157303)
```
chi2, p, _, _=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![Screenshot 2024-04-16 203545](https://github.com/Anusharonselva/EXNO-4-DS/assets/119405600/3cea47d6-ef13-436e-901a-4d7130be86d2)


# RESULT:
      
 Thus feature scaling and selection is performed.
