import pandas as pd
import numpy as np
import pdpbox.pdp
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from pdpbox.pdp import PDPIsolate
from sklearn.inspection import PartialDependenceDisplay, partial_dependence


#Data reading
credit_db_values = pd.read_csv("C:/Users/domin/OneDrive/Pulpit/IiE/UM/Projekt zaliczeniowy/Credit_card.csv")
credit_db_label = pd.read_csv("C:/Users/domin/OneDrive/Pulpit/IiE/UM/Projekt zaliczeniowy/Credit_card_label.csv")
#Merging datasets
credit_db = pd.merge(credit_db_values, credit_db_label, on='Ind_ID', how='inner')
print(f"Kształt pierwotnego zbioru danych: {credit_db.shape}")
#Data cleaning and encoding
credit_db = credit_db.drop(['Ind_ID'], axis=1)
credit_db.dropna(inplace=True)
credit_db['Birthday_count'] = credit_db['Birthday_count']*-1
credit_db['Birthday_count'] = round(credit_db['Birthday_count']/365,0)
credit_db['Employed_days'] = credit_db['Employed_days']*-1
credit_db = credit_db.rename(columns={'Birthday_count': 'Age','GENDER': 'Gender', 'CHILDREN': 'Children', 'EDUCATION': 'Education', 'EMAIL_ID': 'Email_ID', 'label': 'CreditCard_decision'})
credit_db.drop(['Mobile_phone'], axis=1, inplace=True)
credit_db.reset_index(drop=True, inplace=True)
credit_db['Gender'] = credit_db['Gender'].map({'M':1, 'F':0})
credit_db['Car_Owner'] = credit_db['Car_Owner'].map({'Y':1, 'N':0})
credit_db['Propert_Owner'] = credit_db['Propert_Owner'].map({'Y':1, 'N':0})
credit_db['Type_Income'] = credit_db['Type_Income'].replace('Commercial associate', 'Working')
credit_db_encoded1 = pd.get_dummies(credit_db, columns=['Type_Income'])
credit_db_encoded1['Marital_status'] = credit_db_encoded1['Marital_status'].replace({'Single / not married' : 'Single', 'Civil marriage' : 'Married'})
credit_db_encoded1['Marital_status'] = credit_db_encoded1['Marital_status'].replace({'Separated' : pd.NA, 'Widow' : pd.NA})
credit_db_encoded1.dropna(inplace=True)
credit_db_encoded1.reset_index(drop=True, inplace=True)
credit_db_encoded1['Marital_status'] = credit_db_encoded1['Marital_status'].map({'Married' : 1, 'Single': 0})
credit_db_encoded1['Education'] = credit_db_encoded1['Education'].replace('Incomplete higher', pd.NA)
credit_db_encoded1.dropna(inplace=True)
credit_db_encoded1.reset_index(drop=True, inplace=True)
credit_db_encoded2 = pd.get_dummies(credit_db_encoded1, columns=['Education'])
credit_db_encoded2.drop(['Phone', 'Email_ID'], axis=1, inplace=True)
housing_categories = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']
ordinal_encoder_housing = OrdinalEncoder(categories=[housing_categories])
credit_db_encoded2['Housing_type_encoded'] = ordinal_encoder_housing.fit_transform(credit_db_encoded2[['Housing_type']])
credit_db_encoded2.drop(['Housing_type'], axis=1, inplace=True)
occupation_categories = [
    'Managers', 'Core staff', 'High skill tech staff', 'Medicine staff', 'IT staff', 'Accountants', 
    'HR staff', 'Secretaries', 'Realty agents', 'Sales staff', 'Drivers', 
    'Security staff', 'Private service staff', 'Cooking staff', 'Waiters/barmen staff', 
    'Cleaning staff', 'Laborers', 'Low-skill Laborers'
]
ordinal_encoder_occupation = OrdinalEncoder(categories=[occupation_categories])
credit_db_encoded2['Type_Occupation_encoded'] = ordinal_encoder_occupation.fit_transform(credit_db_encoded2[['Type_Occupation']])
credit_db_encoded2.drop(['Type_Occupation'], axis= 1, inplace=True)
columns = list(credit_db_encoded2.columns)
columns.append(columns.pop(columns.index('CreditCard_decision')))
credit_db_encoded2 = credit_db_encoded2[columns]

#EDA - Exploratory Data Analysis - STEP 1
print(f"Kształt zbioru danych po obróbce: {credit_db_encoded2.shape}")
print(credit_db_encoded2.info())
#Outliers detection
print(credit_db_encoded2.describe())
numerical_columns = ['Children', 'Age', 'Family_Members']
sns.boxplot(data = credit_db_encoded2[numerical_columns])
plt.title('Data outliers', fontsize = 14)
plt.xlabel('Categories', fontsize = 11)
plt.ylabel('Distribiution', fontsize = 11)
plt.show()

numerical_columns_anin = ['Annual_income']
sns.boxplot(data = credit_db_encoded2[numerical_columns_anin])
plt.title('Data outliers', fontsize = 14)
plt.xlabel('Category', fontsize = 11)
plt.ylabel('Distribiution', fontsize = 11)
plt.show()

numerical_columns_empld = ['Employed_days']
sns.boxplot(data = credit_db_encoded2[numerical_columns_empld])
plt.title('Data outliers', fontsize = 14)
plt.xlabel('Category', fontsize = 11)
plt.ylabel('Distribiution', fontsize = 11)
plt.show()

#Outliers removing
def remove_outliers(df, columns):
  for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
  
  return df

numerical_columns_outliers = ['Annual_income', 'Employed_days']
credit_db_encoded_no_outliers = remove_outliers(credit_db_encoded2, numerical_columns_outliers)
print(credit_db_encoded_no_outliers.info())

#Data distribiution
credit_rejected = credit_db_encoded_no_outliers[credit_db_encoded_no_outliers['CreditCard_decision'] == 1]['CreditCard_decision'].sum()
credit_approved = credit_db_encoded_no_outliers.shape[0] - credit_rejected

bars = plt.bar(['Approved', 'Rejected'], [credit_approved, credit_rejected], color = ['green', 'red'])
for bar in bars:
  plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3 , str(bar.get_height()), ha = 'center')
plt.title('Data distribiution in data set', fontsize = 14)
plt.xlabel('Categories', fontsize = 11)
plt.ylabel('Number of data', fontsize = 11)  
plt.show()

#Correlation matrix
correlation_matrix = credit_db_encoded_no_outliers.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation matrix')
plt.show()

#Strongly correlated columns removing
credit_db_encoded_no_outliers.drop(['Family_Members'], axis= 1, inplace=True)

#Model building - STEP 2

#Spliting data
X = credit_db_encoded_no_outliers.drop('CreditCard_decision', axis=1)
Y = credit_db_encoded_no_outliers['CreditCard_decision']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=40)

print(f'Zbiór uczący (X_train): {X_train.shape}')


#Oversampling for train dataset
#SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=40)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Zbiór uczący przed SMOTE: {X_train.shape}\n{y_train.value_counts()}")
print(f"Zbiór uczący po SMOTE: {X_train_smote.shape}\n{y_train_smote.value_counts()}")

credit_rejected_final = credit_db_encoded_no_outliers[credit_db_encoded_no_outliers['CreditCard_decision'] == 1]['CreditCard_decision'].sum()
credit_approved_final = credit_db_encoded_no_outliers.shape[0] - credit_rejected_final

bars = plt.bar(['Approved', 'Rejected'], [credit_approved_final, credit_rejected_final], color = ['green', 'red'])
for bar in bars:
  plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3 , str(bar.get_height()), ha = 'center')
plt.title('Data distribiution before ADASYN', fontsize = 14)
plt.xlabel('Categories', fontsize = 11)
plt.ylabel('Number of data', fontsize = 11)  
plt.show()

#ADASYN
adasyn = ADASYN(sampling_strategy='auto', random_state=40)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

credit_rejected_smote = y_train_smote.value_counts().get(1,0)
credit_approved_smote = y_train_smote.value_counts().get(0,0)
bars = plt.bar(['Approved', 'Rejected'], [credit_approved_smote, credit_rejected_smote], color = ['green', 'red'])
for bar in bars:
  plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3 , str(bar.get_height()), ha = 'center')
plt.title('Data distribiution after SMOTE', fontsize = 14)
plt.xlabel('Categories', fontsize = 11)
plt.ylabel('Number of data', fontsize = 11)  
plt.show()

#Decision Tree with SMOTE oversampling method
max_depth_list = []
recall_score_list = []
recall_score_list_train = []
recall_score_list_train_cross_val = []
accuracy_score_list_dt = []
accuracy_score_list_dt_train = []
for i in range(2,10):
  decisiontree_model = DecisionTreeClassifier(max_depth=i, random_state=40)
  decisiontree_model.fit(X_train_smote, y_train_smote)
  max_depth_list.append(i)

  #Recall train data
  y_pred_decisiontree_train = decisiontree_model.predict(X_train_smote)
  recall_score_decisiontree_train = recall_score(y_train_smote, y_pred_decisiontree_train)
  recall_score_list_train.append(recall_score_decisiontree_train)
  #Cross validation training
  recall_score_cross_val = cross_val_score(decisiontree_model, X_train_smote, y_train_smote, cv = 5, scoring = 'recall')
  recall_score_list_train_cross_val.append(np.mean(recall_score_cross_val))

  #Recall test data
  y_pred_decisiontree = decisiontree_model.predict(X_test)
  recall_score_decisiontree = recall_score(y_test, y_pred_decisiontree)
  recall_score_list.append(recall_score_decisiontree)
  print(f"Recall dla test_data: {recall_score_decisiontree}\nRecall dla train_data: {recall_score_decisiontree_train}\nRecall dla cross_val: {np.mean(recall_score_cross_val)}")

  #Confusion matrix
  confusion_matrix_dt = confusion_matrix(y_test,y_pred_decisiontree)
  # sns.heatmap(confusion_matrix_dt, annot=True, fmt="d", cmap="Blues")
  # plt.xlabel('Predicted')
  # plt.ylabel('True')
  # plt.title(f"Max depth = {i}")
  # plt.show()
  # print(confusion_matrix_dt)
  #Accuracy score
  accuracy_score_dt = accuracy_score(y_test, y_pred_decisiontree)
  accuracy_score_list_dt.append(accuracy_score_dt)

  #Accuracy score train
  accuracy_score_dt_train = accuracy_score(y_train_smote, y_pred_decisiontree_train)
  accuracy_score_list_dt_train.append(accuracy_score_dt_train)

  

width = 0.3
bar1 = max_depth_list
bar2 = [i+width for i in bar1]
bar3 = [i+width*2 for i in bar1]
plt.bar(bar1, recall_score_list, width, label ='Test data')
plt.bar(bar2, recall_score_list_train, width, label ='Train data')
plt.bar(bar3, recall_score_list_train_cross_val, width, label ='Cross validation data')
plt.xlabel('Max depth')
plt.ylabel('Recall')
plt.title('DecisionTree SMOTE dataset')
plt.grid(True)
plt.legend()
plt.show()

width = 0.3
bar1 = max_depth_list
bar2 = [i+width for i in bar1]
plt.bar(bar1, accuracy_score_list_dt, width, label = 'Test data')
plt.bar(bar2, accuracy_score_list_dt_train, width, label = 'Train data')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('DecisionTree SMOTE dataset')
plt.grid(True)
plt.legend()
plt.show()


#Decision Tree with ADASYN oversampling method
max_depth_list = []
recall_score_list = []
recall_score_list_train = []
recall_score_list_train_cross_val = []
for i in range(2,10):
  decisiontree_model = DecisionTreeClassifier(max_depth=i, random_state=40)
  decisiontree_model.fit(X_train_adasyn, y_train_adasyn)
  max_depth_list.append(i)

  #Recall train data
  y_pred_decisiontree_train = decisiontree_model.predict(X_train_adasyn)
  recall_score_decisiontree_train = recall_score(y_train_adasyn, y_pred_decisiontree_train)
  recall_score_list_train.append(recall_score_decisiontree_train)
  #Cross validation training
  recall_score_cross_val = cross_val_score(decisiontree_model, X_train_adasyn, y_train_adasyn, cv = 5, scoring = 'recall')
  recall_score_list_train_cross_val.append(np.mean(recall_score_cross_val))

  #Recall test data
  y_pred_decisiontree = decisiontree_model.predict(X_test)
  recall_score_decisiontree = recall_score(y_test, y_pred_decisiontree)
  recall_score_list.append(recall_score_decisiontree)
  print(f"Recall dla test_data: {recall_score_decisiontree}\nRecall dla train_data: {recall_score_decisiontree_train}\nRecall dla cross_val: {np.mean(recall_score_cross_val)}")

width = 0.3
bar1 = max_depth_list
bar2 = [i+width for i in bar1]
bar3 = [i+width*2 for i in bar1]
plt.bar(bar1, recall_score_list, width, label ='Test data')
plt.bar(bar2, recall_score_list_train, width, label ='Train data')
plt.bar(bar3, recall_score_list_train_cross_val, width, label ='Cross validation data')
plt.xlabel('Max depth')
plt.ylabel('Recall')
plt.title('DecisionTree ADASYN dataset')
plt.grid(True)
plt.legend()
plt.show()

#Random Forest with SMOTE data
n_estimators = []
recall_score_list_rf = []
recall_score_list_train_rf = []
recall_score_list_train_cross_val_rf = []
accuracy_score_list_dt = []
accuracy_score_list_dt_train = []
for i in range(10,21):
  randomforest_model = RandomForestClassifier(max_depth=2, n_estimators=i, random_state=40)
  randomforest_model.fit(X_train_smote, y_train_smote)
  n_estimators.append(i)

  #Recall train data
  y_pred_randomforest_train = randomforest_model.predict(X_train_smote)
  recall_score_randomforest_train = recall_score(y_train_smote, y_pred_randomforest_train)
  recall_score_list_train_rf.append(recall_score_randomforest_train)
  #Cross validation training
  recall_score_cross_val_randomforest = cross_val_score(randomforest_model, X_train_smote, y_train_smote, cv = 5, scoring = 'recall')
  recall_score_list_train_cross_val_rf.append(np.mean(recall_score_cross_val_randomforest))

  #Recall test data
  y_pred_randomforest = randomforest_model.predict(X_test)
  recall_score_randomforest = recall_score(y_test, y_pred_randomforest)
  recall_score_list_rf.append(recall_score_randomforest)
  print(f"Recall dla test_data: {recall_score_randomforest}\nRecall dla train_data: {recall_score_randomforest_train}\nRecall dla cross_val: {np.mean(recall_score_cross_val_randomforest)}")

  #Accuracy score
  accuracy_score_dt = accuracy_score(y_test, y_pred_randomforest)
  accuracy_score_list_dt.append(accuracy_score_dt)

  #Accuracy score train
  accuracy_score_dt_train = accuracy_score(y_train_smote, y_pred_randomforest_train)
  accuracy_score_list_dt_train.append(accuracy_score_dt_train)

  confusion_matrix_rf = confusion_matrix(y_test,y_pred_randomforest)
  # sns.heatmap(confusion_matrix_rf, annot=True, fmt="d", cmap="Blues")
  # plt.xlabel('Predicted')
  # plt.ylabel('True')
  # plt.title(f"n_estimators = {i}")
  # plt.show()
  # print(confusion_matrix_rf)

width = 0.3
bar1 = n_estimators
bar2 = [i+width for i in bar1]
bar3 = [i+width*2 for i in bar1]
plt.bar(bar1, recall_score_list_rf, width, label ='Test data')
plt.bar(bar2, recall_score_list_train_rf, width, label ='Train data')
plt.bar(bar3, recall_score_list_train_cross_val_rf, width, label ='Cross validation data')
plt.xlabel('n_estimators')
plt.title('RandomForest Model')
plt.ylabel('Recall')
plt.grid(True)
plt.legend()
plt.show()

width = 0.3
bar1 = n_estimators
bar2 = [i+width for i in bar1]
plt.bar(bar1, accuracy_score_list_dt, width, label = 'Test data')
plt.bar(bar2, accuracy_score_list_dt_train, width, label = 'Train data')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('RandomForest dataset')
plt.grid(True)
plt.legend()
plt.show()

#Data standarization for SVM

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

param_grid = {
  'C' : [0.1, 1, 10],
  'gamma': [0.01, 0.1, 1],
  'kernel': ['linear', 'rbf', 'poly']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_scaled,y_train_smote)

print(f"Best parameters: {grid_search.best_params_}")
svm_model = SVC(kernel='rbf', gamma= 0.1, C= 10, random_state=40)
svm_model.fit(X_train_scaled, y_train_smote)
y_pred_svm = svm_model.predict(X_test_scaled)
recall_score_svm = recall_score(y_test, y_pred_svm)
y_pred_svm_train = svm_model.predict(X_train_scaled)
recall_score_svm_train = recall_score(y_train_smote, y_pred_svm_train)
recall_score_svm_train_cross_val = cross_val_score(svm_model, X_train_scaled, y_train_smote, cv=5, scoring='recall')
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_svm_train = accuracy_score(y_train_smote, y_pred_svm_train)
accuracy_svm_train_cross_val = cross_val_score(svm_model, X_train_scaled, y_train_smote, cv=5, scoring='accuracy')
print(f'Recall dla test_data_SVM: {recall_score_svm}\nRecall dla train_data_SVM: {recall_score_svm_train}\nRecall dla cross_val_SVM: {np.mean(recall_score_svm_train_cross_val)}')

colorss = ['blue', 'lightblue', 'darkblue']
plt.bar(['Test data', 'Train data', 'Cross validation'], [recall_score_svm, recall_score_svm_train, np.mean(recall_score_svm_train_cross_val)], color = colorss)
plt.title('SVM model')
plt.ylabel('Recall')
plt.xlabel('Data')
plt.show()

colors = ['green', 'lightgreen', 'darkgreen']
plt.bar(['Test data', 'Train data', 'Cross validation'], [accuracy_svm, accuracy_svm_train, np.mean(accuracy_svm_train_cross_val)], color = colors)
plt.title('SVM model')
plt.ylabel('Accuracy')
plt.xlabel('Data')
plt.show()


#Undersampling
tl = TomekLinks()
X_train_tomek, y_train_tomek = tl.fit_resample(X_train, y_train)

credit_rejected_tomek = y_train_tomek.value_counts().get(1,0)
credit_approved_tomek = y_train_tomek.value_counts().get(0,0)

bars = plt.bar(['Approved', 'Rejected'], [credit_approved_tomek, credit_rejected_tomek], color = ['green', 'red'])
for bar in bars:
  plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3 , str(bar.get_height()), ha = 'center')
plt.title('Data distribiution after TomekLinks Undersampling', fontsize = 14)
plt.xlabel('Categories', fontsize = 11)
plt.ylabel('Number of data', fontsize = 11)  
plt.show()

#DecisionTree Tomek data
max_depth_list = []
recall_score_list = []
recall_score_list_train = []
recall_score_list_train_cross_val = []
for i in range(20,25):
  decisiontree_model = DecisionTreeClassifier(max_depth=i, random_state=40)
  decisiontree_model.fit(X_train_tomek, y_train_tomek)
  max_depth_list.append(i)

  #Recall train data
  y_pred_decisiontree_train = decisiontree_model.predict(X_train_tomek)
  recall_score_decisiontree_train = recall_score(y_train_tomek, y_pred_decisiontree_train)
  recall_score_list_train.append(recall_score_decisiontree_train)
  #Cross validation training
  recall_score_cross_val = cross_val_score(decisiontree_model, X_train_tomek, y_train_tomek, cv = 5, scoring = 'recall')
  recall_score_list_train_cross_val.append(np.mean(recall_score_cross_val))

  #Recall test data
  y_pred_decisiontree = decisiontree_model.predict(X_test)
  recall_score_decisiontree = recall_score(y_test, y_pred_decisiontree)
  recall_score_list.append(recall_score_decisiontree)
  print(f"Recall dla test_data: {recall_score_decisiontree}\nRecall dla train_data: {recall_score_decisiontree_train}\nRecall dla cross_val: {np.mean(recall_score_cross_val)}")

width = 0.3
bar1 = max_depth_list
bar2 = [i+width for i in bar1]
bar3 = [i+width*2 for i in bar1]
plt.bar(bar1, recall_score_list, width, label ='Test data')
plt.bar(bar2, recall_score_list_train, width, label ='Train data')
plt.bar(bar3, recall_score_list_train_cross_val, width, label ='Cross validation data')
plt.xlabel('Max depth')
plt.ylabel('Recall')
plt.title('DecisionTree TOMEK dataset')
plt.grid(True)
plt.legend()
plt.show()


#RandomForest Ceteris paribus
randomforest_model_cp = RandomForestClassifier(max_depth=5, n_estimators=11, random_state=40)
randomforest_model_cp.fit(X_train_smote, y_train_smote)
y_pred_randomforest_train_cp = randomforest_model_cp.predict(X_train_smote)
recall_score_cross_val_randomforest_cp = cross_val_score(randomforest_model_cp, X_train_smote, y_train_smote, cv = 5, scoring = 'recall')
y_pred_randomforest_cp = randomforest_model_cp.predict(X_test)

#Analiza interpretowalności dla zmiennych liczbowych
PartialDependenceDisplay.from_estimator(
    randomforest_model_cp, 
    X_train_smote, 
    features=['Annual_income', 'Age', 'Employed_days'],
    grid_resolution=50
)
plt.show()

#Analiza interpretowalności dla zmiennych binarnych
binary_columns = [
    'Car_Owner', 
    'Gender', 
    'Marital_status', 
    'Propert_Owner',
    'Type_Income_Pensioner', 
    'Type_Income_Working', 
    'Type_Income_State servant',
    'Education_Higher education', 
    'Education_Lower secondary'
]

differences = []
feature_names = []

for feature in binary_columns:
    feature_index = X_train_smote.columns.get_loc(feature)
    

    pdp_result = partial_dependence(
        estimator=randomforest_model_cp,
        X=X_train_smote,
        features=[feature_index],
        kind='average' 
    )
    
    pdp_values = pdp_result['average'][0]
    class_0_value = pdp_values[0] 
    class_1_value = pdp_values[1] 
    difference = class_1_value - class_0_value  
    
    differences.append(difference)
    feature_names.append(feature)

plt.figure(figsize=(10, 6))
plt.barh(feature_names, differences, color=['green' if diff > 0 else 'red' for diff in differences])
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Partial Dependence Difference (Class 1 - Class 0)')
plt.title('Partial Dependence Difference for Binary Features')
plt.show()