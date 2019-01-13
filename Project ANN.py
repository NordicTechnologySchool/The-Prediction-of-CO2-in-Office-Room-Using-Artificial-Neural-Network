import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_excel('D:\IAQ ANN.xlsx')
print(df.shape)
print(df.head)

print(df.isnull().sum()) #cek missing data
print(list(df.columns))
print(df['CO2'].value_counts()) #cek imbalance
print(df.info())

lable_encoder = LabelEncoder()
temp = df.copy()
temp.iloc[:, 0] = lable_encoder.fit_transform(df.iloc[:, 0])
print(lable_encoder.classes_)
temp.iloc[:, 1] = lable_encoder.fit_transform(df.iloc[:, 1])
print(lable_encoder.classes_)
temp.iloc[:, 3] = lable_encoder.fit_transform(df.iloc[:, 3])
print(lable_encoder.classes_)
print(temp)
print(temp.info())
del temp['Air-Conditioning']
del temp['Inlet Airflow']
#del temp['Inlet Airflow']
#del temp['No Of Occupants']

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
standard_scaler = StandardScaler()


from sklearn.neural_network import MLPRegressor #value
import numpy as np
import matplotlib.pyplot as plt
import  random
np.random.seed(3)

X = temp.drop('CO2', axis=1)
y = temp['CO2']

normalizer = Normalizer()
#min_max_scaler = MinMaxScaler()

print("Standardization")
Xnorm=standard_scaler.fit_transform(X)
df3=pd.DataFrame(Xnorm)
corr=df3.corr()
corr.style.background_gradient()
#print("Normalizing")
#Xnorm=normalizer.fit_transform(X)
#print("MinMax Scaling")
#print(min_max_scaler.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xnorm, y, random_state=3)

#Training the model

mlp = MLPRegressor(activation='logistic', hidden_layer_sizes=(15,), learning_rate_init=0.2)
mlp.fit(X_train,y_train)
print(mlp)

predictions = mlp.predict(X_test)
predictions_train = mlp.predict(X_train)

mlp.coefs_[0]
print(mlp.coefs_[0])

print(mlp.intercepts_[0])

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100
mean_absolute_percentage_error(y_test, predictions)
MAPE=mean_absolute_percentage_error(y_test, predictions)
MAPE_train=mean_absolute_percentage_error(y_train, predictions_train)
print(MAPE_train)
print(MAPE)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predictions)
MSE_train=mean_squared_error(y_train, predictions_train)
MSE=mean_squared_error(y_test, predictions)

from sklearn.metrics import r2_score
R2=r2_score(y_test, predictions)
print(R2)

print(MSE)
print(MSE_train)
print('predicts result')
print(predictions)
print('actual')
print(y_test)
df2=pd.DataFrame(predictions,y_test)
print(df2)

from sklearn import datasets, linear_model

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))





