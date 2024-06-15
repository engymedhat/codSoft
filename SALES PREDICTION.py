import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

data=pd.read_csv("C:/Users/engy medhat/Desktop/advertising.csv")

#describe data
print(data.info())
print(data.isna().sum())

#spliting data
y=data['Sales']
x=data.drop(columns=['Sales'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#train model
model=LinearRegression()
model.fit(X_train,y_train)

#prediction
y_pred=model.predict(X_test)
print('r2_score:\n',r2_score(y_test,y_pred))
print('mean_squared_error:\n', mean_squared_error(y_test, y_pred))
