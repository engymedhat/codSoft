import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix ,accuracy_score

data=pd.read_csv('C:/Users/engy medhat/Desktop/Titanic-Dataset.csv')

#descrice data
print(data.info)
print(data.isna().sum())

#handling nulls
data['Age'].fillna(data['Age'].median(),inplace=True)
data['Embarked'].fillna('S',inplace=True)
data.drop(columns=['Cabin'], inplace=True)
print(data.isna().sum())

#encoding
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

#drop coloumns not important
data.drop(columns=['Name','Ticket','PassengerId'],inplace=True)
print(data.columns)

#split data
y=data['Survived']
x=data.drop(columns=['Survived'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#train model 
model=LogisticRegression()
model.fit(X_train,y_train)

#prediction
y_pred=model.predict(X_test)
print('Accuracy Score:\n',accuracy_score(y_test,y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))



