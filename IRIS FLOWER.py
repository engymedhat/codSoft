import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

data=pd.read_csv("C:/Users/engy medhat/Desktop/IRIS.csv")

#describe data
print(data.info())
print(data.isna().sum())

#encoding
label_encoder = LabelEncoder()
data['species']=label_encoder.fit_transform(data['species'])

#spliting data
y=data['species']
x=data.drop(columns=['species'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#train model
model=KNeighborsClassifier()
model.fit(X_train,y_train)

#prediction
y_pred=model.predict(X_test)
print('Accuracy Score:\n',accuracy_score(y_test,y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
