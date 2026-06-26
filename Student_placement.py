

import kagglehub
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)
import matplotlib.pyplot as plt
import pickle
from google.colab import files





path = kagglehub.dataset_download("sahilislam007/college-student-placement-factors-dataset")
file_path = os.path.join(path,"college_student_placement_dataset.csv")
df = pd.read_csv(file_path)
df_encoded = pd.get_dummies(df,columns=['Placement', 'Internship_Experience'],drop_first=True,dtype=int)
if 'College_ID' in df_encoded.columns:
    df_encoded.drop('College_ID', axis=1, inplace=True)
X = df_encoded.drop('Placement_Yes', axis=1)
Y = df_encoded['Placement_Yes']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
classifier = Sequential()
classifier.add(Dense(units=16,activation='relu',input_shape=(X_train.shape[1],)))
classifier.add(Dense(units=8,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()
history = classifier.fit(X_train,Y_train,validation_data=(X_test, Y_test),epochs=50,batch_size=32,verbose=1)
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5).astype(int)
accuracy = accuracy_score(Y_test,Y_pred)
loss, test_accuracy = classifier.evaluate(X_test,Y_test,verbose=0)



classifier.save("placement_model.keras")

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)




print("Dataset Path:", path)
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nFeatures Shape:", X.shape)
print("Target Shape:", Y.shape)
print("\nAccuracy:")
print(accuracy)
print("\nClassification Report:")
print(
    classification_report(
        Y_test,
        Y_pred
    )
)
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        Y_test,
        Y_pred
    )
)
print("\nTest Accuracy:", test_accuracy)
print("\nModel Saved Successfully!")
print("Scaler Saved Successfully!")



files.download("placement_model.keras")
files.download("scaler.pkl")