#frontend streamlit for the knnreg.ipynb file
from sklearn.model_selection import train_test_split
import streamlit as st
st.title("KNN Regression")
st.write("This is a KNN regression model to predict the ex showroom price of a car based on its features.")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("bike_knnreg.csv")
st.write("The dataset:")
st.dataframe(df.head())
def iqr(col):
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    df[col]=df[col].clip(lower_bound, upper_bound)
iqr("year")
iqr("km_driven")
iqr("ex_showroom_price")
df.plot(kind="box", figsize=(10, 6),layout=(2, 3))
df["ex_showroom_price"]=df["ex_showroom_price"].fillna(df["ex_showroom_price"].median())
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded_cols=ohe.fit_transform(df[["name","seller_type","owner"]])
encoded_df=pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(["name","seller_type","owner"]))
df=pd.concat([df, encoded_df], axis=1)
df.drop(["name","seller_type","owner"], axis=1, inplace=True)
from sklearn.preprocessing import MinMaxScaler
price_scaler=MinMaxScaler()
df["ex_showroom_price"]=price_scaler.fit_transform(df[["ex_showroom_price"]])

x=df.drop("selling_price", axis=1)
y=df["selling_price"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#scaling the features using standard scaler
from sklearn.preprocessing import StandardScaler
feature_scaler=StandardScaler()
x_train=feature_scaler.fit_transform(x_train)
x_test=feature_scaler.transform(x_test)

from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)

#to take user input for prediction
st.write("Enter the features of the car to predict its selling price:")
name=st.text_input("Name of the bike")
year=st.number_input("Year of manufacture", min_value=1990, max_value=2024, value=2010)
km_driven=st.number_input("Kilometers driven", min_value=0, max_value=1000000, value=50000)
ex_showroom_price=st.number_input("Ex showroom price", min_value=0, max_value=1000000, value=50000)
seller_type=st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
owner=st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
#encoding the user input
input_df=pd.DataFrame([[name, year, km_driven, ex_showroom_price, seller_type, owner]], columns=["name", "year", "km_driven", "ex_showroom_price", "seller_type", "owner"])
input_encoded=ohe.transform(input_df[["name","seller_type","owner"]])
input_encoded_df=pd.DataFrame(input_encoded, columns=ohe.get_feature_names_out(["name","seller_type","owner"]))
input_df=pd.concat([input_df, input_encoded_df], axis=1)
input_df.drop(["name","seller_type","owner"], axis=1, inplace=True)
input_df["ex_showroom_price"]=price_scaler.transform(input_df[["ex_showroom_price"]]).ravel()
input_df=input_df.reindex(columns=x.columns, fill_value=0)
input_scaled=feature_scaler.transform(input_df)
#predicting the selling price
if st.button("Predict"):
    prediction=knn.predict(input_scaled)
    st.write("The predicted selling price of the bike is:", prediction[0])

