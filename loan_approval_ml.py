#This is an end-to-end supervised ML pipeline using Logistiic Regression, Naive Bayes, KNN to predict loan approval.
#Implemented binary classification along with EDA, Feature Engineering and Model evalution(Precision,Accuracy,Recall).


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, precision_score, accuracy_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#reading data............

df = pd.read_csv("loan_approval_data.csv")
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

#null values changing..........

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

#Encoding...........

le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)

df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

#Feature Engineering ..............

df ["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

df ["Applicant_Income_log"] = np.log1p(df["Applicant_Income"])

X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income"])
y = df["Loan_Approved"]

#Starting................

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models train ------ 

# Logistic Regression................................
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

print("\nLogistic Regression Model")
print("recall score:", recall_score(y_test, y_pred))
print("accuracy score:", accuracy_score(y_test, y_pred))
print("precision score:", precision_score(y_test, y_pred))

#Navie Bayes Model...................

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred = nb_model.predict(X_test_scaled)

print("\nNavie Bayes Model")
print("recall score : ", recall_score(y_test, y_pred))
print("accuracy score : ", accuracy_score(y_test, y_pred))
print("precision score : ", precision_score(y_test, y_pred))

# KNN Model.................

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_scaled,y_train)

y_pred = knn_classifier.predict(X_test_scaled)

print("\nKNN Model")
print("recal score : ",recall_score(y_test,y_pred))
print("accuracy score : ", accuracy_score(y_test,y_pred))
print("precision score : ",precision_score(y_test, y_pred))


############### THANK YOU ###################