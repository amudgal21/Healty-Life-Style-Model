#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load train and test datasets
train_data = pd.read_csv("D:\Google Downloads\Train_Data (1).csv")
test_data = pd.read_csv("D:\Google Downloads\Test_Data (1).csv")

# Drop any rows with missing values in the train data (you may handle it differently as per your needs)
train_data.dropna(inplace=True)

# Handle missing values in the test data
test_data.fillna(value=test_data.mode().iloc[0], inplace=True)  # Filling with the most frequent value for each column

# Encode categorical variables in train and test datasets
categorical_cols = ["Food preference", "Smoker?", "Living in?", "Any heriditary condition?", "Follow Diet", "Mental health management"]
label_encoder = LabelEncoder()
for col in categorical_cols:
    # Combine train and test data for encoding to handle unseen labels
    combined_data = pd.concat([train_data[col], test_data[col]], axis=0)
    label_encoder.fit(combined_data)
    train_data[col] = label_encoder.transform(train_data[col])
    test_data[col] = label_encoder.transform(test_data[col])

# Separate the target variable from train_data
X_train = train_data.drop(columns=["ID1", "Healthy"])
y_train = train_data["Healthy"]

# Standardize the features for better performance in some models (e.g., SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data.drop(columns=["ID1"]))


# In[17]:


pip install keras


# In[18]:


pip install tensorflow


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
}

# Train the models
for name, model in models.items():
    if name == "Neural Network":
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
    else:
        model.fit(X_train_scaled, y_train)


# In[20]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate the models
results = {}
for name, model in models.items():
    if name == "Neural Network":
        y_pred = model.predict(X_train_scaled)
        y_pred = (y_pred > 0.5).astype(int)
    else:
        y_pred = model.predict(X_train_scaled)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Display results
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print("=" * 30)


# In[21]:


# Make predictions using the selected model ("Random Forest")
selected_model = models["Random Forest"]
selected_model.fit(X_train_scaled, y_train)  # Re-train the model on the entire train data

# Predict the target variable for the test dataset
y_test_pred = selected_model.predict(X_test_scaled)

# Convert predictions to 0 or 1 based on the threshold (0.5 for Random Forest)
y_test_pred = (y_test_pred > 0.5).astype(int)

# Create a DataFrame with predictions and save it to a CSV file
submission = pd.DataFrame({"predictions": y_test_pred})
submission.to_csv("submission.csv", index=False)


# In[22]:


import os

# Get the current working directory
current_directory = os.getcwd()

# Check if the submission.csv file is present in the current directory
file_location = os.path.join(current_directory, "submission.csv")

# Check if the file exists
if os.path.exists(file_location):
    print(f"The submission.csv file is saved in the current directory: {file_location}")
else:
    print("The submission.csv file is not found.")


# In[ ]:




