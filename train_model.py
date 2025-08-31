import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

try:
    print("📥 Loading dataset...")
    dataset = pd.read_csv('heart.csv')
    print("✅ Dataset loaded.")

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    print("🔤 Encoding categorical features...")
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le6 = LabelEncoder()
    le8 = LabelEncoder()
    le10 = LabelEncoder()

    x[:, 1] = le1.fit_transform(x[:, 1])      # sex
    x[:, 2] = le2.fit_transform(x[:, 2])      # cp
    x[:, 6] = le6.fit_transform(x[:, 6])      # restecg
    x[:, 8] = le8.fit_transform(x[:, 8])      # exang
    x[:, 10] = le10.fit_transform(x[:, 10])   # slope

    print("📊 Splitting dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    print("📏 Scaling features...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print("🌲 Training Random Forest model...")
    model_randomforest = RandomForestClassifier()
    model_randomforest.fit(X_train, Y_train)
    print("✅ Model trained.")

    print("💾 Saving model and scaler...")
    os.makedirs('model', exist_ok=True)

    with open('model/model_randomforest.pkl', 'wb') as f:
        pickle.dump(model_randomforest, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(sc, f)

    print("🎉 All done! Model and scaler saved in 'model/'.")

except Exception as e:
    print("❌ An error occurred:", str(e))
