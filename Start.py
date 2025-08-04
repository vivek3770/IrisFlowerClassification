import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load IRIS dataset
@st.cache_data
def load_data():
    df = pd.read_csv("IRIS.csv")
    return df

df = load_data()

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['species'])

# Features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['label']

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

st.title("Iris Flower Classification")

st.write(
    """
    Enter the **sepal** and **petal** measurements of an Iris flower to predict its species!
    """
)

# Input widgets
sepal_length = st.number_input('Sepal Length (cm)', min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=2.0, max_value=4.5, value=3.5, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=1.0, max_value=7.0, value=1.4, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.1, max_value=2.5, value=0.2, step=0.1)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Prediction
if st.button("Classify Iris Flower"):
    pred_label = rf.predict(input_data)[0]
    pred_species = le.inverse_transform([pred_label])[0]
    st.success(f"The predicted species is: **{pred_species}**")

# Optionally, show the dataset and info
if st.checkbox("Show Dataset"):
    st.dataframe(df)