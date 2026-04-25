import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn import metrics
st.set_page_config(page_title="Big Mart Sales Predictor", layout="wide")
# 🌙 DARK UI POLISH
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stApp {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)
st.title("🛒 Big Mart Sales Prediction App")
st.markdown("### Predict product sales using Machine Learning")
@st.cache_data
def load_data():
    df = pd.read_csv('Train_bigmart.csv')
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    mode = df.pivot_table(
        values="Outlet_Size",
        columns="Outlet_Type",
        aggfunc=lambda x: x.mode()[0]
    )
    missing = df['Outlet_Size'].isnull()
    df.loc[missing, 'Outlet_Size'] = df.loc[missing, 'Outlet_Type'].apply(
        lambda x: mode[x].values[0] if x in mode else np.nan
    )
    df['Outlet_Size'].fillna('Medium', inplace=True)
    df['Outlet_Size'] = df['Outlet_Size'].astype(str)
    df.replace({
        "Item_Fat_Content": {
            'low fat': 'Low Fat',
            'LF': 'Low Fat',
            'reg': 'Regular'
        }
    }, inplace=True)
    return df
data = load_data()
original_data = data.copy()
@st.cache_resource
def encode_data(df):
    df = df.copy()
    encoders = {}
    for col in [
        "Item_Identifier", "Item_Fat_Content", "Item_Type",
        "Outlet_Identifier", "Outlet_Size",
        "Outlet_Location_Type", "Outlet_Type"
    ]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders
data, encoder_dict = encode_data(data)
@st.cache_resource
def train_model(df):
    X = df.drop("Item_Outlet_Sales", axis=1)
    Y = df["Item_Outlet_Sales"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = XGBRegressor(random_state=2)
    model.fit(X_train, Y_train)
    return model, X, X_train, X_test, Y_train, Y_test
model, X, X_train, X_test, Y_train, Y_test = train_model(data)
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Train R² Score", f"{metrics.r2_score(Y_train, model.predict(X_train)):.3f}")
col2.metric("Test R² Score", f"{metrics.r2_score(Y_test, model.predict(X_test)):.3f}")
st.subheader("📌 Feature Importance")
fig, ax = plt.subplots()
plot_importance(model, ax=ax)
st.pyplot(fig)
with st.expander("📈 Data Visualization"):
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(original_data['Item_MRP'], kde=True, ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x='Outlet_Type', data=original_data, ax=ax)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        st.pyplot(fig)
st.sidebar.header("🧾 Enter Product Details")
def user_input():
    return {
        "Item_Identifier": st.sidebar.selectbox("Item Identifier", original_data['Item_Identifier'].unique()),
        "Item_Weight": st.sidebar.number_input("Item Weight", min_value=0.0),
        "Item_Fat_Content": st.sidebar.selectbox("Fat Content", original_data['Item_Fat_Content'].unique()),
        "Item_Visibility": st.sidebar.number_input("Item Visibility", min_value=0.0),
        "Item_Type": st.sidebar.selectbox("Item Type", original_data['Item_Type'].unique()),
        "Item_MRP": st.sidebar.number_input("Item MRP", min_value=0.0),
        "Outlet_Identifier": st.sidebar.selectbox("Outlet ID", original_data['Outlet_Identifier'].unique()),
        "Outlet_Establishment_Year": st.sidebar.number_input("Establishment Year", min_value=1980, max_value=2025),
        "Outlet_Size": st.sidebar.selectbox("Outlet Size", original_data['Outlet_Size'].unique()),
        "Outlet_Location_Type": st.sidebar.selectbox("Location Type", original_data['Outlet_Location_Type'].unique()),
        "Outlet_Type": st.sidebar.selectbox("Outlet Type", original_data['Outlet_Type'].unique())
    }
input_dict = user_input()
error = False
for col in encoder_dict:
    if col in input_dict:
        try:
            input_dict[col] = encoder_dict[col].transform([input_dict[col]])[0]
        except:
            st.sidebar.error(f"Invalid value for {col}")
            error = True
if not error and st.sidebar.button("🚀 Predict Sales"):
    input_data = np.array([input_dict[col] for col in X.columns]).reshape(1, -1)
    pred = model.predict(input_data)
    st.subheader("💰 Prediction Result")
    st.success(f"Estimated Sales: ₹ {pred[0]:,.2f}")





