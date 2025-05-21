import streamlit as st
from model_predict import predict_water_quality
from utilities import load_water_data, split_scale_data
import matplotlib.pyplot as plt
import pandas as pd


# file_rawData = "/Users/yanjiasun/Documents/Yanjia_code/WQI-Delaware-ML-WS/Data/Complete_Data_WQI.xlsx"
file_rawData = "Complete_Data_WQI.xlsx"

df_water_data = load_water_data(file_rawData)
print('=== Loaded raw water data successfully!!! ===')

features = ['Sulfate (mg/L)', 'Chloride (mg/L)', 'Sodium (mg/L)', 'Potassium (mg/L)',
                'Calcium (mg/L)', 'Magnesium (mg/L)', 'Total Dissolved Solids (mg/L)',
                'Turbidity (NTU)', 'Temperature (deg C)', 'pH',
                'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Fecal Coliform (cfu/100ml)']


X_train_scaled, X_test_scaled, y_train, y_test, dates_trian, dates_test = split_scale_data(df_water_data)
print('=== Splited and scaled data successfully!!! ===')
print(X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape)


df_X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)
print('Shape of df_X_test_scaled: ', df_X_test_scaled.shape)

df_y_test = y_test.to_frame('WQI')
print('Shape of df_y_test: ', df_y_test.shape)

df_dates_test = dates_test.to_frame('ActivityStartDate')
print('Shape of df_dates_test: ', df_dates_test.shape)



# st.title("Water Quality Predictor\n(XGBoost With Lag Features)")
st.set_page_config(page_title="Water Quality Index Predictor", layout="wide")

st.title("🌊 Water Quality Index Predictor 🌊\nDriven by XGBoost With Lag Features")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://www.americanrivers.org/wp-content/uploads/2018/04/5453470115_facb1d7a45_o.jpg");
    background-size: cover;
    background-position: center;
}}
[data-testid="stHeader"] {{
    background-color: rgba(255, 255, 255, 0.8);
}}
</style>
"""
# background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");

st.markdown(page_bg_img, unsafe_allow_html=True)


with st.container():
    # st.title("🌊 Water Quality Index Predictor 🌊\nDriven by XGBoost With Lag Features")

    # uploaded_file = st.file_uploader("Upload a CSV file (with 13 features + WQI)", type=["csv"])

    if df_X_test_scaled is not None:
        # df = pd.read_csv(uploaded_file, parse_dates=['ActivityStartDate'])
        df = df_X_test_scaled.copy()

        if len(df) < 3:
            st.error("CSV must contain at least 3 rows to extract lag features.")
        else:
            st.subheader("Input Features (Last 3 Days)")

            # latest_features = df.iloc[-3:, 1:-1].values.flatten().reshape(1, -1)
            latest_features = df.iloc[-1:].values.flatten().reshape(1, -1)

            col1, col2 = st.columns(2)
            with col1:
                for i in range(0, len(df.columns)//2):
                    # st.text_input(df.columns[1+i], value=f"{df.iloc[-1, 1+i]:.2f}", disabled=True)
                    st.number_input(df.columns[i], value=df.iloc[-1, i], disabled=False)
            with col2:
                for i in range(len(df.columns)//2, len(df.columns)):
                    # st.text_input(df.columns[1+i], value=f"{df.iloc[-1, 1+i]:.2f}", disabled=True)
                    st.number_input(df.columns[i], value=df.iloc[-1, i], disabled=False)

            if st.button("🔍 Predict WQI"):
                prediction = predict_water_quality(latest_features) 

                if prediction < 25:
                    quality = "🌟 Excellent Water Quality 😊👍"
                elif prediction < 50:
                    quality = "✅ Good Water Quality 🙂"
                elif prediction < 75:
                    quality = "⚠️ Poor Water Quality 😟"
                elif prediction < 100:
                    quality = "🚨 Very Poor Water Quality 😢"
                else:
                    quality = "❌ Unsuitable For Drinking 🚱"

                st.markdown(f"### Predicted WQI: `{prediction:.2f}`")
                st.markdown(f"## {quality}")

            # st.subheader("📈 WQI Trend (Past 10 Days)")

            # fig, ax = plt.subplots()
            # ax.plot(df_dates_test['ActivityStartDate'], df_y_test['WQI'], color='blue', marker='o')
            # ax.set_xlabel("Date")
            # ax.set_ylabel("WQI")
            # ax.set_title("10-Day WQI Trend")
            # ax.grid(True)
            # st.pyplot(fig)
