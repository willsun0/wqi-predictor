import streamlit as st
from model_predict import predict_water_quality
from utilities import load_water_data, split_scale_data, split_data, scale_data, create_lag_features
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



file_rawData = "Complete_Data_WQI.xlsx"

df_water_data = load_water_data(file_rawData)
print('=== Loaded raw water data successfully!!! ===')

features = ['Sulfate (mg/L)', 'Chloride (mg/L)', 'Sodium (mg/L)', 'Potassium (mg/L)',
                'Calcium (mg/L)', 'Magnesium (mg/L)', 'Total Dissolved Solids (mg/L)',
                'Turbidity (NTU)', 'Temperature (deg C)', 'pH',
                'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Fecal Coliform (cfu/100ml)']

# selected_features = ['Temperature (deg C)',
#                 'Dissolved Oxygen (mg/L)', 
#                 'Turbidity (NTU)',
#             #  'Biochemical Oxygen Demand (mg/L)',
#                 'Total Dissolved Solids (mg/L)',
#                 'Fecal Coliform (cfu/100ml)', 
#                 'pH', 
#                 'Sulfate (mg/L)'
#                 ]

# df_water_data_lag = df_water_data.copy()
# for lag in range(1, 4):
#     for col in features:
#         df_water_data_lag[f'{col}_lag{lag}'] = df_water_data_lag[col].shift(lag)
# df_water_data_lag.dropna(inplace=True)
# print('Shape of df_water_data_lag: ', df_water_data_lag.shape)


# X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, dates_trian, dates_test = split_scale_data(df_water_data)
X_train, X_test, y_train, y_test, dates_trian, dates_test = split_data(df_water_data)
print('=== Splited data successfully!!! ===')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

selected_df_water_data = df_water_data.tail(4).copy()
selected_df_water_data = selected_df_water_data.drop(columns=['WQI', 'ActivityStartDate'])


# df_X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)
# print('Shape of df_X_test_scaled: ', df_X_test_scaled.shape)
df_y_test = y_test.to_frame('WQI')
print('Shape of df_y_test: ', df_y_test.shape)
df_dates_test = dates_test.to_frame('ActivityStartDate')
print('Shape of df_dates_test: ', df_dates_test.shape)


# st.title("Water Quality Predictor\n(XGBoost With Lag Features)")
st.set_page_config(page_title="Water Quality Index Predictor", layout="wide")

# st.markdown(
#     """
#     <style>
#     .main {
#         background-image: url("https://www.americanrivers.org/wp-content/uploads/2018/04/5453470115_facb1d7a45_o.jpg");
#         background-size: cover;
#         background-attachment: fixed;
#     }
#     .block-container {
#         background-color: rgba(255, 255, 255, 0.8);
#         padding: 2rem;
#         border-radius: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

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
    st.title("🌊 Water Quality Index Predictor 🌊\nDriven by XGBoost With Lag Features (Past 3 Days)")

    # uploaded_file = st.file_uploader("Upload a CSV file (with 13 features + WQI)", type=["csv"])

    if X_test is not None:
        # df = pd.read_csv(uploaded_file, parse_dates=['ActivityStartDate'])
        if len(X_test) < 3:
            st.error("Raw data must contain at least 3 rows to extract lag features.")
        else:
            st.subheader("Current Input Features")

            col1, col2 = st.columns(2)
            input_values = []
            with col1:
                # for i in range(0, len(X_test.columns)//2):
                for i in range(0, len(features)//2):
                    input_values.append(st.number_input(X_test.columns[i], value=X_test.iloc[-1, i], disabled=False) )
            with col2:
                for i in range(len(X_test.columns)//2, len(X_test.columns)):
                    input_values.append(st.number_input(X_test.columns[i], value=X_test.iloc[-1, i], disabled=False) )

            print('input_values: ', input_values)
            # df = df_X_test_scaled.copy()
            # latest_features = df.iloc[-3:, 1:-1].values.flatten().reshape(1, -1)
            # latest_features = df.iloc[-1:].values.flatten().reshape(1, -1)

            df_input_values = pd.DataFrame(np.array(input_values).reshape(1, -1), columns=features)
            print('Shape of df_input_values: ', df_input_values.shape)
            # print('df_input_values:', df_input_values)
            arr_latest_features = scale_data(X_train, df_input_values)
            print('arr_latest_features:', arr_latest_features)

            if st.button("🔍 Predict WQI"):
                prediction = predict_water_quality(arr_latest_features,'xgb') 

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










# # Example input form
# v_Sulfate = st.number_input("Sulfate", value=-1.43583699)
# v_Chloride = st.number_input("Chloride", value=-1.14929032)
# v_Sodium = st.number_input("Sodium", value=-1.23983555)
# v_Potassium = st.number_input("Potassium", value=-0.20457395)
# v_Calcium = st.number_input("Calcium", value=-1.90392694)
# v_Magnesium = st.number_input("Magnesium", value=-1.67340076)
# v_TDS = st.number_input("Total Dissolved Solids", value=-1.15374372)
# v_Turbidity = st.number_input("Turbidity", value=-0.3243976)
# v_Temperature = st.number_input("Temperature", value=1.14948521)
# v_pH = st.number_input("pH", value=-1.68849039)
# v_DO = st.number_input("Dissolved Oxygen", value=-1.12680255)
# v_Nitrate = st.number_input("Nitrate", value=-0.67250243)
# v_FC = st.number_input("Fecal Coliform", value=0.96385761)


# # Add more inputs as per your features...

# if st.button("Predict"):
#     features = [
#                 v_Sulfate,
#                 v_Chloride,
#                 v_Sodium,
#                 v_Potassium,
#                 v_Calcium,
#                 v_Magnesium,
#                 v_TDS,
#                 v_Turbidity,
#                 v_Temperature,
#                 v_pH,
#                 v_DO,
#                 v_Nitrate,
#                 v_FC
#                 ]  # Add all inputs in order
    
#     prediction = predict_water_quality(features)

#     st.success(f"Predicted Water Quality Index: {prediction:.2f}")
