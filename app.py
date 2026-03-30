import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import isodate
import gdown

def preprocess(data):

    def convert_duration(duration):
        try:
            return str(isodate.parse_duration(duration))
        except:
            return "00:00:00"

    def duration_to_seconds(duration):
        try:
            h, m, s = map(int, duration.split(':'))
            return h * 3600 + m * 60 + s
        except:
            return 0

    def calculate_video_age(upload):
        try:
            upload = pd.to_datetime(upload)
            target = datetime(2024, 12, 31)
            return (target.year - upload.year) * 12 + (target.month - upload.month)
        except:
            return 0

    def count_tags(tags):
        return len(tags) if isinstance(tags, list) else 0

    # Convert numeric
    for col in ['Likes','Views','Comments','Category_Id','Subscribers','Total_Videos']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Feature Engineering
    data['Video_Age'] = data['Upload'].apply(calculate_video_age)

    data['Upload'] = pd.to_datetime(data['Upload'])
    data['Upload_Day_Encoded'] = data['Upload'].dt.dayofweek
    data['Upload_Time_Category_Encoded'] = data['Upload'].dt.hour // 6

    data['Duration'] = data['Duration'].apply(convert_duration)
    data['Duration'] = data['Duration'].apply(duration_to_seconds)

    data['isLive'] = data['isLive'].astype(int)
    data['Title_Length'] = data['Title'].str.len()
    data['Tags_Count'] = data['Tags'].apply(count_tags)
    data['isAgency'] = data['Agency'].apply(lambda x: 0 if x == 'Indie' else 1)

    # One-hot encoding
    data = pd.get_dummies(data, columns=['Category_Id'], prefix='Category')
    data = pd.get_dummies(data, columns=['Agency'], prefix='Agency')

    # Drop columns
    data = data.drop([
        'ChannelName','Upload','Total_Videos','Description',
        'Video_Id','Title','Tags'
    ], axis=1, errors='ignore')

    # Drop training-only features
    data = data.drop(['Likes','Views','Comments'], axis=1, errors='ignore')

    return data

st.title("Prediksi Engagement VTuber")

# Load model
url = "https://drive.google.com/file/d/1RdkAPg0RBf547hEHe-sfq462b5CvsSZu/view?usp=sharing"
output = "model.pkl"

gdown.download(url, output, quiet=False)

model = joblib.load("model.pkl")

#load columns
columns = joblib.load("columns.pkl")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = preprocess(data)
    data = data.reindex(columns=columns, fill_value=0)

    st.write("Data kamu:")
    st.dataframe(data.head())

    if st.button("Predict"):
        pred = model.predict(data)
        data['Predicted_Engagement'] = pred

        st.success("Prediction Completed!")
        st.dataframe(data[['Predicted_Engagement']].head(20))

        st.write("Jumlah Tiap Kategori:")
        st.write(data['Predicted_Engagement'].value_counts())

        csv = data.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download hasil prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )