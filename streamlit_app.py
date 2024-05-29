import streamlit as st
import numpy as np
from math import *
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
import pandas as pd

with st.sidebar:
    selected = option_menu("Main Menu", ["Prediction", 'Visualization'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)

if selected == "Prediction" :
    url = "https://weather.com/id-ID/weather/hourbyhour/l/649d5fb3e40a54653934eae4ec15137dc83b1a1d6f5a4fc2accbb73a0a7e0e39"
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    df=pd.read_csv("https://raw.githubusercontent.com/kritimauludin/prediction-speed-courir/main/dataset/user-distribution-all.csv", parse_dates=["created_at"],index_col=[0])
    df = df.drop(["distribution_code", "customer_code", "total", "process_at", "received_at", "courier_last_stamp", "status", "updated_at"], axis='columns')
    df = df.drop(["start_latitude", "start_longitude", "dest_latitude", "dest_longitude", "created_at"], axis='columns')

    model = load_model('ModelPredictionSpeedCourierKeras.h5')

    X = df.drop(columns='duration')
    y = df.duration

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)# Menentukan banyaknya data test yaitu sebesar 20% data


    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()

    X_train = scalerX.fit_transform(X_train)
    X_test = scalerX.transform(X_test)


    y1 = np.array(y_train)
    y2 = np.array(y_test)

    y_train = y1[:, np.newaxis]
    y_test = y2[:, np.newaxis]

    y_train = scalerY.fit_transform(y_train)
    y_test = scalerY.transform(y_test)

    courierSpeedPrediction = 0
    predictRain = soup.find('span', attrs={'data-testid':'PercentageValue'}).text
    predictRain = float(predictRain.replace('%', ''))

    st.title("Prediksi Kecepatan Kurir")


    def haversine(long1, lat1, long2, lat2):
        # convert decimal to radians
        long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])

        # haversine formula
        dlon = long2 - long1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius in km use 6371. Use 3956 for miles
        return c * r

    def PredictionSpeedCourier(Distance, SpeedAverage, PotentialRain):
        Distance = int(Distance)
        SpeedAverage = int(SpeedAverage)
        PotentialRain = int(PotentialRain)

        x = np.array([[Distance, SpeedAverage, PotentialRain]])
        x = scalerX.transform(x)


        predictData = np.expand_dims(x, axis=1)

        hasil = model.predict(predictData)

        duration=scalerY.inverse_transform(hasil)

        return duration

    #pembagian kolom
    col1, col2 = st.columns(2)

    with col1 : 
        startLatitude = st.number_input(
                            "Start Latitude",
                            step=1e-6,
                            format="%.6f")
    with col2 : 
        startLongitude = st.number_input(
                            "Start Longitude",
                            step=1e-6,
                            format="%.6f")

    with col1 : 
        destLatitude = st.number_input(
                            "Destination Latitude",
                            step=1e-6,
                            format="%.6f")
    with col2 : 
        destLongitude = st.number_input(
                            "Destination Longitude",
                            step=1e-6,
                            format="%.6f")

    distance = haversine(float(startLongitude), float(startLatitude), float(destLongitude), float(destLatitude)) * 1000 #ubah ke satuan meter

    with col1 : 
        st.number_input("Jarak - m (auto)", int(distance), disabled=True)
    with col2 : 
        speedAverage = st.number_input("Laju Rata-rata 30-50 : ", step=1, min_value=30, max_value=50)

    with col1 : 
        potentialRain = st.number_input("Potential Rain Today (auto)%", int(predictRain), disabled=True)
    with col2 : 
        st.text("")
        st.info("Potensi cuaca aktual 1 jam kedepan")

    with col1 : 
        if st.button("Prediksi Sekarang") :
            # Prepare the input data
            #input_data = np.array([[distance, speedAverage, potentialRain]])
            #input_data_reshaped = input_data.reshape((1, 1, 3))  # (samples, timesteps, features)

            courierSpeedPrediction = PredictionSpeedCourier(distance, speedAverage, potentialRain)
    with col2 :
        st.text("")
        st.text("")
        st.text("")
        
    with col1:
        st.markdown("----------------")
    with col2:
        st.markdown("----------------")

    message = """
    Variabel :\n
    - Jarak : """+ str(int(distance)) +"""m\n
    - Laju rata-rata kurir : """+ str(speedAverage) +"""km/h\n
    - Potensi hujan : """+str(potentialRain)+"""%
    """

    output = """
    Hasil Prediksi Kecepatan Pengiriman Kurir :\n
    **"""+str("%.1f" % courierSpeedPrediction)+ """** menit
    """
    with col1 : 
        st.warning(message)
    with col2 : 
        st.success(output)

elif selected == "Visualization" :
    st.title("Visualisasi Dataset")