import streamlit as st
import numpy as np
from math import *
from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "https://weather.com/id-ID/weather/hourbyhour/l/649d5fb3e40a54653934eae4ec15137dc83b1a1d6f5a4fc2accbb73a0a7e0e39"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
soup = BeautifulSoup(html, "html.parser")


model = load_model('ModelPredictionSpeedCourierKeras.h5')

courierSpeedPrediction = 0
predictRain = soup.find('span', attrs={'data-testid':'PercentageValue'}).text
predictRain = float(predictRain.replace('%', ''))

st.title("Courier Speed Prediction")


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
        input_data = np.array([[distance, speedAverage, potentialRain]])
        input_data_reshaped = input_data.reshape((1, 1, 3))  # (samples, timesteps, features)

        courierSpeedPrediction = model.predict(input_data_reshaped)
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