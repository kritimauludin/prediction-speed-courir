import streamlit as st
from math import *

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
    st.number_input("Jarak - m (auto)", int(distance),)
with col2 : 
    speedAverage = st.number_input("Laju Rata-rata 30-50 : ", step=1)

with col1 : 
    potentialRain = st.number_input("Potential Rain Today (auto)", 50)
with col2 : 
    st.text("")
    st.info("Potensi cuaca aktual selama 3 jam")

st.button("Prediksi Sekarang")

courierSpeedPrediction = 10

st.markdown("----------------")
st.success("Dengan jarak "+ str(int(distance)) +"m  dan Laju rata-rata kurir "+ str(speedAverage) +"km/h serta potensi hujan "+str(potentialRain)+"% maka prediksi kecepatan pengiriman kurir "+str(courierSpeedPrediction)+ " menit")