import streamlit as st
import csv
import folium
import jinja2
import numpy as np
from math import *
import pandas as pd
#menu sidebar
from streamlit_option_menu import option_menu
# import library LSTM
from tensorflow.keras.models import load_model
#import minmax
from sklearn.preprocessing import MinMaxScaler
#getting potential rain
from bs4 import BeautifulSoup
from urllib.request import urlopen
#pembagian data
from sklearn.model_selection import train_test_split
#folium
from streamlit_folium import st_folium
from folium.features import DivIcon

startLatitude = float(0)
startLongitude = float(0)
# Initialize session state to store the locations
if 'destination' not in st.session_state:
    st.session_state['destination'] = []

if 'coordinates' not in st.session_state:
    st.session_state['coordinates'] = []

if 'distance' not in st.session_state:
    st.session_state['distance'] = 0

customerfile = 'dataset/customers-all.csv';

@st.cache_data
def read_data() :
    data = []
    with open(customerfile, 'r') as csvfile : 
        reader = csv.DictReader(csvfile)
        for row in reader :
            data.append({
                'customer_name' : row['customer_name'],
                'latitude' : float(row['latitude']),
                'longitude' : float(row['longitude'])
            })
    return data

data = read_data()

CONNECTION_CENTER = (-6.6061381, 106.801851)

def get_lat_lng(lat, lng) :
    return float(lat), float(lng)

with st.sidebar:
    selected = option_menu("Main Menu", ["Prediction", 'Visualization'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)

if selected == "Visualization" : 
    st.title("Visualisasi Model")
    option = st.selectbox(
        "Pilih tampilan yang kamu inginkan?",
        ("'Dataset'", "'Korelasi Variabel'", "'Grafik Prediksi'", "'Insight Untuk Perusahaan'"),
        index=None,
        placeholder="Pilih Tampilan...",
    )

    st.write("Mode Tampilan:", option)
    if option == "'Dataset'" :
        st.markdown('<div style="text-align: justify;">Faktor yang akan dibentuk dalam dataset ini diantaranya jarak titik awal ke titik tujuan X1-(distance), kecepatan rata-rata kendaraan yang dicapai oleh kurir X2-(speed_average) dan terakhir kondisi cuaca terutama dalam hal ini adalah hujan X3-(potential_rain). Dataset history pengiriman Radar Bogor menyediakan 3 variabel yaitu distance, speed_average, dan duration. Variable potential_rain didapat dari sumber open weather yang dicleaning sebelum digabungkan dengan dataset history pengiriman.</div>', unsafe_allow_html=True)
        left_co, cent_co,last_co = st.columns(3)
        with cent_co: 
            st.image('image/dataset-building.png', width=300, caption='Proses Pembentukan Dataset')
        
        st.markdown('<div style="text-align: justify;">Dataset dari history pengiriman dan open weather akan digabungkan sehingga akan memudahkan dalam proses analisis, pengambilan knowledge dan pembangunan model dan untuk karakteristik hasil penggabungan menjadi dataset seperti berikut.</div>', unsafe_allow_html=True)
        st.image('image/karakteristik-dataset.png', width=700, caption='Proses Pembentukan Dataset')
    elif option == "'Korelasi Variabel'" :
        st.image('image/tabel-korelasi.png', width=700, caption='Tabel Korelasi Variabel')
        st.markdown('<div style="text-align: justify;">Nilai korelasi yang menjauh dari angka 1 bahkan terkadang sampai negatif menandakan variable tersebut tidak berkaitan. Variabel duration dan distance memiliki nilai 0.874523 dimana nilai mendekati 1.00000 yang artinya variabel tersebut sangat terhubung.</div>', unsafe_allow_html=True)
    
        st.image('image/grafik-korelasi.png', width=700)
        st.markdown('<div style="text-align: justify;">Tampak ada hubungan positif yang jelas di mana durasi meningkat seiring dengan peningkatan jarak. Ini menunjukkan bahwa perjalanan yang lebih panjang cenderung memakan waktu lebih lama. Korelasi yang positif juga antara duration dan potential rain 0.202287, maka artinya durasi suatu kejadian (misalnya perjalanan) cenderung meningkat seiring dengan peningkatan potensi hujan. </div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-top:10px;">Kondisi cuaca yang lebih buruk (hujan) cenderung menyebabkan perjalanan yang lebih lama. Ini bisa disebabkan oleh berbagai faktor seperti jalan licin, kurir mengamankan surat kabar, kecepatan kendaraan yang lebih lambat, atau lalu lintas yang lebih padat.</div>', unsafe_allow_html=True)
    elif option == "'Grafik Prediksi'" :
        st.markdown('<div style="text-align: justify;">Model yang sudah diinisialisasi dan diujikan perlu dilihat nilai loss, nilai ini akan ditampilkan dalam bentuk grafik menggunakan fungsi loss dari MSE dengan metric default MAE.</div>', unsafe_allow_html=True)
        st.image('image/loss-mse.png', width=700)
        st.markdown('<div style="text-align: justify;">Grafik tersebut menunjukan nilai loss  menghasilkan model yang sangat baik karena nilai validation loss bergerak searah dengan nilai training dan mulai bergerak searah diantara nilai epoch 5 sampai dengan 60. Selanjutnya Metris MSE.</div>', unsafe_allow_html=True)
        st.image('image/metric-mse.png', width=700)
        st.markdown('<div style="text-align: justify;">Grafik MSE juga menunjukan pergerakan yang searah juga mengikuti nilai trainingnya. Pergerakan searah ini dinamakan sebagai kondisi konvergen, sebaliknya jika grafik menunjukan kondisi berlawanan dan tidak searah makan dinamakan divergen.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify;">Perbandingan antara data aktual dan data prediksi dari proses pengolahan model menggunakan data training dan data testing akan ditunjukan.</div>', unsafe_allow_html=True)
        
        st.image('image/aktual-prediksi-train.png', width=700)
        st.markdown('<div style="text-align: justify;">Grafik tersebut menunjukan perbandingan data aktual dan prediksi pada proses training yang menunjukan hasil cukup baik. Jarak eror dari yang ditunjukan tidak terlalu jauh dengan menghasilkan nilai MAPE 0,07. </div>', unsafe_allow_html=True)
    
        st.image('image/aktual-prediksi-test.png', width=700)
        st.markdown('<div style="text-align: justify;">Grafik tersebut juga menunjukan bahwa data prediksi tidak begitu jauh dengan data aktualnya serta pada grafiknya menunjukan pergerakan yang searah dengan nilai MAPE 0,07.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; font-weight: bold;">Nilai MAPE tersebut dapat dikategorikan sangat baik karena berada pada range <10% MAPE.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-top: 15px; font-weight: bold;">Catatan penting : Forcasting atau peramalan tidak harus selalu 100% akurat, karena kesalahan merupakan salah satu dari sifat melakukan peramalan, tetapi tidak mengurangi manfaat dan fungsi yang dapat dilakukan oleh kegiatan ini.</div>', unsafe_allow_html=True)
    elif option == "'Insight Untuk Perusahaan'" :
        st.markdown('<div style="text-align: justify;">Berdasarkan hasil evaluasi model LSTM yang telah dilakukan, berikut adalah beberapa insight dan rekomendasi yang dapat perusahaan pertimbangkan untuk meningkatkan kecepatan pengiriman. Berikut insight dari penulis:</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">1.	Perjalanan pengiriman terlama memakan waktu hingga 27 menit, dan  pengiriman tercepat 2 menit.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">2.	Jarak pengiriman terjauh 5600 meter dan jarak pengiriman terdekat 50 meter.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">3.	Kecepatan rata-rata yang bisa ditempuh oleh kurir dalam melakukan pengiriman berada dirange 30-50 km/jam.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">4.	Potensi hujan dapat mempengaruhi durasi perjalanan dengan kemungkinan besar karena faktor seperti jalan licin, kurir mengamankan surat kabar, kecepatan kendaraan yang lebih lambat, atau lalu lintas yang lebih padat.</div>', unsafe_allow_html=True)
        # st.markdown('<div style="text-align: justify; margin-left: 30px;">5.	Beberapa record dalam dataset menunjukkan durasi pengiriman bernilai 0, kemungkinan disebabkan oleh gangguan koneksi internet saat kurir melakukan stamp, sehingga data yang terekam tidak lengkap.</div>', unsafe_allow_html=True)

        st.markdown('<div style="text-align: justify; margin-top: 20px;">Dari beberapa insight diatas, maka rekomendasi untuk perusahaan sebagai berikut :</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">1.	Melakukan evaluasi agar jarak antar titik pengiriman tidak terlalu jauh sehingga kurir dapat meminimalkan durasi pengiriman sehingga kecepatan pengiriman menjadi lebih baik.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">2.	Membuat standar maksimal jarak pengiriman antar titik sehingga jarak tidak terlalu jauh.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">3.	Membekali kurir jas hujan serta pelindung surat kabar anti air, agar ketika terjadi hujan kurir dapat langsung meneruskan pengiriman dan meminimalisir surat kabar rusak terkena air.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">4.	Memastikan durasi pengiriman terekap secara keseluruhan agar ketika dataset diperlukan kolomnya terisi secara lengkap.</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align: justify; margin-top: 20px;">Saran untuk model yang saat ini dibuat : </div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">1.	Melakukan perbandingan dengan menggunakan algoritma prediksi lainnya.</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">2.	Membentuk model yang lebih baik lagi dengan mengkaji parameter yang digunakan</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">3.	Menambahkan fitur baru seperti data kemacetan dari data lalu lintas </div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify; margin-left: 30px;">4.	Pengembangan model prediksi ini hingga bisa digunakan secara realtime dan menjadi sistem rekomendasi</div>', unsafe_allow_html=True)
    
        st.markdown('<div style="text-align: justify; margin-top: 20px; font-weight: bold;">Kesimpulan : <br>Dengan mengikuti rekomendasi di atas, perusahaan dapat meningkatkan efisiensi dan kecepatan pengiriman, meningkatkan kepuasan pelanggan, dan operasional. Optimalisasi rute, fasilitas kurir yang tepat, dan data akurat akan meminimalkan penundaan dan meningkatkan kinerja layanan pengiriman.</div>', unsafe_allow_html=True)


elif selected == "Prediction" :
    url = "https://weather.com/id-ID/weather/hourbyhour/l/649d5fb3e40a54653934eae4ec15137dc83b1a1d6f5a4fc2accbb73a0a7e0e39"
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    df=pd.read_csv("https://raw.githubusercontent.com/kritimauludin/prediction-speed-courir/main/dataset/user-distribution-all.csv", parse_dates=["created_at"],index_col=[0])
    df = df.drop(["distribution_code", "customer_code", "total", "process_at", "received_at", "status"], axis='columns')
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
    if predictRain < 30 :
        predictRain = 30
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
        r = 6371  # Radius in m use 6371. Use 3956 for miles
        return c * r

    def PredictionSpeedCourier(Distance, SpeedAverage, PotentialRain):
        Distance = int(Distance)
        SpeedAverage = int(SpeedAverage)
        PotentialRain = int(PotentialRain)

        x = np.array([[Distance, SpeedAverage, PotentialRain]])
        x = scalerX.transform(x)


        predictData = np.expand_dims(x, axis=1)

        hasil = model.predict(predictData, verbose=3)

        duration=scalerY.inverse_transform(hasil)

        return duration

    #map area
    st.write('Klik untuk pilih destinasi. Start point Gedung Graha pena Radar Bogor!')
    mapfolium = folium.Map(location=CONNECTION_CENTER, zoom_start=10)
    mapplot = folium.Map(location=CONNECTION_CENTER, zoom_start=10)
    #start-point Radar Bogor
    startLatitude =  float('-6.556787')
    startLongitude = float('106.773193')

    for customer in data :
        location = customer['latitude'], customer['longitude']
        folium.Marker(
            location, 
            popup=customer['customer_name']
        ).add_to(mapfolium)

    
    stmap = st_folium(mapfolium, width=700, height=500)
    
    # Function to add a click to the session state and keep only the last three clicks
    def add_destination(click):
        coordinate = (click[1], click[2])
        if click[0] not in [c[0] for c in st.session_state['destination']]:
            st.session_state['destination'].append(click)
            st.session_state['coordinates'].append(coordinate)

            if len(st.session_state['destination']) >= 1:
                st.session_state['distance']+= int(click[3])

    if stmap['last_object_clicked'] is not None:
        length = len(st.session_state['destination'])

        latBefore = st.session_state['destination'][length-1][1]
        lngBefore = st.session_state['destination'][length-1][2]

        latStore = stmap['last_object_clicked']['lat']
        lngStore = stmap['last_object_clicked']['lng']

        distance = haversine(float(lngBefore), float(latBefore), float(lngStore), float(latStore)) * 1000 #ubah ke satuan meter
        click = (
            stmap['last_object_clicked_popup'],
            latStore, 
            lngStore,
            distance,
        )
        add_destination(click)
    elif len(st.session_state['destination']) == 0:
        #start-point 
        click = (
            "Radar Bogor - startpoint",
            startLatitude, 
            startLongitude,
            "Start Point"
        )
        add_destination(click)

    #pembagian kolom
    col1, col2 = st.columns(2)
    # Display the destination clicked locations
    if st.session_state['destination']:
        with col1 : 
            st.write(st.session_state['destination'][0][0], ' - Start Point')
            st.write("Destinasi :")
        with col2 :
            st.text("")
            st.text("")
            st.text("")

        for idx, loc in enumerate(st.session_state['destination'][1:]):
            if idx < 6:
                with col1 : 
                    st.write(f"{idx + 1}: {loc[0]} - {int(loc[3])} m")
            elif idx >= 6 :
                with col2 : 
                    st.write(f"{idx + 1}: {loc[0]} - {int(loc[3])} m")

    if len(st.session_state['destination']) > 1:
        for customer in st.session_state['destination'] :
            location = customer[1], customer[2]
            folium.Marker(
                location, 
            ).add_to(mapplot)
        folium.PolyLine(st.session_state['coordinates'], color="blue", weight=2.5, opacity=1).add_to(mapplot)
        st_folium(mapplot, width=700, height=500)

    #pembagian kolom
    col1, col2 = st.columns(2)
    distance = st.session_state['distance']


    with col1 : 
        distance = st.number_input("Jarak - m (auto)", value=int(distance), disabled=True)
    with col2 : 
        speedAverage = st.number_input("Laju Rata-rata 30-50 : ", step=1, min_value=30, max_value=50)

    with col1 : 
        potentialRain = st.number_input("Potential Rain Today (auto)%", int(predictRain), disabled=True)
    with col2 : 
        st.text("")
        st.info("Potensi cuaca aktual 1 jam kedepan")

    with col1 : 
        if st.button("Prediksi Sekarang") :
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
    - Jarak Total: """+ str(int(distance)) +"""m\n
    - Laju rata-rata kurir : """+ str(speedAverage) +"""km/h\n
    - Potensi hujan : """+str(potentialRain)+"""%
    """

    output = """
    Prediksi Kecepatan Pengiriman Kurir :\n
    Dengan """+str(len(st.session_state['destination'])-1)+""" titik,\n 
    Total Kecepatan **"""+str("%.1f" % courierSpeedPrediction)+ """** menit
    """
    with col1 : 
        st.warning(message)
    with col2 : 
        st.success(output)