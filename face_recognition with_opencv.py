import face_recognition  # Yüz tanıma işlemleri için gerekli kütüphane
import cv2 as cv  # Kameradan görüntü almak için OpenCV
import numpy as np  # Matematiksel işlemler için numpy

#  1. TANINAN YÜZLERİ KAYDETME
bilinen_yuzler = ["emir.jpg","faruk.jpg", "ibo.jpg", "yunus.jpg"]
bilinen_isimler = ["emir", "faruk", "ibo", "yunus"]

# Yüzleri ve encoding (vektörlerini) saklamak için liste oluşturuyoruz
bilinen_yuz_kodlari = []
#bu kod yeni yuzler vektorlerini kelek için

for i in bilinen_yuzler:
    pic = face_recognition.load_image_file(i)  # Resmi yükle(okuma islemş)
    kodlamalar = face_recognition.face_encodings(pic)  # Encoding al ---vektore mi ne ceviiriyot

    bilinen_yuz_kodlari.append(kodlamalar[0])#

kamera = cv.VideoCapture(0)  # Kamerayı aç

while True:
    basarili, kare = kamera.read()
    if not basarili:
        break  # Kamera görüntü alamazsa döngüden çık

    rbg_kare = cv.cvtColor(kare, cv.COLOR_BGR2RGB)  # BGR -> RGB dönüşümü numpy için

    yuz_konumlari = face_recognition.face_locations(rbg_kare)  # Yüz konumlarını bul
    yuz_kodlari = face_recognition.face_encodings(rbg_kare, yuz_konumlari)  # Encoding hesapla----vektore donustur
    #endonin denilen
    # 📌 Görünen her yüz için işlem yap
    for (ust, sag, alt, sol), yuz_kodlama in zip(yuz_konumlari, yuz_kodlari):#cizim yapmak için kullanilir
        eslesmeler = face_recognition.compare_faces(bilinen_yuz_kodlari, yuz_kodlama)
        yuz_mesafesi = face_recognition.face_distance(bilinen_yuz_kodlari, yuz_kodlama)

        isim = "Bilinmeyen kişi"  # Varsayılan isim

        if True in eslesmeler:
            en_iyi_eslesme = np.argmin(yuz_mesafesi)  # En iyi eşleşmeyi bul
            isim = bilinen_isimler[en_iyi_eslesme]  # İsmi ata

        # 📌 Yüzün etrafına dikdörtgen çiz ve ismi yazdır
        cv.rectangle(kare, (sol, ust), (sag, alt), (0, 255, 0), 2)  # Yeşil çerçeve
        cv.putText(kare, isim, (sol, ust - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv.imshow("Kamera", kare)  # Kamerayı göster

    if cv.waitKey(20) == 27:  # ESC'ye basılırsa çık
        break
        
kamera.release()#kamereayi kapatma işlemi için gerelili
cv.destroyAllWindows()
import os

print(os.path.abspath(__file__))
