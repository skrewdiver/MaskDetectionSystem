import csv
import json
import os
import platform
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime

import config
from scrfd import SCRFD         

onnxruntime.set_default_logger_severity(3)  

PATUH = "PATUH"
JENIS_TIDAK_SESUAI = "JENIS TIDAK SESUAI"
SALAH_PAKAI = "SALAH PAKAI"
TANPA_MASKER = "TANPA MASKER"
BELUM_PASTI = "BELUM PASTI"
PELANGGARAN = [SALAH_PAKAI, TANPA_MASKER, JENIS_TIDAK_SESUAI]


WARNA = {PATUH: (0, 200, 0), JENIS_TIDAK_SESUAI: (0, 215, 255), SALAH_PAKAI: (0, 0, 230),
         TANPA_MASKER: (0, 0, 230), BELUM_PASTI: (160, 160, 160)}
NAMA_JENIS = {"Surgical": "Medis", "N95": "N95", "Cotton": "Kain", "Bandana": "Bandana"}
HURUF = cv2.FONT_HERSHEY_SIMPLEX


# 1. MEMUAT MODEL
def muat_detektor():
    detektor = SCRFD(model_file=config.FILE_DETEKTOR)
    detektor.prepare(-1)                          
    return detektor


def muat_model_masker():
    with open(config.FILE_LABEL, encoding="utf-8") as f:
        info = json.load(f)
    model = {"kelas": info["classes"], "ukuran": info["input_size"]}
    if os.path.exists(config.FILE_MODEL_ONNX):
        model["sesi"] = onnxruntime.InferenceSession(config.FILE_MODEL_ONNX, providers=["CPUExecutionProvider"])
        model["nama_input"] = model["sesi"].get_inputs()[0].name
        model["jenis"] = "onnx"
    elif os.path.exists(config.FILE_MODEL_KERAS):
        import tensorflow as tf
        model["keras"] = tf.keras.models.load_model(config.FILE_MODEL_KERAS, compile=False)
        model["jenis"] = "keras"
    else:
        raise FileNotFoundError("Model tidak ditemukan di folder model/. Latih dulu: python train.py")
    return model


# 2. DETEKSI WAJAH & KLASIFIKASI MASKER
def cari_wajah(detektor, frame):
    ukuran = (config.UKURAN_DETEKSI, config.UKURAN_DETEKSI)
    hasil, _ = detektor.detect(frame, thresh=0.5, input_size=ukuran)
    kotak = []
    for x1, y1, x2, y2, skor in hasil:         
        w, h = int(x2 - x1), int(y2 - y1)
        if w >= 20 and h >= 20:                  
            kotak.append((max(0, int(x1)), max(0, int(y1)), w, h))
    return kotak


def potong_wajah(frame, x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2 + int(h * 0.05)             
    x0, y0 = cx - sisi // 2, cy - sisi // 2
    x1, y1 = x0 + sisi, y0 + sisi
    potongan = frame[max(0, y0):y1, max(0, x0):x1]
    
    atas, kiri = max(0, -y0), max(0, -x0)
    bawah, kanan = max(0, y1 - frame.shape[0]), max(0, x1 - frame.shape[1])
    return cv2.copyMakeBorder(potongan, atas, bawah, kiri, kanan, cv2.BORDER_REPLICATE)


def tebak_masker(model, wajah):
    rgb = cv2.cvtColor(wajah, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (model["ukuran"], model["ukuran"]), interpolation=cv2.INTER_AREA)
    data = rgb.astype(np.float32)[np.newaxis]
    if model["jenis"] == "onnx":
        peluang = model["sesi"].run(None, {model["nama_input"]: data})[0][0]
    else:
        peluang = np.asarray(model["keras"](data, training=False))[0]
    return {kelas: float(p) for kelas, p in zip(model["kelas"], peluang)}

# 3. ATURAN KEPATUHAN & TIMER PERINGATAN

def status_kelas(kelas):
    if kelas == "No_Mask":
        return TANPA_MASKER
    if kelas.startswith("Incorrect"):            
        return SALAH_PAKAI
    jenis = kelas.split("_", 1)[1]                
    return PATUH if jenis in config.JENIS_DIIZINKAN else JENIS_TIDAK_SESUAI


def tentukan_status(peluang):
    skor = {PATUH: 0.0, JENIS_TIDAK_SESUAI: 0.0, SALAH_PAKAI: 0.0, TANPA_MASKER: 0.0}
    for kelas, p in peluang.items():
        skor[status_kelas(kelas)] += p
    status = max(skor, key=skor.get)
    yakin = skor[status]
    if yakin < config.KEYAKINAN_MIN:
        return BELUM_PASTI, "-", yakin
    if status in (SALAH_PAKAI, TANPA_MASKER):
        return status, "-", yakin                  
    kandidat = [k for k in peluang if status_kelas(k) == status]
    jenis = max(kandidat, key=peluang.get).split("_", 1)[1]
    return status, jenis, yakin


def buat_timer():
    return {"mulai": {}, "terakhir_terlihat": {}, "terakhir_bunyi": {}}


def perbarui_timer(timer, status_terlihat, sekarang):
    perlu_bunyi = []
    for status in PELANGGARAN:
        if status in status_terlihat:
            timer["terakhir_terlihat"][status] = sekarang
            timer["mulai"].setdefault(status, sekarang)
            lama = sekarang - timer["mulai"][status]
            sudah_jeda = sekarang - timer["terakhir_bunyi"].get(status, -1e9) >= config.JEDA_PERINGATAN
            if lama >= config.BATAS_WAKTU and sudah_jeda:
                timer["terakhir_bunyi"][status] = sekarang
                perlu_bunyi.append(status)
        elif status in timer["mulai"] and sekarang - timer["terakhir_terlihat"][status] > config.TOLERANSI:
            del timer["mulai"][status]           
    return perlu_bunyi


# 4. SUARA PERINGATAN (MP3)
def cari_file_suara(daftar_nama):
    if not os.path.isdir(config.FOLDER_SUARA):
        return None
    for nama_file in sorted(os.listdir(config.FOLDER_SUARA)):
        dasar, ext = os.path.splitext(nama_file.lower())
        if ext not in (".mp3", ".wav", ".ogg"):
            continue
        while os.path.splitext(dasar)[1] in (".mp3", ".wav", ".ogg"):
            dasar = os.path.splitext(dasar)[0]
        if dasar.strip() in daftar_nama:
            return os.path.join(config.FOLDER_SUARA, nama_file)
    return None


def siapkan_suara():
    suara = {"pemutar": None, "file": {}, "antrean": []}
    for status, nama in config.NAMA_SUARA.items():
        suara["file"][status] = cari_file_suara(nama)
    try:
        import pygame
        pygame.mixer.init()
        suara["pygame"] = pygame
        suara["pemutar"] = "pygame"
    except Exception as e:
        print(f"[SUARA] pygame tidak bisa dipakai ({e})")
        if platform.system() == "Windows":
            suara["pemutar"] = "windows"
    print(f"[SUARA] pemutar: {suara['pemutar'] or 'TIDAK ADA (pip install pygame-ce)'}")
    for status, file in suara["file"].items():
        print(f"[SUARA] {status:<19}: {os.path.basename(file) if file else 'FILE TIDAK DITEMUKAN'}")
    return suara


def perintah_windows(perintah):
    import ctypes
    jawaban = ctypes.create_unicode_buffer(128)
    kode = ctypes.windll.winmm.mciSendStringW(perintah, jawaban, 127, 0)
    return jawaban.value if kode == 0 else None


def sedang_berbunyi(suara):
    if suara["pemutar"] == "pygame":
        return suara["pygame"].mixer.music.get_busy()
    if suara["pemutar"] == "windows":
        return (perintah_windows("status alarm mode") or "").lower() == "playing"
    return False


def mainkan_file(suara, file):
    if suara["pemutar"] == "pygame":
        suara["pygame"].mixer.music.load(file)
        suara["pygame"].mixer.music.play()
    else:
        perintah_windows("close alarm")
        perintah_windows(f'open "{os.path.abspath(file)}" type mpegvideo alias alarm')
        perintah_windows("play alarm")
    return f"memutar {os.path.basename(file)}"


def putar_suara(suara, status):
    file = suara["file"].get(status)
    if file is None:
        return "file suara tidak ditemukan"
    if suara["pemutar"] is None:
        return "tidak ada pemutar suara (pip install pygame-ce)"
    if sedang_berbunyi(suara):
        if file not in suara["antrean"]:
            suara["antrean"].append(file)
        return f"menunggu giliran ({os.path.basename(file)})"
    try:
        return mainkan_file(suara, file)
    except Exception as e:
        return f"gagal memutar: {e}"


def perbarui_antrean_suara(suara):
    if suara["antrean"] and not sedang_berbunyi(suara):
        try:
            mainkan_file(suara, suara["antrean"].pop(0))
        except Exception as e:
            print(f"[SUARA] gagal memutar: {e}")



# 5. PENCATATAN LOG
def buat_log():
    return {"mulai": datetime.now(), "selesai": None, "catatan": [],
            "jumlah": {s: 0 for s in PELANGGARAN}}


def catat_peringatan(log, status, jumlah_orang):
    log["jumlah"][status] += 1
    log["catatan"].append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, jumlah_orang])


def durasi_sesi(log):
    akhir = log["selesai"] or datetime.now()
    detik = int((akhir - log["mulai"]).total_seconds())
    return f"{detik // 3600:02d}:{detik % 3600 // 60:02d}:{detik % 60:02d}"


def ringkasan_sesi(log):
    selesai = log["selesai"] or datetime.now()
    baris = [f"Mulai   : {log['mulai']:%d-%m-%Y %H:%M:%S}",
             f"Selesai : {selesai:%d-%m-%Y %H:%M:%S}",
             f"Durasi  : {durasi_sesi(log)}", ""]
    for status, jumlah in log["jumlah"].items():
        baris.append(f"{status}: {jumlah} peringatan")
    baris.append(f"TOTAL: {sum(log['jumlah'].values())} peringatan")
    return baris


def tanya_lokasi_simpan(nama_bawaan):
    path = ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        jendela = tk.Tk()
        jendela.withdraw()                         
        jendela.attributes("-topmost", True)      
        path = filedialog.asksaveasfilename(title="Simpan log sesi", initialdir=config.FOLDER_LOG,
                                            initialfile=nama_bawaan, defaultextension=".csv",
                                            filetypes=[("File CSV", "*.csv")])
        jendela.destroy()
    except Exception as e:
        print(f"[LOG] jendela simpan tidak bisa dibuka ({e})")
    if not path:
        path = os.path.join(config.FOLDER_LOG, nama_bawaan)
        print("[LOG] lokasi tidak dipilih -> disimpan di folder log/")
    return path


def simpan_log(log, path_csv):
    os.makedirs(os.path.dirname(path_csv) or ".", exist_ok=True)
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        penulis = csv.writer(f)
        penulis.writerow(["waktu", "pelanggaran", "jumlah_orang_melanggar"])
        penulis.writerows(log["catatan"])
    path_txt = os.path.splitext(path_csv)[0] + "_ringkasan.txt"
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("RINGKASAN SESI\n" + "\n".join(ringkasan_sesi(log)) + "\n")
    return path_csv, path_txt

# 6. TAMPILAN
JUDUL_JENDELA = "Deteksi Masker"


def gambar_wajah(frame, kotak, status, jenis, yakin):
    x, y, w, h = kotak
    warna = WARNA[status]
    cv2.rectangle(frame, (x, y), (x + w, y + h), warna, 3)
    if jenis == "-":                              # salah pakai / tanpa masker / belum pasti
        teks = f"{status} {yakin * 100:.0f}%"
    else:                                         # dipakai benar -> tampilkan jenisnya
        teks = f"{status} | {NAMA_JENIS.get(jenis, jenis)} {yakin * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(teks, HURUF, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), warna, -1)
    cv2.putText(frame, teks, (x + 4, y - 6), HURUF, 0.7, (0, 0, 0), 2)


def gambar_info(frame, hitung, fps, log):
    """Baris atas: jumlah orang & FPS. Baris bawah: tanda SIAP atau MEMANTAU."""
    tinggi, lebar = frame.shape[:2]
    melanggar = sum(hitung.get(s, 0) for s in PELANGGARAN)
    teks = f"Orang: {sum(hitung.values())}  Patuh: {hitung.get(PATUH, 0)}  Melanggar: {melanggar}  |  FPS: {fps:.1f}"
    cv2.rectangle(frame, (0, 0), (lebar, 50), (40, 40, 40), -1)
    cv2.putText(frame, teks, (15, 35), HURUF, 1.0, (255, 255, 255), 2)

    if log is None:
        teks, warna = "SIAP  -  tekan SPASI untuk MULAI", (0, 215, 255)
    else:
        teks, warna = f"MEMANTAU  {durasi_sesi(log)}  -  tekan Q untuk SELESAI", (0, 0, 255)
    cv2.rectangle(frame, (0, tinggi - 50), (lebar, tinggi), (40, 40, 40), -1)
    cv2.circle(frame, (30, tinggi - 25), 12, warna, -1)          # lampu tanda
    cv2.putText(frame, teks, (55, tinggi - 14), HURUF, 1.0, warna, 2)


def gambar_tulisan_mulai(frame):
    tinggi, lebar = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize("MULAI", HURUF, 3, 8)
    x, y = (lebar - tw) // 2, (tinggi + th) // 2
    cv2.rectangle(frame, (x - 30, y - th - 30), (x + tw + 30, y + 30), (0, 0, 0), -1)
    cv2.putText(frame, "MULAI", (x, y), HURUF, 3, (0, 200, 0), 8)


def layar_ringkasan(baris_ringkasan, baris_bawah, tunggu=False):
    """Layar hitam berisi ringkasan sesi. Jika tunggu=True, tunggu tombol apa saja."""
    layar = np.full((720, 1280, 3), 30, np.uint8)
    cv2.putText(layar, "SESI SELESAI", (40, 80), HURUF, 1.8, (0, 0, 255), 4)
    y = 145
    for teks in baris_ringkasan:
        cv2.putText(layar, teks, (40, y), HURUF, 0.9, (255, 255, 255), 2)
        y += 40
    y += 15
    for teks in baris_bawah:
        for i in range(0, max(len(teks), 1), 80):      # potong teks panjang (misalnya path folder)
            cv2.putText(layar, teks[i:i + 80], (40, y), HURUF, 0.75, (0, 215, 255), 2)
            y += 34
    if tunggu:
        cv2.putText(layar, "Tekan tombol apa saja untuk menutup", (40, 690), HURUF, 0.8, (0, 200, 0), 2)
    cv2.imshow(JUDUL_JENDELA, layar)
    if not tunggu:
        for _ in range(5):                         # beri waktu jendela untuk menggambar
            cv2.waitKey(1)
        return
    while cv2.getWindowProperty(JUDUL_JENDELA, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) != -1:
            break
        
# 7. PROGRAM UTAMA
def buka_kamera():
    kamera = cv2.VideoCapture(config.KAMERA)
    kamera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # agar lancar di Full HD
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, config.LEBAR_KAMERA)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TINGGI_KAMERA)
    if not kamera.isOpened():
        raise RuntimeError("Kamera tidak dapat dibuka. Coba ganti KAMERA = 1 di config.py")
    print(f"[KAMERA] resolusi: {int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return kamera


def selesaikan_sesi(log):
    """Tampilkan ringkasan, tanya lokasi penyimpanan log, lalu simpan."""
    log["selesai"] = datetime.now()
    ringkasan = ringkasan_sesi(log)
    print("\n===== RINGKASAN SESI =====")
    print("\n".join(ringkasan))
    layar_ringkasan(ringkasan, ["Pilih tempat menyimpan log di jendela 'Simpan log sesi'..."])
    path = tanya_lokasi_simpan(log["mulai"].strftime("log_%Y%m%d_%H%M%S.csv"))
    path_csv, path_txt = simpan_log(log, path)
    print(f"[LOG] tersimpan: {path_csv}")
    print(f"[LOG] ringkasan: {path_txt}")
    layar_ringkasan(ringkasan, ["Log disimpan di:", path_csv, path_txt], tunggu=True)


def main():
    detektor = muat_detektor()
    model = muat_model_masker()
    print(f"[MODEL] klasifikasi: {model['jenis']}, kelas: {model['kelas']}")
    if "No_Mask" not in model["kelas"]:
        print("[MODEL] CATATAN: model belum punya kelas No_Mask -> orang tanpa masker "
              "biasanya terbaca sebagai SALAH PAKAI")
    suara = siapkan_suara()
    kamera = buka_kamera()
    cv2.namedWindow(JUDUL_JENDELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JUDUL_JENDELA, 1280, 720)

    log = None                  # None = sesi belum dimulai
    timer = buat_timer()
    tulisan_mulai_sampai = 0    # tulisan "MULAI" tampil sampai waktu ini
    waktu_lalu = time.time()
    print("[SESI] tekan SPASI di jendela kamera untuk MULAI, Q untuk SELESAI")
    try:
        while True:
            berhasil, frame = kamera.read()
            if not berhasil:
                print("[KAMERA] tidak mengirim gambar.")
                break
            if config.CERMIN:
                frame = cv2.flip(frame, 1)
            sekarang = time.time()

            # a) Deteksi & klasifikasi setiap wajah
            hitung = {}                           # status -> jumlah orang di frame ini
            for kotak in cari_wajah(detektor, frame):
                peluang = tebak_masker(model, potong_wajah(frame, *kotak))
                status, jenis, yakin = tentukan_status(peluang)
                hitung[status] = hitung.get(status, 0) + 1
                gambar_wajah(frame, kotak, status, jenis, yakin)

            # b) Peringatan suara & log (hanya saat sesi berjalan)
            if log is not None:
                status_terlihat = {s for s in hitung if s in PELANGGARAN}
                for status in perbarui_timer(timer, status_terlihat, sekarang):
                    hasil = putar_suara(suara, status)
                    print(f"[{datetime.now():%H:%M:%S}] PERINGATAN: {status} "
                          f"({hitung[status]} orang) -> suara: {hasil}")
                    catat_peringatan(log, status, hitung[status])
                perbarui_antrean_suara(suara)

            # c) Tampilan
            fps = 1.0 / max(sekarang - waktu_lalu, 1e-6)
            waktu_lalu = sekarang
            gambar_info(frame, hitung, fps, log)
            if sekarang < tulisan_mulai_sampai:
                gambar_tulisan_mulai(frame)
            cv2.imshow(JUDUL_JENDELA, frame)

            # d) Tombol: SPASI = mulai, Q / Esc / tutup jendela = selesai
            tombol = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(JUDUL_JENDELA, cv2.WND_PROP_VISIBLE) < 1:
                break
            if tombol == ord(" ") and log is None:
                log = buat_log()
                timer = buat_timer()
                tulisan_mulai_sampai = sekarang + 2
                print(f"[SESI] MULAI {datetime.now():%H:%M:%S}")
            elif tombol in (ord("q"), ord("Q"), 27):
                break
    finally:
        kamera.release()
        if suara["pemutar"] == "windows":
            perintah_windows("close alarm")
        if log is not None:
            selesaikan_sesi(log)
        else:
            print("[SESI] sesi belum dimulai, tidak ada log yang disimpan.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
