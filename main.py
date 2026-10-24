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