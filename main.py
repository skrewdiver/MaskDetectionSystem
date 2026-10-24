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

# Warna format OpenCV: (Biru, Hijau, Merah)
WARNA = {PATUH: (0, 200, 0), JENIS_TIDAK_SESUAI: (0, 215, 255), SALAH_PAKAI: (0, 0, 230),
         TANPA_MASKER: (0, 0, 230), BELUM_PASTI: (160, 160, 160)}
NAMA_JENIS = {"Surgical": "Medis", "N95": "N95", "Cotton": "Kain", "Bandana": "Bandana"}
HURUF = cv2.FONT_HERSHEY_SIMPLEX


# 1. MEMUAT MODEL
def muat_detektor():
    """Detektor wajah SCRFD (model pra-latih InsightFace)."""
    detektor = SCRFD(model_file=config.FILE_DETEKTOR)
    detektor.prepare(-1)                          
    return detektor


def muat_model_masker():
    """Model klasifikasi masker (MobileNetV2 hasil training). Mengembalikan dict berisi model & info."""
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
    """Mengembalikan daftar kotak wajah [(x, y, lebar, tinggi), ...]."""
    ukuran = (config.UKURAN_DETEKSI, config.UKURAN_DETEKSI)
    hasil, _ = detektor.detect(frame, thresh=0.5, input_size=ukuran)
    kotak = []
    for x1, y1, x2, y2, skor in hasil:         
        w, h = int(x2 - x1), int(y2 - y1)
        if w >= 20 and h >= 20:                  
            kotak.append((max(0, int(x1)), max(0, int(y1)), w, h))
    return kotak


def potong_wajah(frame, x, y, w, h):
    """Memotong wajah berbentuk persegi, sedikit diperbesar agar hidung, mulut & dagu ikut."""
    sisi = int(max(w, h) * 1.25)
    cx = x + w // 2
    cy = y + h // 2 + int(h * 0.05)             
    x0, y0 = cx - sisi // 2, cy - sisi // 2
    x1, y1 = x0 + sisi, y0 + sisi
    potongan = frame[max(0, y0):y1, max(0, x0):x1]
    
    atas, kiri = max(0, -y0), max(0, -x0)
    bawah, kanan = max(0, y1 - frame.shape[0]), max(0, x1 - frame.shape[1])
    return cv2.copyMakeBorder(potongan, atas, bawah, kiri, kanan, cv2.BORDER_REPLICATE)


def tebak_masker(model, wajah):
    """Mengembalikan peluang setiap kelas, misalnya {"Correct_N95": 0.91, "Incorrect": 0.06, ...}."""
    rgb = cv2.cvtColor(wajah, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (model["ukuran"], model["ukuran"]), interpolation=cv2.INTER_AREA)
    data = rgb.astype(np.float32)[np.newaxis]
    if model["jenis"] == "onnx":
        peluang = model["sesi"].run(None, {model["nama_input"]: data})[0][0]
    else:
        peluang = np.asarray(model["keras"](data, training=False))[0]
    return {kelas: float(p) for kelas, p in zip(model["kelas"], peluang)}
