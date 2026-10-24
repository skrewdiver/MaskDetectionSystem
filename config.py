"""
PENGATURAN PROGRAM
Semua angka dan nama file yang boleh diubah dikumpulkan di sini.
"""
import os

FOLDER = os.path.dirname(os.path.abspath(__file__))

# ---------------- Kamera ----------------
KAMERA = 0                      # 0 = webcam laptop, 1 = kamera USB
LEBAR_KAMERA = 1920             # Full HD
TINGGI_KAMERA = 1080
CERMIN = True                   # balik gambar kiri-kanan seperti cermin

# ---------------- Aturan ruangan ----------------
# Jenis masker yang boleh dipakai. Pilihan: "Surgical" (medis), "N95", "Cotton" (kain), "Bandana"
JENIS_DIIZINKAN = ["Surgical", "N95"]

BATAS_WAKTU = 3.0               # detik pelanggaran terus-menerus sebelum suara diputar
JEDA_PERINGATAN = 15.0          # detik minimal antar-suara untuk pelanggaran yang sama
TOLERANSI = 1.5                 # pelanggaran hilang sebentar (< 1,5 detik) tidak mengulang timer
KEYAKINAN_MIN = 0.5             # tebakan model di bawah 50% dianggap "belum pasti"

# ---------------- File model ----------------
FOLDER_MODEL = os.path.join(FOLDER, "model")
FILE_DETEKTOR = os.path.join(FOLDER_MODEL, "scrfd_2.5g.onnx")   # detektor wajah SCRFD (InsightFace)

FILE_MODEL_ONNX = os.path.join(FOLDER_MODEL, "mask_classifier.onnx")
FILE_MODEL_KERAS = os.path.join(FOLDER_MODEL, "mask_classifier.keras")
FILE_LABEL = os.path.join(FOLDER_MODEL, "labels.json")

# ---------------- Deteksi wajah ----------------
# Ukuran gambar untuk SCRFD. 640 = seimbang. Naikkan ke 960 bila kamera memantau seluruh ruangan
# (wajah jauh/kecil lebih banyak tertangkap, tetapi FPS turun).
UKURAN_DETEKSI = 640

# ---------------- Suara peringatan ----------------
FOLDER_SUARA = os.path.join(FOLDER, "suara")
# Nama file yang dicari di folder "suara" (boleh .mp3 / .wav / .ogg).
# Beberapa nama diterima agar file dari versi lengkap (incorrect.mp3, dst.) juga bisa dipakai.
NAMA_SUARA = {
    "SALAH PAKAI": ["salah_pakai", "incorrect"],
    "TANPA MASKER": ["tanpa_masker", "no_mask"],
    "JENIS TIDAK SESUAI": ["jenis_tidak_sesuai", "wrong_type"],
}

# ---------------- Log ----------------
FOLDER_LOG = os.path.join(FOLDER, "log")
