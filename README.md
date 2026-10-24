# Nama Kelompok 5
Muhammad Aqzhara Fathyan Khairi	5026241053
Athalah Hilman Hilalazka		5026241145
Yustisio Priyatno    			5026241104
Benedictus Nicholas Christian 	5026241187
Claudio Huge Siburian 		    5026241112

# Deteksi Kepatuhan Masker (Versi Sederhana, Tanpa OOP)

Program Python tanpa GUI dan tanpa kelas (hanya fungsi) untuk memantau pemakaian masker lewat webcam. Video tampil di jendela OpenCV, dan peringatan serta ringkasan dicetak di terminal.

## Isi folder

```
mask_monitor_sederhana/
├── main.py              <- JALANKAN INI untuk pemantauan (semua fungsi pemantauan ada di sini)
├── train.py             <- JALANKAN INI untuk melatih model masker
├── config.py            <- semua pengaturan (kamera, aturan, waktu, nama file)
├── scrfd.py             <- kode resmi InsightFace untuk detektor wajah SCRFD (tidak perlu diubah)
├── SCRFD_LICENSE.txt    <- lisensi Apache 2.0 untuk scrfd.py
├── model/               <- scrfd_2.5g.onnx (deteksi wajah) + mask_classifier.onnx + labels.json
├── suara/               <- taruh file MP3 di sini
└── log/                 <- hasil log (bawaan)
```

`main.py` dibagi menjadi 7 bagian, dibaca dari atas ke bawah:

| Bagian | Fungsi utama |
| --- | --- |
| 1. Memuat model | `muat_detektor()`, `muat_model_masker()` |
| 2. Deteksi & klasifikasi | `cari_wajah()`, `potong_wajah()`, `tebak_masker()` |
| 3. Aturan & timer | `tentukan_status()`, `perbarui_timer()` |
| 4. Suara | `siapkan_suara()`, `putar_suara()` |
| 5. Log | `buat_log()`, `catat_peringatan()`, `simpan_log()` |
| 6. Tampilan | `gambar_wajah()`, `gambar_info()`, `layar_ringkasan()` |
| 7. Program utama | `main()`: loop kamera |

## Menjalankan pemantauan

```
pip install -r requirements.txt
python main.py
```

1. Taruh MP3 di folder `suara/`: `salah_pakai.mp3`, `tanpa_masker.mp3`, `jenis_tidak_sesuai.mp3`.
2. **SIAP** (lampu kuning): kamera menyala dan kotak wajah tampil, tapi belum ada peringatan atau log.
3. Tekan **SPASI** untuk mulai. Tulisan **MULAI** muncul, lalu tanda **MEMANTAU 00:00:00** tampil.
4. Tekan **Q** untuk selesai. Layar **SESI SELESAI** menampilkan ringkasan, lalu jendela **Simpan log sesi** muncul. Jika dibatalkan, log disimpan di `log/`.

Warna kotak:
- hijau = patuh
- kuning = jenis masker tidak sesuai
- merah = salah pakai / tanpa masker
- abu-abu = belum pasti


