import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime

import config

EKSTENSI_GAMBAR = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
URUTAN_JENIS = ["Surgical", "N95", "Cotton", "Bandana"]
URUTAN_KELAS = [f"Correct_{j}" for j in URUTAN_JENIS] + ["Incorrect", "No_Mask"]



def label_dari_path(path, folder_dataset):
    """'.../Correct/N95/Female/x.jpg' -> 'Correct_N95', '.../Incorrect/N95/x.jpg' -> 'Incorrect'."""
    bagian = [p.lower() for p in os.path.relpath(path, folder_dataset).replace("\\", "/").split("/")]
    if "incorrect" in bagian:
        return "Incorrect"                       
    jenis = next((j for j in URUTAN_JENIS if j.lower() in bagian), None)
    if "correct" in bagian and jenis:
        return f"Correct_{jenis}"
    return None


def kumpulkan_gambar(folder_dataset, folder_tanpa_masker=None):
    """Mengembalikan daftar (path_gambar, label)."""
    data = []
    for akar, _, files in os.walk(folder_dataset):
        for f in files:
            if f.lower().endswith(EKSTENSI_GAMBAR):
                path = os.path.join(akar, f)
                label = label_dari_path(path, folder_dataset)
                if label:
                    data.append((path, label))
    if folder_tanpa_masker:
        for akar, _, files in os.walk(folder_tanpa_masker):
            data += [(os.path.join(akar, f), "No_Mask") for f in files if f.lower().endswith(EKSTENSI_GAMBAR)]
    if not data:
        raise RuntimeError(f"Tidak ada gambar yang dikenali di {folder_dataset}")
    return data


def bagi_data(data, porsi_validasi=0.2):
    """Membagi data per kelas: 80% latih, 20% validasi (acak tapi selalu sama, seed 42)."""
    acak = random.Random(42)
    latih, validasi = [], []
    for label in sorted({lb for _, lb in data}):
        isi = [d for d in data if d[1] == label]
        acak.shuffle(isi)
        n_val = max(1, int(len(isi) * porsi_validasi))
        validasi += isi[:n_val]
        latih += isi[n_val:]
    acak.shuffle(latih)
    return latih, validasi


def buat_dataset(tf, data, kelas, ukuran, batch, untuk_latih):
    indeks = {k: i for i, k in enumerate(kelas)}
    ds = tf.data.Dataset.from_tensor_slices(([p for p, _ in data], [indeks[lb] for _, lb in data]))

    def baca(path, y):
        gambar = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        gambar = tf.image.resize(gambar, (ukuran, ukuran), antialias=True)
        return tf.clip_by_value(gambar, 0, 255), y

    ds = ds.map(baca, num_parallel_calls=tf.data.AUTOTUNE).batch(batch)
    if untuk_latih:
        augmentasi = tf.keras.Sequential([          
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.25),
        ])
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmentasi(x, training=True), 0, 255), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)



def buat_model(tf, jumlah_kelas, ukuran, bobot_awal="imagenet"):
    """MobileNetV2 + lapisan Dense baru. Mengembalikan (model, backbone)."""
    masukan = tf.keras.Input((ukuran, ukuran, 3), name="image")             
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1.0)(masukan)          
    backbone = tf.keras.applications.MobileNetV2(include_top=False, weights=bobot_awal,
                                                 input_shape=(ukuran, ukuran, 3))
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    keluaran = tf.keras.layers.Dense(jumlah_kelas, activation="softmax", name="probs")(x)
    return tf.keras.Model(masukan, keluaran), backbone



def latih(tf, folder_dataset, folder_tanpa_masker, ukuran=128, batch=32, epoch=5, epoch_finetune=3,
          bobot_awal="imagenet"):
    data = kumpulkan_gambar(folder_dataset, folder_tanpa_masker)
    jumlah = Counter(lb for _, lb in data)
    kelas = [k for k in URUTAN_KELAS if k in jumlah]
    print(f"Jumlah gambar per kelas: {dict(jumlah)}")
    if "No_Mask" not in kelas:
        print("CATATAN: tidak ada kelas No_Mask -> orang tanpa masker tidak bisa dikenali. "
              "Tambahkan --tanpa-masker FOLDER")

    data_latih, data_val = bagi_data(data)
    print(f"Data latih: {len(data_latih)} | data validasi: {len(data_val)}")
    ds_latih = buat_dataset(tf, data_latih, kelas, ukuran, batch, True)
    ds_val = buat_dataset(tf, data_val, kelas, ukuran, batch, False)

   
    n_latih = Counter(lb for _, lb in data_latih)
    bobot = {i: len(data_latih) / (len(kelas) * n_latih[k]) for i, k in enumerate(kelas)}

    model, backbone = buat_model(tf, len(kelas), ukuran, bobot_awal)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    print("\n== Tahap 1: latih lapisan baru ==")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    model.fit(ds_latih, validation_data=ds_val, epochs=epoch, class_weight=bobot)

    if epoch_finetune > 0:
        print("\n== Tahap 2: fine-tune sebagian MobileNetV2 ==")
        backbone.trainable = True
        for lapisan in backbone.layers[:100]:       
            lapisan.trainable = False
        for lapisan in backbone.layers:
            if isinstance(lapisan, tf.keras.layers.BatchNormalization):
                lapisan.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=["accuracy"])
        model.fit(ds_latih, validation_data=ds_val, epochs=epoch_finetune, class_weight=bobot)

    _, akurasi = model.evaluate(ds_val, verbose=0)
    print(f"\nAkurasi validasi akhir: {akurasi * 100:.2f}%")
    info = {"classes": kelas, "input_size": ukuran, "backbone": "mobilenetv2",
            "preprocess": "RGB float32 0..255 (normalisasi di dalam model)",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "n_train": len(data_latih), "n_val": len(data_val), "best_val_accuracy": float(akurasi)}
    return model, info



def ekspor_onnx(tf, model, ukuran, path):
    try:
        model.export(path, format="onnx")            
        return True
    except Exception:
        pass
    try:
        import tf2onnx
        spek = (tf.TensorSpec((None, ukuran, ukuran, 3), tf.float32, name="image"),)
        tf2onnx.convert.from_keras(model, input_signature=spek, opset=13, output_path=path)
        return True
    except Exception:
        return False


def simpan(tf, model, info, folder_output):
    os.makedirs(folder_output, exist_ok=True)
    file_keras = os.path.join(folder_output, "mask_classifier.keras")
    file_onnx = os.path.join(folder_output, "mask_classifier.onnx")
    model.save(file_keras)
    with open(os.path.join(folder_output, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Model tersimpan: {file_keras}")
    if ekspor_onnx(tf, model, info["input_size"], file_onnx):
        print(f"Model ONNX     : {file_onnx}")
    else:
       
        if os.path.exists(file_onnx):
            os.remove(file_onnx)
        print("Ekspor ONNX gagal (opsional; pip install tf2onnx). main.py akan memakai file .keras")



def main():
    ap = argparse.ArgumentParser(description="Latih model klasifikasi masker")
    ap.add_argument("--dataset", required=True, help="folder NewFace Mask Dataset (berisi Correct/ dan Incorrect/)")
    ap.add_argument("--tanpa-masker", help="folder gambar wajah tanpa masker (opsional, untuk kelas No_Mask)")
    ap.add_argument("--epoch", type=int, default=5, help="epoch tahap 1")
    ap.add_argument("--epoch-finetune", type=int, default=3, help="epoch tahap 2 (0 = lewati)")
    ap.add_argument("--ukuran", type=int, default=128, help="ukuran gambar input (piksel)")
    args = ap.parse_args()

    import tensorflow as tf                          
    tf.keras.utils.set_random_seed(42)
    model, info = latih(tf, args.dataset, args.tanpa_masker, args.ukuran,
                        epoch=args.epoch, epoch_finetune=args.epoch_finetune)
    simpan(tf, model, info, config.FOLDER_MODEL)


if __name__ == "__main__":
    main()
