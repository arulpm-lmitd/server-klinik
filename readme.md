# 📘 API Dokumentasi: `POST /predict`

## 🔗 Endpoint
```
POST http://172.17.0.230:8000/predict
```

## 🧾 Headers

| Header         | Value               |
|----------------|---------------------|
| Content-Type   | application/json    |

---

## 📤 Request Body (JSON)

| Field       | Tipe     | Wajib | Deskripsi                                               |
|-------------|----------|--------|---------------------------------------------------------|
| `keluhan`   | string   | Ya     | Keluhan utama pasien, misalnya `"sakit kepala"`        |
| `anamnesa`  | string   | Ya     | Riwayat singkat kondisi pasien, misalnya `"demam tinggi sejak 2 hari"` |
| `top_k`     | integer  | Tidak  | Jumlah maksimal rekomendasi obat yang diinginkan (default: 5) |

### Contoh Request JSON
```json
{
  "keluhan": "sakit kepala",
  "anamnesa": "demam tinggi sejak 2 hari",
  "top_k": 5
}
```

---

## 🖥️ Contoh cURL

```bash
curl -X POST http://172.17.0.230:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "keluhan": "sakit kepala",
    "anamnesa": "demam tinggi sejak 2 hari",
    "top_k": 5
  }'
```

---

## ✅ Response (200 OK)

### Contoh Response
```json
{
  "predictions": [
    {
      "nama_obat": "Sanmol Forte 650 mg",
      "deskripsi_obat": "Paracetamol dosis tinggi untuk meredakan demam dan nyeri",
      "similarity_score": 0.8211547136306763,
      "confidence": "high",
      "rank": 1
    },
    {
      "nama_obat": "Farsifen 400 mg",
      "deskripsi_obat": "Obat antiinflamasi nonsteroid (NSAID) dosis tinggi untuk mengurangi nyeri",
      "similarity_score": 0.7806090712547302,
      "confidence": "medium",
      "rank": 2
    },
    {
      "nama_obat": "Farsifen 200 mg",
      "deskripsi_obat": "Obat antiinflamasi nonsteroid (NSAID) untuk mengurangi nyeri",
      "similarity_score": 0.7658687829971313,
      "confidence": "medium",
      "rank": 3
    },
    {
      "nama_obat": "Paracetamol 500 mg",
      "deskripsi_obat": "Analgesik antipiretik untuk demam dan nyeri",
      "similarity_score": 0.749209463596344,
      "confidence": "medium",
      "rank": 4
    }
  ]
}
```

---

## ℹ️ Keterangan Field pada Response

| Field             | Tipe     | Deskripsi |
|------------------|----------|-----------|
| `nama_obat`       | string   | Nama obat yang direkomendasikan |
| `deskripsi_obat`  | string   | Deskripsi singkat mengenai obat |
| `similarity_score`| float    | Skor kesamaan antara input dan deskripsi obat (semakin tinggi semakin relevan) |
| `confidence`      | string   | Estimasi tingkat kepercayaan model terhadap prediksi: `high`, `medium`, atau `low` |
| `rank`            | integer  | Peringkat hasil berdasarkan skor kemiripan |

---

## ⚠️ Error Response

### 400 Bad Request
```json
{
  "detail": "Field 'keluhan' tidak boleh kosong"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Terjadi kesalahan di server"
}
