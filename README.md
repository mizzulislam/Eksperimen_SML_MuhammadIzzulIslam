# Eksperimen_SML_MuhammadIzzulIslam

Repository ini berisi tahapan eksperimen awal untuk proyek **Sistem Machine
Learning: Credit Scoring Classification**, sebagai bagian dari submission
"Proyek Akhir Membangun Sistem Machine Learning" - Dicoding.

## Struktur Folder

```
Eksperimen_SML_MuhammadIzzulIslam/
├── .github/workflows/preprocessing.yml   -> GitHub Actions untuk otomatisasi preprocessing
├── credit_scoring_raw/                   -> Dataset mentah
│   └── credit_score_data.csv
└── preprocessing/
    ├── Eksperimen_MuhammadIzzulIslam.ipynb   -> Notebook eksperimen (EDA & preprocessing manual)
    ├── automate_MuhammadIzzulIslam.py        -> Script otomatisasi preprocessing
    └── credit_scoring_preprocessing/         -> Output dataset siap latih
        ├── train_data.csv
        └── test_data.csv
```

## Dataset

**Credit Score Classification Dataset** - sumber: [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

Dataset berisi informasi finansial dan riwayat kredit nasabah untuk
mengklasifikasikan skor kredit ke dalam 3 kategori: `Poor`, `Standard`, `Good`.

## Cara Menjalankan

### Eksperimen Manual (Notebook)
Buka dan jalankan `preprocessing/Eksperimen_MuhammadIzzulIslam.ipynb` secara berurutan.

### Preprocessing Otomatis (Script)
```bash
cd preprocessing
pip install pandas numpy scikit-learn
python automate_MuhammadIzzulIslam.py
```

### GitHub Actions
Workflow akan otomatis berjalan setiap ada push ke folder `preprocessing/`,
atau dapat dipicu manual melalui tab Actions (workflow_dispatch).
