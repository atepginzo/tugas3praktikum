from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import re

app = Flask(__name__)

# Load model
model = joblib.load('model_regresi.pkl')

# Fungsi parsing harga dari string (Juta/Miliar)
def parse_price(price_str):
    price_str = str(price_str)
    if 'Miliar' in price_str:
        return float(re.sub(r'[^\d,]', '', price_str).replace(',', '.')) * 1_000_000_000
    elif 'Juta' in price_str:
        return float(re.sub(r'[^\d,]', '', price_str).replace(',', '.')) * 1_000_000
    else:
        return None

@app.route('/')
def home():
    # Load dataset asli
    df = pd.read_csv(r'venv/harga_rumah.csv')

    # Bersihkan dan siapkan kolom harga
    df['Harga'] = df['price'].apply(parse_price)

    # Bersihkan kolom lainnya
    df['LT'] = df['surface_area'].str.replace(' m²', '', regex=False)
    df['LB'] = df['building_area'].str.replace(' m²', '', regex=False)

    df = df.dropna(subset=['LT', 'LB', 'Harga'])
    df['LT'] = df['LT'].astype(float)
    df['LB'] = df['LB'].astype(float)
    df['Lokasi'] = df['listing-location']

    # Ambil lokasi yang dikenal oleh model
    encoder = model.named_steps['preprocessor'].named_transformers_['cat']
    lokasi_valid = list(encoder.categories_[0])

    # Filter hanya data dengan lokasi yang dikenal
    df = df[df['Lokasi'].isin(lokasi_valid)]

    # Siapkan fitur untuk prediksi
    input_X = df[['LT', 'LB', 'Lokasi']]
    df['Prediksi Harga'] = model.predict(input_X)

    tabel_data = df[['LT', 'LB', 'Harga', 'Lokasi', 'Prediksi Harga']].round(2).to_dict(orient='records')

    return render_template('index.html', lokasi_options=lokasi_valid, tabel=tabel_data)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input
        LT = float(request.form['LT'])
        LB = float(request.form['LB'])
        lokasi = request.form['Lokasi']

        # Cek validitas lokasi
        encoder = model.named_steps['preprocessor'].named_transformers_['cat']
        lokasi_valid = list(encoder.categories_[0])

        if lokasi not in lokasi_valid:
            return render_template('result.html', prediction=None, input_data=None,
                                   error="Lokasi tidak dikenali oleh model.")

        # Bentuk input untuk prediksi
        input_df = pd.DataFrame({
            'LT': [LT],
            'LB': [LB],
            'Lokasi': [lokasi]
        })

        prediksi = model.predict(input_df)[0]

        return render_template('result.html',
                               prediction=round(prediksi, 2),
                               input_data={'LT': LT, 'LB': LB, 'Lokasi': lokasi},
                               error=None)

    except Exception as e:
        return render_template('result.html',
                               prediction=None,
                               input_data=None,
                               error=f"Terjadi kesalahan saat memproses data: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
