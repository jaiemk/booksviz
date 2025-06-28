import pandas as pd
import numpy as np
import os
import json

CSV_FILE = 'GoodReads_100k_books.csv.xz'
JSON_FILE = 'scatter_data.json'
SAMPLE_SIZE = 5000
RANDOM_SEED = 42


def main():
    # Load required columns
    df = pd.read_csv(CSV_FILE, usecols=['pages', 'desc', 'reviews', 'rating'])

    # Compute blurb length and drop the description text
    df['blurb'] = df['desc'].astype(str).str.len()
    df.drop(columns='desc', inplace=True)

    # Filter outliers per column using 0.5 to 99.5 percentiles
    for col in df.columns:
        low, high = np.percentile(df[col].dropna(), [0.5, 99.5])
        df = df[df[col].between(low, high)]

    # Random sample up to SAMPLE_SIZE rows
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)

    # Output to JSON
    df.to_json(JSON_FILE, orient='records')
    size = os.path.getsize(JSON_FILE)
    print(f"{JSON_FILE} size: {size} bytes")


if __name__ == '__main__':
    main()
