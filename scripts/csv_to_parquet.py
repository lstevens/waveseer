#!/usr/bin/env python3
"""
Convert sample CSVs into Parquet files under build/cache for use by the seer CLI.
"""
import json
from pathlib import Path
import polars as pl


def main():
    # Load sample index
    idx_path = Path('data/sample_index.json')
    if not idx_path.exists():
        print('Sample index not found; run scripts/sample_data.py first')
        return
    # Load sample index entries
    entries = json.loads(idx_path.read_text())

    for ent in entries:
        csv_path = Path(ent['path'])
        mapping = ent['columns']
        ts_col = mapping.get('timestamp')
        if not ts_col:
            continue

        # Read CSV
        df = pl.read_csv(str(csv_path))
        # Rename and cast columns
        df = df.with_columns([
            pl.col(ts_col).alias('datetime'),
            pl.col(mapping.get('open')).cast(pl.Float64).alias('open'),
            pl.col(mapping.get('high')).cast(pl.Float64).alias('high'),
            pl.col(mapping.get('low')).cast(pl.Float64).alias('low'),
            pl.col(mapping.get('close')).cast(pl.Float64).alias('close'),
            pl.col(mapping.get('volume')).cast(pl.Float64).alias('volume'),
        ])

        # Derive symbol and tf from filename
        name = csv_path.stem  # e.g. 'btcusd_1-min_2017'
        parts = name.split('_')
        symbol = parts[0]
        # Derive tf to match config (e.g., '1-min' -> '1m')
        tf_raw = parts[1]
        if tf_raw.endswith('min'):
            tf = tf_raw.replace('-min', 'm')
        elif tf_raw.endswith('hour'):
            tf = tf_raw.replace('-hour', 'h')
        else:
            tf = tf_raw.replace('-', '')

        # Write Parquet under build/cache/<symbol>/<tf>.parquet
        out_dir = Path('build') / 'cache' / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{tf}.parquet"
        df.write_parquet(str(out_path))
        print(f"Wrote Parquet for {symbol}/{tf} to {out_path}")

if __name__ == '__main__':
    main()
