#!/usr/bin/env python3
"""
Scan a directory for CSV files, infer column mappings, and write an index JSON for replay.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def infer_mapping(columns):
    """Infer standard schema mapping from CSV column names."""
    mapping = {}
    low = {c.lower(): c for c in columns}
    def find(keys):
        for key in keys:
            for lc, orig in low.items():
                if key in lc:
                    return orig
        return None
    mapping['timestamp'] = find(['time', 'date', 'ts'])
    mapping['open'] = find(['open'])
    mapping['high'] = find(['high'])
    mapping['low'] = find(['low'])
    mapping['close'] = find(['close'])
    mapping['volume'] = find(['vol'])
    return mapping


def main():
    parser = argparse.ArgumentParser(description='Collect CSV metadata for replay')
    parser.add_argument('--data-dir', default='data', help='Directory to scan for CSV files')
    parser.add_argument('--output', default='data/index.json', help='Path to write the index JSON file')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = list(data_dir.rglob('*.csv'))
    entries = []

    for f in files:
        try:
            df = pd.read_csv(f, nrows=0)
            cols = df.columns.tolist()
            mapping = infer_mapping(cols)
            entries.append({'path': str(f), 'columns': mapping})
        except Exception as e:
            print(f"Error reading {f}: {e}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as fp:
        json.dump(entries, fp, indent=2)
    print(f"Wrote {len(entries)} CSV entries to {out_path}")


if __name__ == '__main__':
    main()
