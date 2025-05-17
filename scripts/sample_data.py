#!/usr/bin/env python3
"""
Sample CSV data by a time window and generate a sample index for integration tests.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Sample CSV data by datetime window")
    parser.add_argument('--index', default='data/index.json', help='Path to full index JSON')
    parser.add_argument('--start', required=True, help='Start datetime inclusive (ISO format)')
    parser.add_argument('--end', required=True, help='End datetime inclusive (ISO format)')
    parser.add_argument('--sample-dir', default='data/sample', help='Dir to write sampled CSVs')
    parser.add_argument('--output-index', default='data/sample_index.json', help='Path for sample index JSON')
    args = parser.parse_args()

    # Load full index
    full_index = json.load(open(args.index))
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_entries = []

    for ent in full_index:
        csv_path = Path(ent['path'])
        mapping = ent['columns']
        ts_col = mapping.get('timestamp')
        if ts_col is None:
            continue
        try:
            df = pd.read_csv(csv_path)
            # Ensure timestamp column is datetime
            from pandas.api.types import is_numeric_dtype
            if is_numeric_dtype(df[ts_col]):
                # epoch seconds to datetime
                df[ts_col] = pd.to_datetime(df[ts_col], unit='s')
            else:
                df[ts_col] = pd.to_datetime(df[ts_col])
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
        # Filter by time window
        mask = (df[ts_col] >= start) & (df[ts_col] <= end)
        df_sample = df.loc[mask]
        if df_sample.empty:
            continue
        # Write sampled CSV
        sample_path = sample_dir / csv_path.name
        df_sample.to_csv(sample_path, index=False)
        sample_entries.append({'path': str(sample_path), 'columns': mapping})

    # Write sample index
    out_index = Path(args.output_index)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    with open(out_index, 'w') as f:
        json.dump(sample_entries, f, indent=2)
    print(f"Wrote {len(sample_entries)} sampled CSV(s) to {sample_dir} and index {out_index}")

if __name__ == '__main__':
    main()
