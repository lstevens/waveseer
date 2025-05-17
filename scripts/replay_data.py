#!/usr/bin/env python3
"""
Replay CSV data to the ingest stream at adjustable speed.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


def replay_file(path, mapping, stream_url, speed, max_events=None):
    df = pd.read_csv(path, parse_dates=[mapping['timestamp']])
    df = df.sort_values(mapping['timestamp']).reset_index(drop=True)
    times = df[mapping['timestamp']].tolist()
    for idx, row in df.iterrows():
        if max_events is not None and idx >= max_events:
            break
        payload = {
            'ts_start': row[mapping['timestamp']].isoformat(),
            'open': float(row[mapping['open']]),
            'high': float(row[mapping['high']]),
            'low': float(row[mapping['low']]),
            'close': float(row[mapping['close']]),
            'volume': float(row[mapping['volume']]) if mapping.get('volume') else None
        }
        try:
            resp = requests.post(stream_url, json=payload)
            resp.raise_for_status()
        except Exception as e:
            print(f"Error posting row {idx} from {path}: {e}")
        if idx < len(df) - 1:
            delta = (times[idx+1] - times[idx]).total_seconds()
            time.sleep(delta / speed)


def main():
    parser = argparse.ArgumentParser(description='Replay CSV data into ingest stream')
    parser.add_argument('--index', default='data/index.json', help='Path to CSV index JSON')
    parser.add_argument('--stream_url', default='http://localhost:8000/stream', help='Ingest stream endpoint')
    parser.add_argument('--speed', type=float, default=1.0, help='Replay speed multiplier')
    parser.add_argument('--max-events', type=int, default=None, help='Maximum number of rows to send per file')
    args = parser.parse_args()

    idx_file = Path(args.index)
    if not idx_file.exists():
        print(f"Index file not found: {idx_file}")
        return
    entries = json.load(idx_file.open())
    for ent in entries:
        path = Path(ent['path'])
        mapping = ent['columns']
        print(f"Replaying {path}...")
        replay_file(path, mapping, args.stream_url, args.speed, args.max_events)


if __name__ == '__main__':
    main()
