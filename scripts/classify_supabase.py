#!/usr/bin/env python
"""Fetch rows from Supabase REST API, classify using local model, and optionally update table with safety class.

Usage:
    export SUPABASE_URL="https://..."
    export SUPABASE_KEY="<api_key>"
    python scripts/classify_supabase.py --table sensor_data --checkpoint checkpoints --out_csv results.csv --update
"""
import os
import argparse
import requests
import pandas as pd
import numpy as np
from src.inference import load_model_and_meta, classify_series


def fetch_table_rows(supabase_url, api_key, table, select_cols=None):
    headers = {
        'apikey': api_key,
        'Authorization': f'Bearer {api_key}',
    }
    select = select_cols or '*'
    url = f"{supabase_url.rstrip('/')}/rest/v1/{table}?select={select}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def update_row_class(supabase_url, api_key, table, pk_name, pk_value, payload):
    headers = {
        'apikey': api_key,
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation',
    }
    # patch by primary key equality
    url = f"{supabase_url.rstrip('/')}/rest/v1/{table}?{pk_name}=eq.{pk_value}"
    resp = requests.patch(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


def determine_pk(df):
    # prefer 'id' then first integer column
    if 'id' in df.columns:
        return 'id'
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col].dtype) and df[col].is_unique:
            return col
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supabase_url', default=os.getenv('SUPABASE_URL'))
    parser.add_argument('--api_key', default=os.getenv('SUPABASE_KEY'))
    parser.add_argument('--table', required=True)
    parser.add_argument('--checkpoint', default='checkpoints')
    parser.add_argument('--window', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--out_csv', default='supabase_classification.csv')
    parser.add_argument('--update', action='store_true', help='If set, update table with safety class in column "safety_class"')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if not args.supabase_url or not args.api_key:
        raise SystemExit('Supabase URL and API key must be provided via args or SUPABASE_URL/SUPABASE_KEY env vars')

    print('Fetching rows from Supabase...')
    rows = fetch_table_rows(args.supabase_url, args.api_key, args.table)
    if not rows:
        print('No rows returned from table.')
        return

    df = pd.DataFrame(rows)
    needed = ['AccX', 'AccY', 'AccZ']
    if not all(c in df.columns for c in needed):
        raise SystemExit(f"Table must contain columns: {needed}. Found: {list(df.columns)}")

    # order by primary key if available
    pk = determine_pk(df)
    if pk:
        df = df.sort_values(by=pk).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    arr = df[['AccX', 'AccY', 'AccZ']].astype(float).values

    # load model and metadata
    model, scaler, thresh = load_model_and_meta(checkpoint_dir=args.checkpoint, device=args.device)

    print('Classifying series with model...')
    _, per_window_err, sample_flags = classify_series(arr, model, scaler, thresh,
                                                     window=args.window, stride=args.stride,
                                                     device=args.device)

    df['unsafe'] = sample_flags.astype(int)
    df['safety_class'] = df['unsafe'].map({0: 'Class 1', 1: 'Class 2'})

    print('Summary:')
    print(df['safety_class'].value_counts(dropna=False))

    df.to_csv(args.out_csv, index=False)
    print(f'Wrote results to {args.out_csv}')

    if args.update:
        if pk is None:
            raise SystemExit('Cannot update table without a unique integer primary key column (e.g., id)')
        print('Updating rows on Supabase (safety_class column) ...')
        # Update rows individually to set safety_class
        for _, row in df.iterrows():
            payload = {'safety_class': row['safety_class']}
            try:
                update_row_class(args.supabase_url, args.api_key, args.table, pk, row[pk], payload)
            except Exception as e:
                print(f'Failed to update row {row[pk]}: {e}')
        print('Update complete.')


if __name__ == '__main__':
    main()
