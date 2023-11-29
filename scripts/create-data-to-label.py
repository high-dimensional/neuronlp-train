#!/usr/bin/env python
"""Sample and convert domain data for prodigy.

This script takes a CSV of report data in UCL format and creates a prodigy-compatible JSONL for
pathological domain labelling
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from srsly import write_jsonl

PATHOLOGY_DOMAINS = [
    "pathology-cerebrovascular",
    "pathology-congenital-developmental",
    "pathology-csf-disorders",
    "pathology-endocrine",
    "pathology-haemorrhagic",
    "pathology-infectious",
    "pathology-inflammatory-autoimmune",
    "pathology-ischaemic",
    "pathology-metabolic-nutritional-toxic",
    "pathology-neoplastic-paraneoplastic",
    "pathology-neurodegenerative-dementia",
    "pathology-opthalmological",
    "pathology-traumatic",
    "pathology-treatment",
    "pathology-vascular",
]


def load_input(location):
    """load in data or take input from stdin"""
    df = pd.read_csv(
        location,
        low_memory=False,
    )
    return df


def sample_data(data, n_sample):
    """perform the necessary transformation on the input data"""
    columns_names = ["asserted-" + i for i in PATHOLOGY_DOMAINS]
    has_one = data[(data[columns_names] > 0).any(axis=1)]
    counts = has_one[columns_names].sum(axis=0)
    sort_normal = (counts / counts.sum()).sort_values()
    max_ = sort_normal.max()
    reweighted = (max_ - sort_normal) ** 2
    renormal = reweighted / reweighted.sum()
    per_class = (renormal * (n_sample + 2)).round().astype(int)
    selections = per_class.to_dict()
    df_samp = sample_multilabel(has_one, selections)
    (df_samp[columns_names].astype(bool).astype(int)).sum(axis=0).sort_values()
    df_samp = df_samp.sample(n_sample)
    return df_samp


def sample_multilabel(df, n_per_class):
    """Sample a multilabel dataset with a relatively even label distribution"""
    has_X = df[n_per_class.keys()] > 0
    index_set = pd.Index([])
    for col, n in n_per_class.items():
        index_set = index_set.append(has_X[has_X[col]].sample(n).index)
    return df.loc[index_set]


def get_jsonl_dict(rows):
    as_records = rows.to_dict("records")
    options = [{"id": i, "text": i} for i in PATHOLOGY_DOMAINS]
    records = [
        {
            "text": r["report_body_masked"],
            "options": options,
            "_view_id": "choice",
            "config": {"choice_style": "multiple"},
            "answer": "accept",
            "accept": [j for j in PATHOLOGY_DOMAINS if r["asserted-" + j] > 0],
        }
        for r in as_records
    ]
    return records


def output_results(data, args):
    """output analysis, save to file or send to stdout"""
    filename = str(args.input.stem) + "_to_label.jsonl"
    random.shuffle(data)
    write_jsonl(args.outdir / filename, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV of report data to sample", type=Path)
    parser.add_argument(
        "-o", "--outdir", help="output directory", type=Path, default=Path.cwd()
    )
    parser.add_argument(
        "-n", "--samples", help="number of samples", type=int, default=100
    )
    args = parser.parse_args()
    if not args.outdir.exists():
        args.outdir.mkdir()
    args = parser.parse_args()
    data = load_input(args.input)
    sample = sample_data(data, args.samples)
    converted_data = get_jsonl_dict(sample)
    random.shuffle(converted_data)
    output_results(converted_data, args)


if __name__ == "__main__":
    main()
