#!/usr/bin/env python

import argparse
import os
import subprocess
import csv

bert_batch_matmul_workloads = [
    [2, 1, 128, 3072, 768],
    [2, 1, 128, 2304, 768],
    [2, 1, 128, 768, 768],
    [2, 1, 128, 768, 3072],
    [2, 1, 128, 768, 2304],
    [2, 2, 3072, 768, 128],
    [2, 2, 768, 3072, 128],
    [24, 24, 128, 128, 64],
    [24, 24, 128, 64, 128],
    [24, 24, 64, 128, 128],
    [2, 2, 2304, 768, 128],
    [2, 2, 768, 768, 128],
]

baseline = (
    "nvprof " +
    "./build/tools/profiler/cutlass_profiler " +
    "--operation=Gemm --m={M} --n={N} --k={K} --beta=0 " +
    "--A={DTYPE}:t --B={DTYPE}:n --accumulator-type={DTYPE} " +
    "--batch-count={BATCH_COUNT}"
)

cutlass = (
    "./build/tools/profiler/cutlass_profiler " +
    "--operation=Gemm --m={M} --n={N} --k={K} --beta=0 " +
    "--A={DTYPE}:t --B={DTYPE}:n --accumulator-type={DTYPE} " +
    "--batch-count={BATCH_COUNT} " +
    "--output=report_batch"
)

for workload in bert_batch_matmul_workloads:
    for dtype in ["f32", "f16"]:
        print("=" * 80, flush=True)
        print("workload: ", workload, flush=True)
        print("dtype: ", dtype, flush=True)
        batch_count1, batch_count2, m, n, k = workload
        batch_count = max(batch_count1, batch_count2)
        values = {
            "BATCH_COUNT": batch_count,
            "M": n,
            "N": m,
            "K": k,
            "DTYPE": dtype,
        }
        cmd = baseline.format(
            **values
        ).strip()
        subprocess.run(cmd.split())
        cmd = cutlass.format(
            **values
        ).strip()
        subprocess.run(cmd.split())
        best_row = None
        with open('report_batch.gemm.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if best_row is None or float(best_row["Runtime"]) > float(row["Runtime"]):
                    best_row = row
        print("cutlass best: ", flush=True)
        print("workload: ", workload, flush=True)
        print("dtype: ", dtype, flush=True)
        print(best_row["Operation"], flush=True)
        print("time: ", best_row["Runtime"], flush=True)
