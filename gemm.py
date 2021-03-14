#!/usr/bin/env python

import argparse
import os
import subprocess
import csv

vgg_matmul_workloads = [
    [32, 4096, 25088],
    [32, 4096, 4096],
    [32, 1000, 4096],
    [32, 4096, 1000],
    [32, 25088, 4096],
    [4096, 25088, 32],
    [4096, 4096, 32],
    [1000, 4096, 32],
]

baseline = (
    "nvprof " +
    "./build/tools/profiler/cutlass_profiler " + 
    "--operation=Gemm --m={M} --n={N} --k={K} --beta=0 " +
    "--A={DTYPE}:t --B={DTYPE}:n --accumulator-type={DTYPE} " +
    "--verification-providers=cublas"
)

cutlass = (
    "./build/tools/profiler/cutlass_profiler " + 
    "--operation=Gemm --m={M} --n={N} --k={K} --beta=0 " +
    "--A={DTYPE}:t --B={DTYPE}:n --accumulator-type={DTYPE} " +
    "--verification-enabled=false " +
    "--split-k-slices=1,2,4,8 " +
    "--output=report"
)

for workload in vgg_matmul_workloads:
    for dtype in ["f32", "f16"]:
        print("=" * 80, flush=True)
        print("workload: ", workload, flush=True)
        print("dtype: ", dtype, flush=True)
        m, n, k = workload
        values = {
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
        with open('report.gemm.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if best_row is None or float(best_row["Runtime"]) > float(row["Runtime"]):
                    best_row = row
        print("cutlass best: ", flush=True)
        print("workload: ", workload, flush=True)
        print("dtype: ", dtype, flush=True)
        print(best_row["Operation"], flush=True)
        print("split_k_slices: ", best_row["split_k_slices"], flush=True)
        print("time: ", best_row["Runtime"], flush=True)
