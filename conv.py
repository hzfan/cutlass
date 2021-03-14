#!/usr/bin/env python

import argparse
import os
import subprocess
import csv

vgg_conv_workloads = [
    # [32, 3, 64, 3, 3, 224, 224, 1, 1],
    [32, 64, 128, 3, 3, 112, 112, 1, 1],
    [32, 128, 256, 3, 3, 56, 56, 1, 1],
    [32, 256, 256, 3, 3, 56, 56, 1, 1],
    [32, 256, 512, 3, 3, 28, 28, 1, 1],
    [32, 512, 512, 3, 3, 28, 28, 1, 1],
    [32, 512, 512, 3, 3, 14, 14, 1, 1]
]

split_k_slices = [1, 2, 4, 8]
split_k_mode=["serial", "parallel"]
# split_k_slices = [1]
# split_k_mode=["serial"]

baseline = (
    "nvprof " +
    "./build/tools/profiler/cutlass_profiler " +
    "--operation=Conv2d " +
    "--Activation={DTYPE}:nhwc --Filter={DTYPE}:nhwc --Output={DTYPE} --accumulator-type=f32 " +
    "--n=32 --h={H} --w={W} --c={C} --k={K} --r=3 --s=3 " +
    "--pad_h=1 --pad_w=1 " +
    "--stride::h=1 --stride::w=1 " +
    "--dilation::h=1 --dilation::w=1 " +
    "--conv_kind={CONV_KIND} " +
    "--verification-providers=cudnn " +
    "--warps_k=1 "
)

cutlass = (
    "./build/tools/profiler/cutlass_profiler " +
    "--operation=Conv2d " +
    "--Activation={DTYPE}:nhwc --Filter={DTYPE}:nhwc --Output={DTYPE} --accumulator-type=f32 " +
    "--n=32 --h={H} --w={W} --c={C} --k={K} --r=3 --s=3 " +
    "--pad_h=1 --pad_w=1 " +
    "--stride::h=1 --stride::w=1 " +
    "--dilation::h=1 --dilation::w=1 " +
    "--conv_kind={CONV_KIND} " +
    "--warps_k=1 " +
    "--split-k-slices=1,2,4,8 " +
    "--split-k-mode=serial,parallel " +
    "--output=report"
)

for workload in vgg_conv_workloads:
    for dtype in ["f32", "f16"]:
        for conv_kind in ["fprop", "wgrad", "dgrad"]:
            print("=" * 80, flush=True)
            print("workload: ", workload, flush=True)
            print("conv_kind: ", conv_kind, flush=True)
            print("dtype: ", dtype, flush=True)
            N, C, K, R, S, H, W, stride, padding = workload
            values = {
                "H" : H,
                "W" : W,
                "C" : C,
                "K" : K,
                "CONV_KIND" : conv_kind,
                "DTYPE" : dtype,
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
            with open('report.conv2d.csv', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if best_row is None or float(best_row["Runtime"]) > float(row["Runtime"]):
                        best_row = row
            print("cutlass best: ", flush=True)
            print("workload: ", workload, flush=True)
            print("conv_kind: ", conv_kind, flush=True)
            print("dtype: ", dtype, flush=True)
            print(best_row["Operation"], flush=True)
            print("split_k_mode: ", best_row["split_k_mode"], flush=True)
            print("split_k_slices: ", best_row["split_k_slices"], flush=True)
            print("time: ", best_row["Runtime"], flush=True)
