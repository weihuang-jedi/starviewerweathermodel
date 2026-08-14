#!/usr/bin/env python3
"""
check_netcdf_nans.py
--------------------
Audits newly generated NetCDF files for NaNs, Infs, and unhandled FillValues.
Can inspect a single NetCDF file or recursively scan an entire directory.
"""

import argparse
import glob
import os
import sys
import numpy as np
import xarray as xr


def check_file(filepath: str, verbose: bool = False) -> tuple[int, dict]:
    """
    Scans a single NetCDF file variable by variable.
    Returns: (total_nans_in_file, dict_of_nan_counts_per_var)
    """
    nan_summary = {}
    total_nans = 0

    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        print(f"[ERROR] Could not open file '{filepath}': {e}")
        return -1, {}

    print(f"\nScanning: {filepath}")
    print("-" * 70)

    for var_name in ds.data_vars:
        data = ds[var_name].values
        
        # Count NaNs and Infs
        n_nan = int(np.isnan(data).sum())
        n_inf = int(np.isinf(data).sum())
        n_bad = n_nan + n_inf

        nan_summary[var_name] = n_bad
        total_nans += n_bad

        if n_bad > 0 or verbose:
            total_elements = data.size
            pct_bad = (n_bad / total_elements) * 100 if total_elements > 0 else 0.0
            status = "❌ HAS NaNs/Infs" if n_bad > 0 else "✅ CLEAN"
            print(
                f"  ├─ {var_name:25s} | Shape: {str(data.shape):20s} | "
                f"NaN/Inf Count: {n_bad:10d} ({pct_bad:6.2f}%) | {status}"
            )

    ds.close()

    if total_nans == 0:
        print(f"  └─ File Status: ✅ PASSED (0 NaNs/Infs found)")
    else:
        print(f"  └─ File Status: ❌ FAILED ({total_nans} total NaNs/Infs found!)")

    return total_nans, nan_summary


def main():
    parser = argparse.ArgumentParser(description="Audit newly generated NetCDF files for NaNs and Infs.")
    parser.add_argument("-i", "--input", help="Single NetCDF file path")
    parser.add_argument("-d", "--dir", help="Directory containing NetCDF files")
    parser.add_argument("-p", "--pattern", default="*.nc", help="File matching pattern (default: '*.nc')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print status for clean variables as well")

    args = parser.parse_args()

    if not args.input and not args.dir:
        parser.error("Must provide either -i/--input or -d/--dir")

    files_to_check = []
    if args.input:
        files_to_check.append(args.input)
    elif args.dir:
        files_to_check = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
        if not files_to_check:
            # Try recursive search if glob pattern didn't catch subdirectories
            files_to_check = sorted(glob.glob(os.path.join(args.dir, "**", args.pattern), recursive=True))

    if not files_to_check:
        print(f"[ERROR] No NetCDF files found matching pattern '{args.pattern}'")
        sys.exit(1)

    print("=" * 80)
    print(f" NETCDF NAN AUDIT ENGINE | Checking {len(files_to_check)} file(s)")
    print("=" * 80)

    corrupted_files = 0
    total_files = len(files_to_check)

    for filepath in files_to_check:
        total_nans, _ = check_file(filepath, verbose=args.verbose)
        if total_nans != 0:
            corrupted_files += 1

    print("\n" + "=" * 80)
    print(" SUMMARY REPORT")
    print("=" * 80)
    print(f" Total Files Audited : {total_files}")
    print(f" Clean Files (0 NaNs): {total_files - corrupted_files}")
    print(f" Corrupted Files     : {corrupted_files}")

    if corrupted_files > 0:
        print("\n❌ RESULT: NaNs detected! Do NOT proceed to Zarr packing until files are fixed.")
        sys.exit(1)
    else:
        print("\n✅ RESULT: All files are completely clean! Ready for Zarr packing.")
        sys.exit(0)


if __name__ == "__main__":
    main()
