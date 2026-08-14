#!/usr/bin/env python3
"""
pack_icosahedral_zarr.py
------------------------
An object-oriented packaging engine designed to compile large multi-file collections
of 3D terrain-following unstructured icosahedral weather NetCDF files into an optimized,
cloud-native, and ML-ready consolidated Zarr dataset for AIDA GNN surrogate models.
"""

import argparse
import glob
import os
import warnings
import xarray as xr
import zarr
from numcodecs import Blosc

# Silence library notifications
warnings.filterwarnings("ignore")


class IcosahedralToZarrPacker:
    """
    Consolidates multi-file terrain-following icosahedral NetCDF datasets into Zarr format.
    """
    def __init__(self, file_pattern: str, output_zarr_path: str, time_chunk_size: int = 32):
        """
        Args:
            file_pattern (str): Shell glob pattern matching source NetCDF files
            output_zarr_path (str): Target directory destination path for the compiled Zarr store
            time_chunk_size (int): Temporal sequence step length inside each chunk (default: 32)
        """
        self.file_pattern = file_pattern
        self.output_zarr_path = output_zarr_path
        self.time_chunk_size = time_chunk_size
        self.ds = None

    def execute_conversion(self, compression_level: int = 3) -> None:
        """
        Loads files lazily, builds the targeted multi-dimensional chunk graph, binds
        high-performance Blosc compression maps, and writes the Zarr store.
        """
        matched_files = sorted(glob.glob(self.file_pattern))
        if not matched_files:
            raise FileNotFoundError(f"[ERROR] No files matched the pattern: '{self.file_pattern}'")

        print(f"[STAGE 1] Resolving {len(matched_files)} netCDF files with pattern: {self.file_pattern}")

        # Open files along the time dimension lazily
        self.ds = xr.open_mfdataset(
            matched_files,
            concat_dim="time",
            combine="nested",
            data_vars="minimal",
            coords="minimal",
            compat="override"
        )

        print("[STAGE 2] Enforcing training chunk boundaries (time, level, node alignment)...")

        # Determine vertical dimension name ('level' or 'height')
        vert_dim = 'level' if 'level' in self.ds.dims else ('height' if 'height' in self.ds.dims else None)

        # Enforce time chunking on meteorological fields while keeping spatial and vertical dimensions unified
        chunk_spec = {'time': self.time_chunk_size}
        if vert_dim and vert_dim in self.ds.dims:
            chunk_spec[vert_dim] = -1
        if 'node' in self.ds.dims:
            chunk_spec['node'] = -1
        if 'face' in self.ds.dims:
            chunk_spec['face'] = -1

        self.ds = self.ds.chunk(chunk_spec)

        print(f"[STAGE 3] Building Blosc ZStandard encoding profiles (effort level={compression_level})...")
        # Configure high-efficiency BitShuffle compression mapping used by modern deep learning architectures
        compressor = Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
        
        # Apply encoding to data variables
        encoding = {}
        for var in self.ds.data_vars:
            encoding[var] = {'compressor': compressor}

        print(f"[STAGE 4] Writing consolidated Zarr warehouse to: {self.output_zarr_path}...")
        
        # Clean up target directory if an incomplete write exists
        if os.path.exists(self.output_zarr_path):
            print(f" -> Removing existing Zarr directory at '{self.output_zarr_path}' for clean rewrite.")
            import shutil
            shutil.rmtree(self.output_zarr_path)

        self.ds.to_zarr(
            self.output_zarr_path,
            mode='w',
            encoding=encoding,
            consolidated=True
        )

        self.ds.close()
        print(f"SUCCESS: Consolidated Multi-Level Terrain-Following Icosahedral Zarr compilation finalized at '{self.output_zarr_path}'!\n")


# =====================================================================
# CLI SCRIPT INTERFACE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="High-Speed Multi-Temporal and Multi-Level Icosahedral NetCDF to Compressed Consolidated Zarr Packer."
    )
    parser.add_argument("-i", "--input", default="icosahedral-grid/icosahedral_logstate_m4.202*.nc",
                        help="Input shell glob matching NetCDF files (default: icosahedral-grid/icosahedral_logstate_m4.202*.nc)")
    parser.add_argument("-o", "--output", default="icosahedral_logstate.zarr",
                        help="Output Zarr path target directory (default: icosahedral_logstate.zarr)")
    parser.add_argument("-c", "--chunk_size", type=int, default=32,
                        help="Time dimension sequence array chunk size limits (default: 32)")
    parser.add_argument("-l", "--level", type=int, default=3,
                        help="Zstd compressor effort configuration setting [1-9] (default: 3)")

    args = parser.parse_args()

    packer = IcosahedralToZarrPacker(
        file_pattern=args.input,
        output_zarr_path=args.output,
        time_chunk_size=args.chunk_size
    )
    packer.execute_conversion(compression_level=args.level)


if __name__ == "__main__":
    main()
