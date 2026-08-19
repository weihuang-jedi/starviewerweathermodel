#!/usr/bin/env python3
"""
plot_terrain_regular_panel.py
-----------------------------
Generates a 6-panel diagnostic visualization (T, p, q, u, v, w) on a global
Robinson projection for a specified vertical level index from a 2D regular
lat-lon terrain-following NetCDF dataset.
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point


def plot_regular_terrain_panel(nc_file: str, level_idx: int = 0, output_png: str = None, show = False):
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"[ERROR] Input NetCDF file not found: '{nc_file}'")

    print(f"[PLOTTER] Opening NetCDF dataset: {nc_file}")
    ds = xr.open_dataset(nc_file)

    num_levels = ds.sizes.get('level', 32)
    if level_idx < 0 or level_idx >= num_levels:
        raise ValueError(f"[ERROR] Requested level_idx {level_idx} out of range [0, {num_levels - 1}].")

    # Extract coordinates
    lat_key = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_key = 'longitude' if 'longitude' in ds.coords else 'lon'

    lats = ds[lat_key].values
    lons = ds[lon_key].values

    # Ensure latitude is ascending (-90 to +90) for plotting
    if lats[0] > lats[-1]:
        print(f"[PLOTTER] Reversing descending latitude axis [{lat_key}] for Cartopy alignment...")
        ds = ds.sortby(lat_key)
        lats = ds[lat_key].values

    # Read dynamic fields for requested level
    t_k = ds['t'].isel(level=level_idx).values        # Temperature (K)
    p_hpa = ds['p'].isel(level=level_idx).values      # Pressure (hPa)
    q_kgkg = ds['q'].isel(level=level_idx).values     # Specific humidity (kg/kg)
    q_gkg = q_kgkg * 1000.0                           # Convert to g/kg
    u_ms = ds['u'].isel(level=level_idx).values       # Zonal wind (m/s)
    v_ms = ds['v'].isel(level=level_idx).values       # Meridional wind (m/s)
    w_pas = ds['w'].isel(level=level_idx).values      # Vertical velocity (Pa/s)

    # Extract height metadata for title
    if 'h' in ds:
        h_mean = float(np.nanmean(ds['h'].isel(level=level_idx).values))
        level_str = f"Level {level_idx + 1}/{num_levels} (~{h_mean:.0f} m ASL)"
    else:
        level_str = f"Level {level_idx + 1}/{num_levels}"

    print(f"[PLOTTER] Rendering 6-panel figure for {level_str}...")

    # Set up 2x3 Subplot Grid using Cartopy Robinson Projection
    proj = ccrs.Robinson(central_longitude=0.0)
    data_proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={'projection': proj})
    axes = axes.flatten()

    # Panel Configuration Dictionary
    panels = [
        {
            'data': t_k,
            'title': 'Temperature (T)',
            'units': 'K',
            'cmap': 'coolwarm',
            'ax_idx': 0
        },
        {
            'data': p_hpa,
            'title': 'Pressure (p)',
            'units': 'hPa',
            'cmap': 'viridis_r',
            'ax_idx': 1
        },
        {
            'data': q_gkg,
            'title': 'Specific Humidity (q)',
            'units': 'g/kg',
            'cmap': 'Blues',
            'ax_idx': 2
        },
        {
            'data': u_ms,
            'title': 'Zonal Wind (u)',
            'units': 'm/s',
            'cmap': 'RdBu_r',
            'ax_idx': 3
        },
        {
            'data': v_ms,
            'title': 'Meridional Wind (v)',
            'units': 'm/s',
            'cmap': 'RdBu_r',
            'ax_idx': 4
        },
        {
            'data': w_pas,
            'title': 'Vertical Velocity (w)',
            'units': 'Pa/s',
            'cmap': 'seismic',
            'ax_idx': 5
        },
    ]

    for p in panels:
        ax = axes[p['ax_idx']]
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', edgecolor='gray')
        ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')

        # Add cyclic point along longitude to eliminate Prime Meridian white line seam
        data_cyclic, lons_cyclic = add_cyclic_point(p['data'], coord=lons)

        # Filled contour plot
        cntr = ax.contourf(
            lons_cyclic,
            lats,
            data_cyclic,
            levels=40,
            cmap=p['cmap'],
            transform=data_proj,
            extend='both'
        )

        ax.set_title(f"{p['title']} [{p['units']}]", fontsize=12, fontweight='bold', pad=8)
        cbar = fig.colorbar(cntr, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85, aspect=25)
        cbar.ax.tick_params(labelsize=9)

    file_title = os.path.basename(nc_file)
    fig.suptitle(f"Terrain-Following Height Regular Lat-Lon Grid Panel\nFile: {file_title} | {level_str}",
                 fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_png is None:
        output_png = f"regular_panel_level_{level_idx + 1:02d}_{os.path.splitext(file_title)[0]}.png"

    plt.savefig(output_png, dpi=250, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    ds.close()

    print(f"[SUCCESS] Multi-panel diagnostic plot generated and saved to: '{output_png}'\n")


def main():
    parser = argparse.ArgumentParser(description="Plot 6-panel (T, p, q, u, v, w) Diagnostic Figure from Regular Terrain NetCDF")
    parser.add_argument("-i", "--input", required=True, help="Path to input terrain-regular NetCDF file")
    parser.add_argument("-l", "--level", type=int, default=0, help="Vertical level index (0 to 31, default: 0)")
    parser.add_argument("-o", "--output", help="Destination path for output PNG image")
    parser.add_argument("-s", "--show", action="store_true", help="Display plot interactively")

    args = parser.parse_args()

    plot_regular_terrain_panel(
        nc_file=args.input,
        level_idx=args.level,
        output_png=args.output,
        show=args.show
    )


if __name__ == "__main__":
    main()
