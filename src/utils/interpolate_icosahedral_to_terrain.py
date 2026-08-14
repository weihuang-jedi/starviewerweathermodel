#!/usr/bin/env python3
"""
interpolate_icosahedral_to_terrain.py
-------------------------------------
Transforms an unstructured terrain-icosahedral NetCDF dataset back onto a regular
latitude-longitude terrain-height grid, using a reference terrain-regular NetCDF file
as the target spatial coordinate template.

Converts log-state variables (ln_t, ln_p, ln_rho) back to physical units (T, p, rho),
interpolates dynamic fields (T, p, q, u, v, w, rho, h) level-by-level, and produces
CF-compliant regular NetCDF outputs ready for downstream verification and analysis.
"""

import argparse
import glob
import os
import sys
import warnings
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# Silence harmless xarray / netCDF4 warnings
warnings.filterwarnings("ignore")


class IcosahedralToTerrainPipeline:
    """
    Backward transformation engine: Icosahedral Mesh -> Regular Lat-Lon Terrain Grid.
    """
    def __init__(self, reference_regular_nc: str):
        self.reference_nc_path = reference_regular_nc
        self.ref_ds = None
        self.target_lats = None
        self.target_lons = None
        self.lat_key = 'latitude'
        self.lon_key = 'longitude'
        self._load_reference_grid()

    def _load_reference_grid(self) -> None:
        if not os.path.exists(self.reference_nc_path):
            raise FileNotFoundError(f"[ERROR] Reference regular NetCDF template not found: '{self.reference_nc_path}'")

        print(f"[REVERSE PIPELINE] Loading spatial grid template from: {self.reference_nc_path}")
        self.ref_ds = xr.open_dataset(self.reference_nc_path)

        self.lat_key = 'latitude' if 'latitude' in self.ref_ds.coords else 'lat'
        self.lon_key = 'longitude' if 'longitude' in self.ref_ds.coords else 'lon'

        self.target_lats = self.ref_ds[self.lat_key].values
        self.target_lons = self.ref_ds[self.lon_key].values

        # Build 2D target regular grid meshgrid
        self.grid_lons, self.grid_lats = np.meshgrid(self.target_lons, self.target_lats)
        self.target_shape = self.grid_lats.shape  # (n_lats, n_lons)

    def _prepare_src_nodes(self, src_lons: np.ndarray, src_lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Appends circular longitude wrapped nodes at 360 deg and -360 deg to bridge the Date Line seam.
        """
        clean_lons = np.mod(src_lons, 360.0)

        # Append shifted nodes near 0 and 360 to prevent interpolation gaps along the Prime Meridian
        mask_near_zero = clean_lons < 30.0
        mask_near_360 = clean_lons > 330.0

        wrapped_lons = np.concatenate([clean_lons, clean_lons[mask_near_zero] + 360.0, clean_lons[mask_near_360] - 360.0])
        wrapped_lats = np.concatenate([src_lats, src_lats[mask_near_zero], src_lats[mask_near_360]])

        src_points = np.column_stack([wrapped_lons, wrapped_lats])
        return src_points, np.concatenate([np.arange(len(src_lons)), np.where(mask_near_zero)[0], np.where(mask_near_360)[0]])

    def process_file(self, icosahedral_nc: str, output_nc: str) -> None:
        print(f"\n[REVERSE PIPELINE] Ingesting Icosahedral Dataset: {icosahedral_nc}")
        ds_ico = xr.open_dataset(icosahedral_nc)

        # Extract icosahedral node coordinates
        ico_lons = ds_ico['longitude'].values
        ico_lats = ds_ico['latitude'].values

        # Build circular padded interpolation point cloud
        src_points, node_index_map = self._prepare_src_nodes(ico_lons, ico_lats)

        # Target query coordinates (ensure longitudes in [0, 360])
        target_query_lons = np.mod(self.grid_lons, 360.0)
        query_points = np.column_stack([target_query_lons.ravel(), self.grid_lats.ravel()])

        print("  -> Building Delaunay Triangulator over icosahedral node positions...")
        # Primary linear interpolator with nearest-neighbor fallback for outer pole edges
        nearest_interp = NearestNDInterpolator(src_points, node_index_map)

        # Detect dynamic level dimension
        vert_dim = 'level' if 'level' in ds_ico.dims else ('height' if 'height' in ds_ico.dims else None)
        num_levels = ds_ico.sizes.get(vert_dim, 32) if vert_dim else 32

        output_vars = {}

        # ---------------------------------------------------------------------
        # 1. Transform Log-States Back to Physical Variables (T, p, rho)
        # ---------------------------------------------------------------------
        print("  -> Decoding log-state variables (ln_t, ln_p, ln_rho) to physical units (T, p, rho)...")

        if 'ln_t_icosahedral' in ds_ico:
            t_k_ico = np.exp(ds_ico['ln_t_icosahedral'].values)
        else:
            t_k_ico = None

        if 'ln_p_icosahedral' in ds_ico:
            p_pa_ico = np.exp(ds_ico['ln_p_icosahedral'].values)
            p_hpa_ico = p_pa_ico / 100.0  # Convert Pa -> hPa
        else:
            p_hpa_ico = None

        if 'ln_rho_icosahedral' in ds_ico:
            rho_ico = np.exp(ds_ico['ln_rho_icosahedral'].values)
        else:
            rho_ico = None

        # Helper mapping for dynamic level fields
        field_payload = {}
        if t_k_ico is not None:
            field_payload['t'] = (t_k_ico, "Air Temperature", "K", "air_temperature")
        if p_hpa_ico is not None:
            field_payload['p'] = (p_hpa_ico, "Atmospheric Pressure at Target Level", "hPa", "air_pressure")
        if rho_ico is not None:
            field_payload['rho'] = (rho_ico, "Air Density", "kg m**-3", "air_density")

        # Add remaining dynamic variables (u, v, w, q, h)
        passthrough_keys = {
            'u_icosahedral': ('u', "Eastward Wind Component", "m s**-1", "eastward_wind"),
            'v_icosahedral': ('v', "Northward Wind Component", "m s**-1", "northward_wind"),
            'w_icosahedral': ('w', "Vertical Velocity", "Pa s**-1", "lagrangian_tendency_of_air_pressure"),
            'q_icosahedral': ('q', "Specific Humidity", "kg kg**-1", "specific_humidity"),
            'h_icosahedral': ('h', "3D Terrain-Following Height Above Sea Level", "meters", "height")
        }

        for ico_key, (std_key, long_name, units, std_name) in passthrough_keys.items():
            if ico_key in ds_ico:
                field_payload[std_key] = (ds_ico[ico_key].values, long_name, units, std_name)

        # ---------------------------------------------------------------------
        # 2. Interpolate Dynamic 3D Fields Level-by-Level
        # ---------------------------------------------------------------------
        for std_key, (ico_data, long_name, units, std_name) in field_payload.items():
            print(f"  -> Interpolating dynamic field to regular grid: '{std_key}' ({long_name})")

            if ico_data.ndim == 2:  # [level, node]
                reg_3d = np.zeros((num_levels, self.target_shape[0], self.target_shape[1]), dtype=np.float32)

                for k in range(num_levels):
                    level_vals = ico_data[k]
                    padded_vals = level_vals[node_index_map]

                    lin_interp = LinearNDInterpolator(src_points, padded_vals)
                    interp_flat = lin_interp(query_points)

                    nan_mask = np.isnan(interp_flat)
                    if np.any(nan_mask):
                        nearest_interp = NearestNDInterpolator(src_points, padded_vals)
                        interp_flat[nan_mask] = nearest_interp(query_points[nan_mask])

                    reg_3d[k] = interp_flat.reshape(self.target_shape)

                # Specific Humidity non-negativity constraint
                if std_key == 'q':
                    reg_3d = np.clip(reg_3d, 0.0, None)

                output_vars[std_key] = (
                    [vert_dim or 'level', self.lat_key, self.lon_key],
                    reg_3d,
                    {"long_name": long_name, "units": units, "standard_name": std_name}
                )

            elif ico_data.ndim == 1:  # 2D Surface fields [node]
                padded_vals = ico_data[node_index_map]
                lin_interp = LinearNDInterpolator(src_points, padded_vals)
                interp_flat = lin_interp(query_points)

                nan_mask = np.isnan(interp_flat)
                if np.any(nan_mask):
                    interp_flat[nan_mask] = padded_vals[nearest_interp(query_points[nan_mask])]

                output_vars[std_key] = (
                    [self.lat_key, self.lon_key],
                    interp_flat.reshape(self.target_shape),
                    {"long_name": long_name, "units": units, "standard_name": std_name}
                )

        # ---------------------------------------------------------------------
        # 3. Interpolate Surface Elevation / Topography (h_terrain)
        # ---------------------------------------------------------------------
        if 'h_terrain' in self.ref_ds:
            output_vars['h_terrain'] = (
                [self.lat_key, self.lon_key],
                self.ref_ds['h_terrain'].values,
                self.ref_ds['h_terrain'].attrs
            )
        elif 'h_terrain_icosahedral' in ds_ico:
            padded_vals = ds_ico['h_terrain_icosahedral'].values[node_index_map]
            lin_interp = LinearNDInterpolator(src_points, padded_vals)
            interp_flat = lin_interp(query_points)

            nan_mask = np.isnan(interp_flat)
            if np.any(nan_mask):
                interp_flat[nan_mask] = padded_vals[nearest_interp(query_points[nan_mask])]

            output_vars['h_terrain'] = (
                [self.lat_key, self.lon_key],
                interp_flat.reshape(self.target_shape),
                {"long_name": "Surface Topography Elevation from ETOPO2022", "units": "meters"}
            )

        # ---------------------------------------------------------------------
        # 4. Construct Output Regular Dataset
        # ---------------------------------------------------------------------
        coords_out = {
            self.lat_key: self.target_lats,
            self.lon_key: self.target_lons
        }

        if vert_dim and vert_dim in self.ref_ds.coords:
            coords_out[vert_dim] = self.ref_ds[vert_dim].values
        elif vert_dim and vert_dim in ds_ico.coords:
            coords_out[vert_dim] = ds_ico[vert_dim].values
        else:
            coords_out['level'] = np.arange(1, num_levels + 1, dtype=np.int32)

        for meta_coord in ['eta', 'target_level']:
            if meta_coord in self.ref_ds.coords:
                coords_out[meta_coord] = self.ref_ds[meta_coord]
            elif meta_coord in ds_ico.coords:
                coords_out[meta_coord] = ds_ico[meta_coord]

        for time_coord in ['time', 'step', 'valid_time']:
            if time_coord in ds_ico.coords:
                coords_out[time_coord] = ds_ico[time_coord].values

        ds_out = xr.Dataset(
            data_vars=output_vars,
            coords=coords_out,
            attrs={
                "title": "AIDA Meteorological Data Interpolated Back to Regular Terrain Grid",
                "source_icosahedral_file": os.path.basename(icosahedral_nc),
                "reference_template_file": os.path.basename(self.reference_nc_path),
                "terrain_formula": "h = Hmax - eta * (Hmax - Hterrain)",
                "conventions": "CF-1.8"
            }
        )

        os.makedirs(os.path.dirname(output_nc) or ".", exist_ok=True)
        ds_out.to_netcdf(output_nc, format="NETCDF4")

        ds_ico.close()
        ds_out.close()
        print(f"  -> Successfully generated clean regular terrain NetCDF: '{output_nc}'")


def main():
    parser = argparse.ArgumentParser(description="Reverse Interpolator: Terrain-Icosahedral -> Regular Lat-Lon Terrain NetCDF")
    parser.add_argument("-i", "--input", help="Single input icosahedral NetCDF file")
    parser.add_argument("--input_dir", help="Directory containing icosahedral NetCDF files")
    parser.add_argument("-r", "--ref", required=True, help="Path to reference regular terrain-height NetCDF file (grid template)")
    parser.add_argument("-o", "--output", help="Destination path for single output NetCDF file")
    parser.add_argument("--output_dir", help="Destination directory for batch output NetCDF files")
    parser.add_argument("--pattern", default="*.nc", help="File pattern to match in --input_dir")

    args = parser.parse_args()

    pipeline = IcosahedralToTerrainPipeline(reference_regular_nc=args.ref)

    if args.input and args.output:
        pipeline.process_file(args.input, args.output)
    elif args.input_dir and args.output_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not files:
            print(f"[ERROR] No files matching '{args.pattern}' found in '{args.input_dir}'")
            return
        for f in files:
            out_file = os.path.join(args.output_dir, os.path.basename(f))
            pipeline.process_file(f, out_file)
    else:
        parser.error("Must supply either (-i/--input and -o/--output) or (--input_dir and --output_dir)")


if __name__ == "__main__":
    main()
