import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

FIXED_DEPTH_M = 50.0

@jax.jit
def _region_kernel(theta, uo, vo, wo, hfds, dx, dy, dz, seconds):
    """Compute a complete time/latitude/longitude region with JAX."""
    nz = dz.shape[0]
    tm = theta[:, :nz, 1:-1, 1:-1]
    tn, ts = theta[:, :nz, 2:, 1:-1], theta[:, :nz, :-2, 1:-1]
    te, tw = theta[:, :nz, 1:-1, 2:], theta[:, :nz, 1:-1, :-2]
    ute = .5 * (uo[:, :nz, 2:, 2:] + uo[:, :nz, 1:-1, 2:]) * (te - tm)
    utw = .5 * (uo[:, :nz, 2:, 1:-1] + uo[:, :nz, 1:-1, 1:-1]) * (tm - tw)
    vtn = .5 * (vo[:, :nz, 2:, 2:] + vo[:, :nz, 2:, 1:-1]) * (tn - tm)
    vts = .5 * (vo[:, :nz, 1:-1, 2:] + vo[:, :nz, 1:-1, 1:-1]) * (tm - ts)
    depth, weights = dz.sum(), dz[None, :, None, None]
    ute = (ute * weights).sum(1) / (dx[None] * depth)
    utw = (utw * weights).sum(1) / (dx[None] * depth)
    vtn = (vtn * weights).sum(1) / (dy[None] * depth)
    vts = (vts * weights).sum(1) / (dy[None] * depth)
    temp = (tm * weights).sum(1) / depth
    tend = jnp.gradient(temp, seconds, axis=0)
    wtb = wo * (theta[:, nz - 1, 1:-1, 1:-1] -
                theta[:, nz, 1:-1, 1:-1]) / depth
    shf = hfds / (1025. * 3996. * depth)
    uti, vti = .5 * (ute + utw), .5 * (vtn + vts)
    adv = -(uti + vti + wtb)
    return temp, wtb, uti, vti, wtb, tend, adv, shf, adv + shf, ute, utw, vtn, vts


def calculate_climate_metrics_region(ilon_start, ilon_stop, ilat_start,
                                     ilat_stop, nlev, ds_t, ds_u, ds_v,
                                     ds_w, ds_hf):
    """Calculate an inclusive rectangular region in one compiled JAX call."""
    if ds_t.sizes["time"] < 2:
        raise ValueError("at least two time samples are required")
    if nlev < 1 or nlev >= ds_t.sizes["lev"]:
        raise ValueError("nlev must leave one extra thetao level")
    if (ilon_start < 1 or ilon_stop >= ds_t.sizes["i"] - 1 or
            ilat_start < 1 or ilat_stop >= ds_t.sizes["j"] - 1):
        raise ValueError("region must leave a one-cell horizontal halo")
    halo = dict(i=slice(ilon_start - 1, ilon_stop + 2),
                j=slice(ilat_start - 1, ilat_stop + 2),
                lev=slice(0, nlev + 1))

    def array4(da):
        data = da.isel(**halo).transpose("time", "lev", "j", "i").values
        return jnp.asarray(data)

    theta, uo, vo = array4(ds_t.thetao), array4(ds_u.uo), array4(ds_v.vo)
    centre = dict(i=slice(ilon_start, ilon_stop + 1),
                  j=slice(ilat_start, ilat_stop + 1))
    # The bottom flux belongs at the requested fixed 50 m interface.  This is
    # exact on CM2 and interpolates between adjacent interfaces when needed.
    wo = jnp.asarray(ds_w.wo.interp(lev=FIXED_DEPTH_M).isel(**centre)
                     .transpose("time", "j", "i").values)
    hfds = jnp.asarray(ds_hf.hfds.isel(**centre)
                       .transpose("time", "j", "i").values)
    lon, lat = np.asarray(ds_t.longitude), np.asarray(ds_t.latitude)
    dlon = np.pad(np.diff(lon, axis=1), ((0, 0), (0, 1)), mode="edge")
    dlat = np.pad(np.diff(lat, axis=0), ((0, 1), (0, 0)), mode="edge")
    dx_all = (6371000. * np.deg2rad(dlon) *
              np.cos(np.deg2rad(lat.mean(1)[:, None])))
    dy_all = 6371000. * np.deg2rad(dlat)
    region = np.s_[ilat_start:ilat_stop + 1, ilon_start:ilon_stop + 1]
    bounds_name = ds_t.lev.attrs.get("bounds", "lev_bnds")
    bounds = np.asarray(ds_t[bounds_name].isel(lev=slice(0, nlev)))
    dz_np = np.maximum(0.0, np.minimum(bounds[:, 1], FIXED_DEPTH_M) -
                       np.minimum(bounds[:, 0], FIXED_DEPTH_M))
    if not np.isclose(dz_np.sum(), FIXED_DEPTH_M):
        raise ValueError(f"nlev={nlev} covers only {dz_np.sum()} m, not fixed 50 m")
    dz = jnp.asarray(dz_np)
    seconds = np.asarray((ds_t.time.values - ds_t.time.values[0]) / np.timedelta64(1, "s"), dtype=np.float64)
    values = _region_kernel(theta, uo, vo, wo, hfds,
                            jnp.asarray(dx_all[region]),
                            jnp.asarray(dy_all[region]), dz,
                            jnp.asarray(seconds))
    names = ("temp_avg", "WTb", "UTi", "VTi", "WTi", "TEND", "ADV",
             "SHF", "RHS", "UTe", "UTw", "VTn", "VTs")
    coords = {"time": ds_t.time.values,
              "lat": lat[ilat_start:ilat_stop + 1, 0],
              "lon": lon[0, ilon_start:ilon_stop + 1]}
    result = xr.Dataset({name: (("time", "lat", "lon"), np.asarray(value))
                         for name, value in zip(names, values)}, coords=coords)
    for name in names:
        result[name].attrs["units"] = "degC" if name == "temp_avg" else "degC s-1"
    result.attrs["fixed_integration_depth_m"] = FIXED_DEPTH_M
    result.attrs["vertical_weights"] = "thetao lev_bnds clipped exactly at 50 m"
    return result



def find_closest_lat_lon(ds_t, target_lon, target_lat):
    # Calculate the absolute difference between each longitude, latitude, and the target value
    lon_lim = ds_t.longitude[0, :].size - 2
    lat_lim = ds_t.latitude[:, 0].size - 2
    abs_diff_lon = np.abs(ds_t.longitude[0, :] - target_lon)
    abs_diff_lat = np.abs(ds_t.latitude[:, 0] - target_lat)

    # Find the index of the minimum difference, which corresponds to the closest value
    ilon = abs_diff_lon.argmin().item()
    ilat = abs_diff_lat.argmin().item()

    # Check and adjust ilon
    if ilon < 1 or ilon > lon_lim:
        raise ValueError("ilon is not in the range")
    ilon = max(1, min(ilon, lon_lim))

    # Check and adjust ilat
    if ilat < 1 or ilat > lat_lim:
        raise ValueError("ilat is not in the range")
    ilat = max(1, min(ilat, lat_lim))

    return ilon, ilat

print("Functions successully loaded")


