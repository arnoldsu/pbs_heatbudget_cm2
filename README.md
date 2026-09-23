## Offline Heat-Budget Formulation

The analysis diagnoses the temperature budget offline from archived
ACCESS-CM2 ocean fields. The temperature tendency is written schematically as

\[
\frac{\partial T}{\partial t}
=
\mathrm{SHF}
+
\mathrm{ADV}
+
\mathrm{VDIFF}
+
\mathrm{RES},
\]

where SHF is the surface heat-flux contribution, ADV is ocean heat
advection, VDIFF represents vertical diffusion/mixing, and RES contains
unresolved processes and budget closure errors.

For a control volume, horizontal advection can alternatively be interpreted
from heat transport across the boundaries. The zonal contribution is

\[
\mathrm{ADV}_x
=
\frac{1}{V}
\left[
\int_{A_W} u_W (T_W-\bar{T})\,dA
-
\int_{A_E} u_E (T_E-\bar{T})\,dA
\right],
\]

and the meridional contribution is

\[
\mathrm{ADV}_y
=
\frac{1}{V}
\left[
\int_{A_S} v_S (T_S-\bar{T})\,dA
-
\int_{A_N} v_N (T_N-\bar{T})\,dA
\right].
\]

Thus, the commonly used shorthand

\[
u_i = u_E-u_W
\]

describes the difference between eastern and western boundary flow, but
the heat-budget contribution depends on the corresponding velocity-temperature
transport, \(uT\), rather than velocity alone.

This boundary-flux formulation follows the control-volume heat-budget
approach used for ENSO Niño-3 and Niño-4 regions by Guan and McPhaden (2016).

# ACCESS-CM2 upper-ocean heat budget with JAX

This directory calculates a monthly upper-ocean temperature budget from
ACCESS-CM2 output. The production kernel is vectorized across time, depth,
latitude, and longitude and compiled with JAX. Xarray handles NetCDF input,
coordinates, and output.

## Heat-budget calculation

At each horizontal grid cell, the code evaluates

```text
TEND = ADV + SHF
ADV  = -(UTi + VTi + WTi)
RHS  = ADV + SHF
```

All tendency terms are output in `degC s-1`. Constants are Earth radius
`R = 6,371,000 m`, seawater density `rho0 = 1,025 kg m-3`, heat capacity
`cp = 3,996 J kg-1 K-1`. TEND uses the decoded real monthly time coordinate in seconds.

### Depth and temperature tendency

For `nlev` active layers:

```text
dz[k] = lev[k+1] - lev[k]
H     = sum(dz[k])
temp_avg = sum(thetao[k]*dz[k])/H
TEND = gradient(temp_avg, time_seconds)
```

Interior records use centred differences with the real, possibly unequal month spacing; endpoints use one-sided differences. `temp_avg` is `dz` weighted. CM2 uses `nlev=5`, giving H=50 m.

### Horizontal advection

Velocity is averaged onto temperature-cell faces and multiplied by the
neighbouring temperature difference:

```text
ute[k] = 0.5*(uo_NE[k] + uo_SE[k])*(T_E[k] - T_M[k])
utw[k] = 0.5*(uo_NW[k] + uo_SW[k])*(T_M[k] - T_W[k])
vtn[k] = 0.5*(vo_EN[k] + vo_WN[k])*(T_N[k] - T_M[k])
vts[k] = 0.5*(vo_ES[k] + vo_WS[k])*(T_M[k] - T_S[k])

UTe = sum(ute[k]*dz[k])/(dx*H)
UTw = sum(utw[k]*dz[k])/(dx*H)
VTn = sum(vtn[k]*dz[k])/(dy*H)
VTs = sum(vts[k]*dz[k])/(dy*H)
UTi = 0.5*(UTe + UTw)
VTi = 0.5*(VTn + VTs)
```

Distances are estimated from the 2-D grid coordinates:

```text
dx = R * dlon * cos(mean latitude)
dy = R * dlat
```

Angular differences are converted from degrees to radians.

### Vertical advection and surface flux

```text
WTi = WTb = wo_bottom*(T[nlev-1] - T[nlev])/H
SHF = hfds/(rho0*cp*H)
ADV = -(UTi + VTi + WTi)
RHS = ADV + SHF
```

An extra temperature level, `T[nlev]`, is required for the bottom gradient.
The scripts save `UTe`, `UTw`, `VTn`, `VTs`, `UTi`, `VTi`, `WTi`, `TEND`, and
`SHF`. The region function additionally returns `temp_avg`, `WTb`, `ADV`, and
`RHS`.

## Implementation

`calculate_climate_metrics_region()` in `func_hb_ars599.py` is the production
interface. It extracts the region plus a one-cell halo, converts data to JAX
arrays, calls the JIT-compiled `_region_kernel()` once, and returns an xarray
Dataset. This removes the old horizontal loops, repeated Joblib startup, and
nested time/depth loops.

The first call includes compilation time. The production kernel now differs intentionally from the legacy point implementation: TEND uses a real-time centred derivative and temperature is dz weighted.

## Decomposition variants

| Variant | Temperature | Velocity and surface flux |
|---|---|---|
| `raw` | raw | raw |
| `aa` | anomaly | anomaly |
| `ab` | anomaly | background/bar |
| `ba` | background/bar | anomaly |
| `bb` | background/bar | background/bar |

Scripts are named `get_hb_shayne_function_v2_<variant>.py`.

## Inputs and requirements

By default, input is read from `/g/data/p66/ars599/work_mhg/cm2r1`. Override
this using `HB_INPUT_DIR`. Raw uses `_v10.nc` files for `thetao`, `uo`, `vo`,
`wo`, and `hfds`; decomposition jobs also require matching `_v10.anom.nc` and
`_v10.bar.nc` files.

Expected dimensions are:

```text
thetao, uo, vo, wo: (time, lev, j, i)
hfds:                (time, j, i)
longitude, latitude: (j, i)
```

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## PBS execution

Review project, storage, queue, walltime, memory, and module settings in
`run_heatbudget.pbs`. The corrected TEND requires all five variants to be regenerated:

```bash
#PBS -J 0-4
variants=(raw aa ab ba bb)
```

Submit from this directory:

```bash
qsub run_heatbudget.pbs
```

Override configuration if needed:

```bash
qsub -v HB_INPUT_DIR=/path/to/input,HB_OUTPUT_DIR=/path/to/output,HB_PERIOD=190001-200912 run_heatbudget.pbs
```

Monitor with `qstat -u "$USER"`. Outputs go to `output/`; logs go to `logs/`.
For `190001-200912`, output suffixes are `.nc`, `_aa.nc`, `_ab.nc`, `_ba.nc`,
and `_bb.nc`. Do not commit generated NetCDF files or scheduler logs to Git.


### References

Guan, C., & McPhaden, M. J. (2016).
Ocean Processes Affecting the Twenty-First-Century Shift in ENSO SST Variability.
*Journal of Climate*, 29, 6861–6879.
https://doi.org/10.1175/JCLI-D-15-0870.1
