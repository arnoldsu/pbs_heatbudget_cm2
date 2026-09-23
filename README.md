# JAX-HeatBudget-CM2: Offline Ocean Heat-Budget Analysis for ACCESS-CM2

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22908714.svg)](https://doi.org/10.5281/zenodo.22908714)

This repository provides an offline monthly upper-ocean temperature
heat-budget calculation using archived ACCESS-CM2 ocean fields from
CMIP-standard datasets.

The production kernel is vectorized across time, depth, latitude, and
longitude and compiled with JAX. Xarray handles NetCDF input, coordinates,
and output.

---

## Offline Heat-Budget Formulation

The upper-ocean temperature tendency is written schematically as

```math
\frac{\partial T}{\partial t}
=
\mathrm{ADV}
+
\mathrm{SHF}
+
\mathrm{RES}
```

where **ADV** represents three-dimensional ocean temperature advection,
**SHF** represents the surface heat-flux contribution, and **RES** represents
processes not explicitly included in the diagnosed budget together with
numerical budget-closure error.

In the present implementation, the total advective contribution is

```math
\mathrm{ADV}
=
-\left(
\mathrm{UTi}
+
\mathrm{VTi}
+
\mathrm{WTi}
\right)
```

where **UTi**, **VTi**, and **WTi** represent the diagnosed zonal,
meridional, and vertical advection contributions, respectively.

The diagnosed right-hand side is

```math
\mathrm{RHS}
=
\mathrm{ADV}
+
\mathrm{SHF}
```

and the residual is diagnosed as

```math
\mathrm{RES}
=
\mathrm{TEND}
-
\mathrm{RHS}
```

or equivalently,

```math
\mathrm{RES}
=
\mathrm{TEND}
-
\left(
\mathrm{ADV}
+
\mathrm{SHF}
\right)
```

Thus, the complete diagnosed temperature budget is

```math
\mathrm{TEND}
=
\mathrm{ADV}
+
\mathrm{SHF}
+
\mathrm{RES}
```

The calculation is performed offline from archived ACCESS-CM2 temperature,
velocity, and surface heat-flux fields.

The horizontal advection terms are evaluated using the eastern/western and
northern/southern faces surrounding each temperature grid cell. The zonal
contribution therefore combines the temperature-advection contributions
diagnosed from the eastern and western faces, while the meridional contribution
combines those from the northern and southern faces.

This face-based formulation is related to the broader control-volume
heat-budget approach used to diagnose ENSO heat transport across the
boundaries of the Niño-3 and Niño-4 regions, for example by
Lee et al. (2004) and Guan and McPhaden (2016).

---

## Heat-Budget Calculation

At each horizontal grid cell, the code evaluates

```text
TEND = d<T>/dt

ADV  = -(UTi + VTi + WTi)

RHS  = ADV + SHF

RES  = TEND - RHS
```

All temperature tendency terms are output in `degC s-1`.

The constants used are

```text
Earth radius, R       = 6,371,000 m
Seawater density      = 1,025 kg m-3
Heat capacity, cp     = 3,996 J kg-1 K-1
```

`TEND` is calculated using the decoded monthly time coordinate converted
to seconds.

---

## Depth-Averaged Temperature and Temperature Tendency

For `nlev` active layers,

```text
dz[k] = lev[k+1] - lev[k]

H = sum(dz[k])

temp_avg = sum(thetao[k] * dz[k]) / H

TEND = gradient(temp_avg, time_seconds)
```

The depth-averaged upper-ocean temperature is

```math
\overline{T}
=
\frac{
\sum_k T_k \Delta z_k
}{
\sum_k \Delta z_k
}
```

where the total layer depth is

```math
H
=
\sum_k \Delta z_k
```

The temperature tendency is therefore

```math
\mathrm{TEND}
=
\frac{\partial \overline{T}}{\partial t}
```

Interior time records use centred differences with the actual, potentially
unequal, monthly time spacing. Endpoints use one-sided differences.

For the ACCESS-CM2 configuration used here,

```text
nlev = 5
H    = 50 m
```

so the diagnosed temperature represents the upper 50-m ocean layer.

---

## Horizontal Advection

Velocity is averaged onto the corresponding temperature-cell faces and
multiplied by the neighbouring temperature difference.

### Zonal Advection

For the eastern and western faces,

```text
ute[k] = 0.5*(uo_NE[k] + uo_SE[k])*(T_E[k] - T_M[k])

utw[k] = 0.5*(uo_NW[k] + uo_SW[k])*(T_M[k] - T_W[k])
```

The depth-integrated eastern and western contributions are

```text
UTe = sum(ute[k]*dz[k])/(dx*H)

UTw = sum(utw[k]*dz[k])/(dx*H)
```

and the zonal contribution used by the heat-budget calculation is

```text
UTi = 0.5*(UTe + UTw)
```

Schematically,

```math
\mathrm{UTi}
=
\frac{1}{2H}
\left[
\frac{1}{\Delta x}
\sum_k
u_E
\left(
T_E-T_M
\right)
\Delta z_k
+
\frac{1}{\Delta x}
\sum_k
u_W
\left(
T_M-T_W
\right)
\Delta z_k
\right]
```

Thus, **UTi** represents the zonal temperature-advection contribution
calculated from the eastern and western cell faces.

### Meridional Advection

For the northern and southern faces,

```text
vtn[k] = 0.5*(vo_EN[k] + vo_WN[k])*(T_N[k] - T_M[k])

vts[k] = 0.5*(vo_ES[k] + vo_WS[k])*(T_M[k] - T_S[k])
```

The depth-integrated northern and southern contributions are

```text
VTn = sum(vtn[k]*dz[k])/(dy*H)

VTs = sum(vts[k]*dz[k])/(dy*H)
```

and

```text
VTi = 0.5*(VTn + VTs)
```

Schematically,

```math
\mathrm{VTi}
=
\frac{1}{2H}
\left[
\frac{1}{\Delta y}
\sum_k
v_N
\left(
T_N-T_M
\right)
\Delta z_k
+
\frac{1}{\Delta y}
\sum_k
v_S
\left(
T_M-T_S
\right)
\Delta z_k
\right]
```

Thus, **VTi** represents the meridional temperature-advection contribution
calculated from the northern and southern cell faces.

The discrete horizontal advection terms therefore depend on velocity
multiplied by the local temperature difference across the corresponding
cell faces, rather than on velocity alone.

---

## Horizontal Grid Distances

Distances are estimated from the two-dimensional model grid coordinates.

The zonal grid distance is

```math
\Delta x
=
R\,\Delta\lambda\cos(\phi)
```

and the meridional grid distance is

```math
\Delta y
=
R\,\Delta\phi
```

where `R` is the Earth radius, `phi` is the representative latitude,
and the longitude and latitude differences are converted from degrees
to radians.

In the code,

```text
dx = R * dlon * cos(mean latitude)

dy = R * dlat
```

---

## Vertical Advection

Vertical advection across the bottom of the diagnosed upper-ocean layer is

```text
WTi = WTb = wo_bottom*(T[nlev-1] - T[nlev])/H
```

or schematically,

```math
\mathrm{WTi}
=
\frac{
w_b
\left(
T_{nlev-1}-T_{nlev}
\right)
}{H}
```

where `w_b` is the vertical velocity at the bottom of the diagnosed layer.

An additional temperature level, `T[nlev]`, is therefore required below
the five active upper-ocean layers to calculate the bottom temperature
gradient.

---

## Surface Heat Flux

The surface heat-flux contribution is

```math
\mathrm{SHF}
=
\frac{
Q_{\mathrm{net}}
}{
\rho_0 c_p H
}
```

and is implemented as

```text
SHF = hfds/(rho0*cp*H)
```

where `hfds` is the net downward surface heat flux into the ocean,
`rho0` is the reference seawater density, `cp` is the seawater heat
capacity, and `H` is the diagnosed upper-ocean depth.

The complete diagnosed advective contribution is

```math
\mathrm{ADV}
=
-\left(
\mathrm{UTi}
+
\mathrm{VTi}
+
\mathrm{WTi}
\right)
```

and the diagnosed right-hand side is

```math
\mathrm{RHS}
=
\mathrm{ADV}
+
\mathrm{SHF}
```

The residual budget term is

```math
\mathrm{RES}
=
\mathrm{TEND}
-
\mathrm{RHS}
```

or

```math
\mathrm{RES}
=
\mathrm{TEND}
-
\left(
\mathrm{ADV}
+
\mathrm{SHF}
\right)
```

---

## Output Variables

The scripts save

```text
UTe
UTw
VTn
VTs
UTi
VTi
WTi
TEND
SHF
```

The regional calculation additionally returns

```text
temp_avg
WTb
ADV
RHS
```

The diagnosed budget structure is

```text
                    TEND
                      |
          -------------------------
          |                       |
         ADV                     SHF
          |
     -----------------
     |       |       |
    UTi     VTi     WTi
     |       |
   E + W   N + S
```

with the closure residual diagnosed from

```text
RES = TEND - RHS
```

---

## Implementation

`calculate_climate_metrics_region()` in `func_hb_ars599.py` is the production
interface.

It extracts the requested region together with a one-cell halo, converts the
input fields to JAX arrays, calls the JIT-compiled `_region_kernel()` once,
and returns an Xarray Dataset.

This implementation removes the previous horizontal loops, repeated Joblib
startup, and nested time/depth loops.

The first call includes JAX compilation time.

The production kernel intentionally differs from the legacy point
implementation in two important respects:

1. `TEND` uses the actual decoded time coordinate and a centred time
   derivative for interior records.
2. Upper-ocean temperature is weighted by layer thickness (`dz`).

---

## Decomposition Variants

The heat-budget calculation can be applied to raw fields or to combinations
of anomaly and background fields.

| Variant | Temperature | Velocity and surface flux |
|---|---|---|
| `raw` | raw | raw |
| `aa` | anomaly | anomaly |
| `ab` | anomaly | background/bar |
| `ba` | background/bar | anomaly |
| `bb` | background/bar | background/bar |

Scripts are named

```text
get_hb_shayne_function_v2_<variant>.py
```

---

## Inputs and Requirements

By default, input is read from

```text
/g/data/p66/ars599/work_mhg/cm2r1
```

Override this location using `HB_INPUT_DIR`.

The raw calculation uses `_v10.nc` files for

```text
thetao
uo
vo
wo
hfds
```

while the decomposition jobs additionally require matching

```text
_v10.anom.nc
_v10.bar.nc
```

files.

Expected dimensions are

```text
thetao, uo, vo, wo: (time, lev, j, i)

hfds:                (time, j, i)

longitude, latitude: (j, i)
```

Install dependencies with

```bash
python -m pip install -r requirements.txt
```

---

## PBS Execution

Review the project, storage, queue, walltime, memory, and module settings in

```text
run_heatbudget.pbs
```

The corrected `TEND` calculation requires all five variants to be regenerated:

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

Monitor jobs with

```bash
qstat -u "$USER"
```

Outputs are written to `output/`, while PBS logs are written to `logs/`.

For `190001-200912`, output suffixes are

```text
.nc
_aa.nc
_ab.nc
_ba.nc
_bb.nc
```

Generated NetCDF files and scheduler logs should not be committed to Git.

---

## Citation

If you use this software in research, please cite:

**Sullivan, Arnold. (2026).  
JAX-HeatBudget-CM2: Offline Ocean Heat-Budget Analysis for ACCESS-CM2.  
Zenodo.  
DOI: 10.5281/zenodo.22908714**

BibTeX:

```bibtex
@software{Sullivan_2026_JAX_HeatBudget_CM2,
  author    = {Sullivan, Arnold},
  title     = {JAX-HeatBudget-CM2: Offline Ocean Heat-Budget Analysis for ACCESS-CM2},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22908714},
  url       = {https://doi.org/10.5281/zenodo.22908714}
}
```

---

## References
Lee, T., I. Fukumori, and B. Tang, 2004: Temperature Advection: Internal versus External Processes. J. Phys. Oceanogr., 34, 1936–1944, https://doi.org/10.1175/1520-0485(2004)034<1936:TAIVEP>2.0.CO;2.

Guan, C., & McPhaden, M. J. (2016).
Ocean Processes Affecting the Twenty-First-Century Shift in ENSO SST Variability.
*Journal of Climate*, **29**, 6861–6879.
https://doi.org/10.1175/JCLI-D-15-0870.1
