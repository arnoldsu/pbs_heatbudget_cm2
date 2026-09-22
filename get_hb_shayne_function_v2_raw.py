import xarray as xr
import numpy as np
from joblib import Parallel, delayed
from func_hb_ars599 import *
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Define the path to the script directory
script_directory = os.environ.get("HB_OUTPUT_DIR", os.getcwd())
os.makedirs(script_directory, exist_ok=True)

# Change to the script directory
os.chdir(script_directory)

# Print the new working directory to confirm the change
print("New working directory:", os.getcwd())


# ========= Load the NetCDF datasets ===========
# Get the argument passed from the Bash script
import sys
rpfName = "r1i1p1f1"
prdName = sys.argv[1]  # sys.argv[0] is the script name, so [1] is the first argument
print(f"Received prdName: {prdName}")
#inDir="/datastore/sul086/work_mhg/cm2r1"
inDir = os.environ.get("HB_INPUT_DIR", os.path.join(os.getcwd(), "input"))
ds_t = xr.open_dataset(f"{inDir}/thetao_Omon_ACCESS-CM2_historical_" + rpfName + "_gn_" + prdName + "_v10.nc")
ds_u = xr.open_dataset(f"{inDir}/uo_Omon_ACCESS-CM2_historical_" + rpfName + "_gn_" + prdName + "_v10.nc")
ds_v = xr.open_dataset(f"{inDir}/vo_Omon_ACCESS-CM2_historical_" + rpfName + "_gn_" + prdName + "_v10.nc")
ds_w = xr.open_dataset(f"{inDir}/wo_Omon_ACCESS-CM2_historical_" + rpfName + "_gn_" + prdName + "_v10.nc")
ds_hf = xr.open_dataset(f"{inDir}/hfds_Omon_ACCESS-CM2_historical_" + rpfName + "_gn_" + prdName + "_v10.nc")

# need a function to get the following parameters
# with given lat lon is 240 and 0
# try to find the clost ilon or ilat base on ds_t
# ds_t.longitude[ilat,ilon].values #: array(239.5)
# ds_t.latitude[ilat,ilon].values #: array(-0.16620922)
# ds_u.longitude[ilat,ilon].values #: array(239.)
# ds_u.latitude[ilat,ilon].values #: array(-0.33333334)
#ilon = 120
#ilat = 48

# Define your target corners

nlev = 5 # one more extra 0-50 + 60
target_latlon_sw = find_closest_lat_lon(ds_t, 122, -17)
target_latlon_ne = find_closest_lat_lon(ds_t, 278, 17)
#print( target_latlon_sw, target_latlon_ne)


# --------- for testing only ---------
#target_latlon_sw = find_closest_lat_lon(ds_t,239,-0.2)
#target_latlon_ne = find_closest_lat_lon(ds_t,241,0.2)
#print( target_latlon_sw, target_latlon_ne)


"""
so I need an array starts from the south west conor ( target_latlon_sw ) to the north east ( target_latlon_ne )

setup this array also the time axis is the same length to the ds_t so that we have new array time x j x i as

size(ds_t.time) * (target_latlon_ne[1] - target_latlon_sw[1]) * (target_latlon_ne[0] - target_latlon_sw[0])

calculate_climate_metrics
"""
import numpy as np

# Define your time dimension length (replace ds_t.time with actual length)
time_length = len(ds_t.time)

# Calculate number of points in each dimension
# for new array
lon_points = int( target_latlon_ne[0] - target_latlon_sw[0] ) + 1
lat_points = int( target_latlon_ne[1] - target_latlon_sw[1] ) + 1

#print(lon_points)
#print(lat_points)

# Create the array with dimensions (time, lat_points, lon_points)
array_shape = (time_length, lat_points, lon_points)
data_array = np.zeros(array_shape)

# Print the shape of the created array
print("Shape of the data_array:", data_array.shape)



# following no need!!
lon_array = range(target_latlon_ne[1], target_latlon_sw[1]+5)
lat_array = range(target_latlon_ne[0], target_latlon_sw[0]+5)
#print(lon_array)
#print(lat_array)


#==============================================================================
# Calculate the complete region in one JAX-compiled call.
metrics = calculate_climate_metrics_region(
    target_latlon_sw[0], target_latlon_ne[0],
    target_latlon_sw[1], target_latlon_ne[1], nlev,
    ds_t=ds_t, ds_u=ds_u, ds_v=ds_v, ds_w=ds_w, ds_hf=ds_hf)

TEND = np.asarray(metrics["TEND"])
UTi = np.asarray(metrics["UTi"])
VTi = np.asarray(metrics["VTi"])
UTe = np.asarray(metrics["UTe"])
UTw = np.asarray(metrics["UTw"])
VTn = np.asarray(metrics["VTn"])
VTs = np.asarray(metrics["VTs"])
WTi = np.asarray(metrics["WTi"])
SHF = np.asarray(metrics["SHF"])

lon = ds_t.longitude[0, target_latlon_sw[0]:target_latlon_ne[0] + 1].data
lat = ds_t.latitude[target_latlon_sw[1]:target_latlon_ne[1] + 1, 0].data
time = ds_t.time.data

# Create xarray dataset for the variables
ds = xr.Dataset(
    {
        'UTe': (['time', 'lat', 'lon'], UTe),
        'UTw': (['time', 'lat', 'lon'], UTw),
        'VTn': (['time', 'lat', 'lon'], VTn),
        'VTs': (['time', 'lat', 'lon'], VTs),
        'UTi': (['time', 'lat', 'lon'], UTi),
        'VTi': (['time', 'lat', 'lon'], VTi),
        'WTi': (['time', 'lat', 'lon'], WTi),
        'TEND': (['time', 'lat', 'lon'], TEND),
        'SHF': (['time', 'lat', 'lon'], SHF),
    },
    coords={
        'lat': (['lat'], lat),
        'lon': (['lon'], lon),
        'time': (['time'], time),
    }
)


# Add metadata (optional)
ds.UTe.attrs['units'] = 'C/s'
ds.UTw.attrs['units'] = 'C/s'
ds.VTn.attrs['units'] = 'C/s'
ds.VTs.attrs['units'] = 'C/s'
ds.UTi.attrs['units'] = 'C/s'
ds.VTi.attrs['units'] = 'C/s'
ds.WTi.attrs['units'] = 'C/s'
ds.TEND.attrs['units'] = 'C/s'
ds.SHF.attrs['units'] = 'C/s'

# Save the dataset to a NetCDF file
ds.to_netcdf(f'{script_directory}/climate_metrics_mpi_heatbudget_pac_' + rpfName + '_gn_' + prdName + '.nc')

print('NetCDF file saved as climate_metrics_mpi_heatbudget_pac_' + rpfName + '_gn_' + prdName + '.nc.')



# vts,vtn,ute 1e-11 totally wrong!!!
# utw = 0, totally wrong!!!
# plot at this point i=3, j=1 !!
#




