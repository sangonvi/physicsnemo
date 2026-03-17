import os
import requests
import numpy as np
import xarray as xr
import cfgrib
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CONFIGURAÇÃO
# ==============================

DATE = "20240101"
HOURS = range(0,6)

VARIABLES = [
    "t2m",
    "u10",
    "v10",
    "sp",
]

PATCH_SIZE = 128
STRIDE = 128

DATA_DIR = "hrrr_data"
os.makedirs(DATA_DIR,exist_ok=True)

# ==============================
# 1 DOWNLOAD HRRR
# ==============================

def download_hrrr(date,hour):

    url=f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/hrrr.t{hour:02d}z.wrfsfcf00.grib2"

    file=f"{DATA_DIR}/hrrr_{date}_{hour}.grib2"

    if os.path.exists(file):
        return file

    r=requests.get(url)

    with open(file,"wb") as f:
        f.write(r.content)

    return file

# ==============================
# 2 DETECTAR VARIÁVEIS
# ==============================

def load_variables(file):

    datasets = cfgrib.open_datasets(
        file,
        backend_kwargs={"indexpath":""}
    )

    variables={}

    for ds in datasets:
        for v in ds.data_vars:
            variables[v]=ds[v]

    return variables


# ==============================
# 3 EXTRAIR CAMPOS
# ==============================

def extract_fields(files):

    stacks={v:[] for v in VARIABLES}

    for file in tqdm(files):

        vars=load_variables(file)

        for v in VARIABLES:

            if v in vars:

                field=vars[v].values

                stacks[v].append(field)

    for v in stacks:
        stacks[v]=np.stack(stacks[v])

    return stacks

# ==============================
# 4 EXTRAIR PATCHES
# ==============================

def extract_patches(field,size,stride):

    patches=[]

    T,H,W=field.shape

    for t in range(T):

        f=field[t]

        for i in range(0,H-size,stride):
            for j in range(0,W-size,stride):

                patches.append(f[i:i+size,j:j+size])

    return np.array(patches)

# ==============================
# 5 GERAR DATASET
# ==============================

def build_dataset(fields):

    inputs={}
    outputs={}

    for v,data in fields.items():

        patches=extract_patches(data,PATCH_SIZE,STRIDE)

        hr=patches
        lr=gaussian_filter(hr,sigma=(0,4,4))

        inputs[v]=lr
        outputs[v]=hr

    return inputs,outputs

def save_dataset(inputs, outputs, filename="../data/corrdiff_dataset.nc"):

    variables = list(inputs.keys())

    sample_var = inputs[variables[0]]

    N = sample_var.shape[0]
    H = sample_var.shape[1]
    W = sample_var.shape[2]

    # -------------------------
    # ROOT DATASET
    # -------------------------

    times = np.arange(N).astype("int64")

    # Coordenadas dos patches
    # Como o grid é exatamente 128x128,
    # os patches precisam começar em (0,0)
    coords = np.zeros((N, 2), dtype=np.int32)

    nc = Dataset(filename, "w")

    nc.createDimension("sample", N)
    nc.createDimension("coord_dim", 2)

    time_var = nc.createVariable("time", "i8", ("sample",))
    coord_var = nc.createVariable("coord", "i4", ("sample", "coord_dim"))

    time_var[:] = times
    coord_var[:] = coords

    nc.close()

    # -------------------------
    # INPUT / OUTPUT DATASETS
    # -------------------------

    y = np.arange(H)
    x = np.arange(W)

    ds_input = xr.Dataset()
    ds_output = xr.Dataset()

    for v in variables:

        ds_input[v] = xr.DataArray(
            inputs[v],
            dims=("time", "y", "x"),
            coords={
                "time": np.arange(N),
                "y": y,
                "x": x
            }
        )

        ds_output[v] = xr.DataArray(
            outputs[v],
            dims=("time", "y", "x"),
            coords={
                "time": np.arange(N),
                "y": y,
                "x": x
            }
        )

    ds_input.to_netcdf(
        filename,
        group="input",
        mode="a",
        engine="netcdf4"
    )

    ds_output.to_netcdf(
        filename,
        group="output",
        mode="a",
        engine="netcdf4"
    )

    # -------------------------
    # INVARIANT DATASET
    # -------------------------

    elevation = np.random.rand(H, W)

    landmask = (np.random.rand(H, W) > 0.5).astype(float)

    ds_inv = xr.Dataset(
        {
            "elev_mean": (["y", "x"], elevation),
            "lsm_mean": (["y", "x"], landmask)
        },
        coords={
            "y": y,
            "x": x
        }
    )

    ds_inv.to_netcdf(
        filename,
        group="invariant",
        mode="a",
        engine="netcdf4"
    )

    print("Dataset salvo:", filename)
    
# ==============================
# PIPELINE
# ==============================

def main():

    files=[]

    for h in HOURS:
        f=download_hrrr(DATE,h)
        files.append(f)

    fields=extract_fields(files)

    inputs,outputs=build_dataset(fields)

    save_dataset(inputs,outputs)

    print("Dataset gerado: corrdiff_dataset.nc")


if __name__=="__main__":
    main()
    
