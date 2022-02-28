from pathlib import Path
from textwrap import dedent
import numpy as np
import ee
import json
import numpy as np
from multiprocessing import Pool
from functools import partial
import pandas as pd
import datetime
from retry import retry

ee.Initialize(opt_url = 'https://earthengine-highvolume.googleapis.com')


def write_pcse_csv(
    variables,
    elev,
    lon,
    lat,
    csv_file,
    c1=-0.18,
    c2=-0.55,
):
    """Write a PCSE-friendly CSV meteo file. Uses the data I have stored in
    JASMIN (ERA5)

    Parameters
    ----------
    data: array
        The actual daily data as a pandas data frame. Should have columns
        `DAY`, `IRRAD`, `TMIN`, `TMAX`, `VAP`, `WIND`, `RAIN`, in standard
        WOFOST units.
    elev: float
        Elevation in m ASL
    lon : float
        Longitude in decimal degrees.
    lat : float
        Latitude in decimal degrees
    csv_file : str
        CSV filename to store
    c1 : float, optional
        The `c1` parameter, by default -0.18
    c2 : float, optional
        The `c2` parameter, by default -0.55

    Returns
    A pathlib object to the CSV file.
    """
    
    # if file exists, return old file

    # elev = retrieve_pixel_value(lon, lat, DEM_FILE)
    country = 'somewhere'
    site_name = 'anything'
    hdr_chunk = f"""Country     = '{country}'
Station     = '{site_name}'
Description = 'Reanalysis data'
Source      = 'ERA5'
Contact     = 'J Gomez-Dans'
Longitude = {lon}; Latitude = {lat}; Elevation = {elev}; AngstromA = {c1}; AngstromB = {c2}; HasSunshine = False
## Daily weather observations (missing values are NaN)
    """
    hdr_chunk = dedent(hdr_chunk)
    variables["SNOWDEPATH"] = variables["IRRAD"] * np.nan
    variables.columns = [
        "DAY",
        "IRRAD",
        "TMIN",
        "TMAX",
        "VAP",
        "WIND",
        "RAIN",
        "SNOWDEPTH",
    ]
    with csv_file.open(mode="w", newline="") as fp:
        fp.write(hdr_chunk)
        variables.to_csv(fp, index=False, na_rep="nan")
    return csv_file

def write_internal_file(inputfile, year, lon, lat, solar_radiation, temperature_2m_min, temperature_2m_max, humidity, wind, total_precipitation):
    site = 'anything'
    c1 = -0.18
    c2 = -0.55
    with open(inputfile+".%s"%(str(year)[-3:]),"w+") as f:
        f.write("*------------------------------------------------------------*"+"\n"
                    +'*'+"%12s"%("Country: ")+"By Coordinate"+"\n"
                    +'*'+"%12s"%("Station: ")+"%s"%site+"\n"
                    +'*'+"%12s"%("Year: ")+"%d"%(year)+"\n"
                    +'*'+"%12s"%("Origin: ")+"ERA5 Reanalysis"+"\n"
                    +'*'+"%12s"%("Author: ")+"UCL"+"\n"
                    +'*'+"%12s"%("Longitude: ")+"%f"%(lon)+" E"+"\n"
                    +'*'+"%12s"%("Latitude: ")+"%f"%(lat)+" N"+"\n"
                    +'*'+"%12s"%("Elevation: ")+"%.2f"%(elev)+" m"+"\n"
                    +'*'+"%12s"%("Columns: ")+"\n"
                    +'*'+"%12s"%("======== ")+"\n"
                    +'*'+"  station number"+"\n"
                    +'*'+"  year"+"\n"
                    +'*'+"  day"+"\n"
                    +'*'+"  irradiation (kJ·m-2·d-1)"+"\n"
                    +'*'+"  minimum temperature (degrees Celsius)"+"\n"
                    +'*'+"  maximum temperature (degrees Celsius)"+"\n"
                    +'*'+"  vapour pressure (kPa)"+"\n"
                    +'*'+"  mean wind speed (m·s-1)"+"\n"
                    +'*'+"  precipitation (mm·d-1)"+"\n"
                    +'**'+" WCCDESCRIPTION="+site+", Africa"+"\n"
                    +'**'+" WCCFORMAT=2"+"\n"
                    +'**'+" WCCYEARNR="+"%d"%(year)+"\n"
                    +"*------------------------------------------------------------*"+"\n"
                    +"%.2f  %.2f  %.2f  %.2f  %.2f\n"%(lon, lat, elev, c1, c2)
                    )

        for d in range(solar_radiation.shape[0]):
            f.write("%d"%(station_number)+"\t"+"%d"%(year)+"\t"+"%3d"%(1+d)+"\t"
                        +"%5d"%(round(solar_radiation[d]))+"\t"
                        +"%5.1f"%(round(temperature_2m_min[d]*10)/10)+"\t"
                        +"%5.1f"%(round(temperature_2m_max[d]*10)/10)+"\t"
                        +"%5.3f"%(round(humidity[d]*1000)/1000)+"\t"
                        +"%4.1f"%(round(wind[d]*10)/10)+"\t"
                        +"%4.1f"%(np.clip(round(total_precipitation[d]*10),0,250)/10)+"\n")
        return inputfile


def get_weather_images(parameter, year):
    dataset = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")\
                    .filter(ee.Filter.date('%d-01-01'%year, '%d-01-01'%(year+1)))\
                    .select([parameter])\
                    .toBands()
    return dataset

@retry(tries=10, delay=1, backoff=2)
def download_image_over_region(image, geom, scale):
    return image.reduceRegion(ee.Reducer.mean(), geom, scale).getInfo()

def calculate_hum(tdew):
    tdew = tdew - 273.15
    tmp = (17.27 * tdew) / (tdew + 237.3)
    ea = 0.6108 * np.exp(tmp)
    return ea

def get_era5_gee(year, lat, lon, dest_folder='data/'):
    geom = ee.Geometry.Point(lon, lat).buffer(5500)
    
    country = "Somewhere"
    site = 'anything'
    product="ERA5"
    lat_str = '%.02f'%lat
    lon_str = '%.02f'%lon
    
    csv_file = (
        Path(dest_folder) / f"{product}_{country}_{lat_str}_{lon_str}_{year}.csv"
    )
    if csv_file.exists():
        return csv_file
    
    parameters = ['dewpoint_temperature_2m', 'temperature_2m', 'surface_solar_radiation_downwards_hourly', 'total_precipitation_hourly', 'u_component_of_wind_10m', 'v_component_of_wind_10m']
    
    images = [get_weather_images(parameter, year) for parameter in parameters]
    images += [ee.Image('USGS/GTOPO30').select('elevation')]

    par = partial(download_image_over_region, geom = geom, scale = 11132)
    pool = Pool(len(images))
    ret = pool.map(par, images)
    pool.close()
    pool.join()

    ret = {k: v for d in ret for k, v in d.items()}

    transforms = [lambda val: calculate_hum(val), lambda val: val - 273.15,  lambda val: val / 1000, lambda val: val * 1000, lambda val: val, lambda val: val]
    transforms = dict(zip(parameters, transforms))

    total_days = (datetime.datetime(year+1, 1, 1) - datetime.datetime(year, 1, 1)).days
    available_keys = ret.keys() 

    para_vals = []
    for parameter in parameters:
        tranform = transforms[parameter]
        vals = []
        for i in range(total_days):
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=i)
            date_str = date.strftime('%Y%m%d')
            for hour in range(24):
                header = '%sT%02d_%s'%(date_str, hour, parameter)
                if header in available_keys:
                    if ret[header] is not None:
                        val = tranform(ret[header])
                    else:
                        val = np.nan
                else:
                    val = np.nan
                vals.append(val)
        para_vals.append(vals)      

    para_vals = np.array(para_vals).reshape(6, total_days, 24)

    humidity            = np.maximum(np.nanmean(para_vals[0], axis=1), 0)
    temperature_2m_max  = np.nanmax (para_vals[1], axis=1)
    temperature_2m_min  = np.nanmin (para_vals[1], axis=1)
    solar_radiation     = np.maximum(np.nansum (para_vals[2], axis=1), 0)
    total_precipitation = np.maximum(np.nansum (para_vals[3], axis=1), 0)
    wind                = np.nanmean(np.sqrt(para_vals[4]**2 + para_vals[5]**2), axis=1)
    elev                = ret['elevation']
    
    
    keys = ['DAY', 'IRRAD', 'TMIN', 'TMAX', 'VAP', 'WIND', 'RAIN']
    vals = [pd.date_range('%d-01-01'%year, '%d-12-31'%year), solar_radiation, temperature_2m_min, temperature_2m_max, humidity, wind, total_precipitation]
    df = pd.DataFrame(dict(zip(keys, vals)))
    csv_file = write_pcse_csv(df, elev, lon, lat, csv_file)
    return csv_file

    from pcse.fileinput import CSVWeatherDataProvider

    wdp = CSVWeatherDataProvider(
                 era5_file, 
                 dateformat="%Y-%m-%d",
                 delimiter=","
                 )
    
    inputfile = write_internal_file(dest_folder +'/test', year, lon, lat, solar_radiation, temperature_2m_min, temperature_2m_max, humidity, wind, total_precipitation)
    CABOWeatherDataProvider(inputfile, fpath='./')
    
    