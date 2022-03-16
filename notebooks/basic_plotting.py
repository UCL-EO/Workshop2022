import geemap, ee
import os, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

MIN_YEAR = 2006
MAX_YEAR = 2016
END_YEAR = 2018


# use Tamale loc of 9.38, -0.68
tamale_centre = [9.38, -0.68]
# alternatively, start with a GSSTI farm location with high expected yield of 1500 per acre
farm_7021YAM = [9.70065, -0.54129]
Map = geemap.Map(center=farm_7021YAM, zoom=9)
#Map = Map.add_basemap('HYBRID')

#tamale_rec = ee.Geometry.Rectangle([-0.75, 9.26, -0.23, 9.71])
ghana_country = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Ghana'))

cities = ee.FeatureCollection("FAO/GAUL/2015/level2")
#Map.addLayer(cities, {}, 'Cities', False)

ghana_district = cities.filter(ee.Filter.eq('ADM0_NAME', 'Ghana'))
northern_district = ghana_district.filter(ee.Filter.eq('ADM1_NAME', 'Northern'))
aoi_bole = northern_district.filter(ee.Filter.eq('ADM2_NAME', 'Bole'))
aoi_tamale = northern_district.filter(ee.Filter.eq('ADM2_NAME', 'Tamale North Sub Metro'))

outline = ee.Image().byte().paint(**{
  'featureCollection': northern_district,
  'color': 1,
  'width': 2
})


landsat_vis_param = {
            'min': 0,
            'max': 3000,
            'bands': ['NIR', 'Red', 'Green']  # False Colour Composit bands to be visualised
}
ndvi_colorized_vis = {
            'min': 0.0,
            'max': 1.0,
            'palette': [
            'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
            '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
            '012E01', '011D01', '011301']
}
lai_colorized_vis = {
            'min': 0.0,
            'max': 2.5,
            'palette': ['e1e4b4', '999d60', '2ec409', '0a4b06']
}
fpar_colorized_vis = {
            'min': 0.0,
            'max': 1.0,
            'palette': ['e1e4b4', '999d60', '2ec409', '0a4b06']
}
igbpLandCoverVis = {
    'min': 0.0,
    'max': 17.0,
    'palette': [ '1c0dff',
    '05450a', '086a10', '54a708', '78d203', '009900', 'c6b044', 'dcd159',
    'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44', 'a5a5a5', 'ff6d4c',
    '69fff8', 'f9ffa4', '1c0dff']
}

igbpLandCoverVis3 = {
    'min': 0.0,
    'max': 10.0,
    'palette': [ '1c0dff',
    'b6ff05', 'dcd159', 'c24f44', 'fbff13', '086a10',
    '78d203', '05450a', '54a708', 'f9ffa4', 'a5a5a5']
}

colorized_vis = {
                 'Lai': lai_colorized_vis,\
                 'NDVI': ndvi_colorized_vis,\
                 'Fpar': fpar_colorized_vis,\
                 'EVI': ndvi_colorized_vis,\
                 'LC_Type1': igbpLandCoverVis,\
                 'LC_Type2': igbpLandCoverVis,
                 'LC_Type3': igbpLandCoverVis3
}


def refresh_base_map(loc=tamale_centre):
    Map = geemap.Map(center=loc, zoom=9)
    Map.addLayer(outline, {}, 'Ghana districts')
    Map


#refresh_base_map(loc=farm_7021YAM)


def load_landsat_collection(year, aoi, cloud_tolerance=3.0,
                            DISPLAY_ON_MAP=False, MEDIAN_ONLY=False):
    '''This function allows GEE to display a Landsat data collection
    from any year between 1984 and present year
    that fall within the AOI and cloud tolerance, e.g. 3.0%.
    There are two optional flag:
    When DISPLAY_ON_MAP is TRUE, display this layer onto Map;
    When return_series = 'MEDIAN_ONLY', only median SR is loaded into landsat_ts, and
    Setting this option to MEDIAN_ONLY would be faster than loading other collections.
    '''
    assert year >= 1984

    def renameBandsETM(image):
        # if year >=2013: #LS8
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']  # , 'pixel_qa'
        new_bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']  # , 'pixel_qa'

        if year <= 1984:
            bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa']
            new_bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa']
        return image.select(bands).rename(new_bands)

    if not (MEDIAN_ONLY):
        if year >= 2013:
            layer_name = 'LC08'  # LS8: 2013-now
        elif year == 2012:  # # LS7: 1999- , however SLC error >= 1999:
            layer_name = 'LE07'
        elif year >= 1984:
            layer_name = 'LT05'  # LS5: 1984-2012

        collection_name_sr = f"LANDSAT/{layer_name}/C01/T1_SR"
        # You can also use the following line, if interested in incorperating ndvi
        collection_name_ndvi = f"LANDSAT/{layer_name}/C01/T1_ANNUAL_NDVI"

        all_sr_image = ee.ImageCollection(collection_name_sr) \
            .filterBounds(aoi) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_tolerance)) \
            .sort('system:time_start') \
            .select('B[1-7]') \
            .sort('CLOUD_COVER')

        all_sr_image = all_sr_image.map(renameBandsETM)  # rename bands with 'renameBandsETM' function

        # reduce all_sr_image to annual average per pixel
        mean_image = all_sr_image.mean()
        mean_image = mean_image.clip(aoi).unmask()

        ndvi_image = ee.ImageCollection(collection_name_ndvi) \
            .filterBounds(aoi) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .select('NDVI') \
            .first()
        ndvi_image = ndvi_image.clip(aoi).unmask()

        # mean_image.addBands(ndvi_image, 'NDVI')

    # This line loads all annual median surface ref
    landsat_ts = geemap.landsat_timeseries(roi=tamale_rec, start_year=year, end_year=year, \
                                           start_date='01-01', end_date='12-31')

    median_image = landsat_ts.first().clip(aoi).unmask()

    if DISPLAY_ON_MAP == True:

        if not (MEDIAN_ONLY):
            Map.addLayer(ndvi_image, ndvi_colorized_vis, 'NDVI ' + str(year), opacity=0.9)
            Map.addLayer(mean_image, landsat_vis_param, "Mean Ref " + str(year))
        Map.addLayer(median_image, landsat_vis_param, "Median Ref " + str(year))

    if MEDIAN_ONLY:
        return median_image
    else:
        return all_sr_image, mean_image, median_image, ndvi_image


def load_modis_collection(year, aoi,
                          collection='MOD13Q1',
                          band='NDVI', scale=0.0001,
                          TIME_REDUCER='None', DISPLAY_REDUCED_ON_MAP=False):
    '''This function allows GEE to display a MODIS data collection
    from any year
    that fall within the AOI and cloud tolerance, e.g. 3.0%.
    There are two optional flag:
    When DISPLAY_ON_MAP is 'None', don't display this layer onto Map,
    otherwise, display 'mean', 'median', or 'max', etc.
    '''
    assert year >= 2000
    # print('TIME_REDUCER=',TIME_REDUCER, 'DISPLAY_REDUCED_ON_MAP=', DISPLAY_REDUCED_ON_MAP)

    # if not LC, filter maize planitng season only, or whole year f'{year}-01-01', f'{year}-12-31
    if collection == 'MCD12Q1':
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
    else:
        start_date = f'{year}-06-01'  # start of maize growing season
        end_date = f'{year}-11-30'  # end of maize growing season

    all_image = ee.ImageCollection(f"MODIS/006/{collection}") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .sort('system:time_start') \
        .select(band)
    # .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 10)

    # print('-------all_image----------',all_image.getInfo())
    # rescale if scale <>1
    if scale != 1:
        all_image = all_image.map(lambda image: image.multiply(scale))

    # reduce all_sr_image to e.g. annual average per pixel
    # mean_image = all_image.mean().clip(aoi) #.unmask()
    if TIME_REDUCER != 'None':
        methods = {'mean': all_image.mean, \
                   'max': all_image.max, \
                   'min': all_image.min, \
                   'median': all_image.median}

        if TIME_REDUCER in methods:
            reduced_image = methods[TIME_REDUCER]().clip(aoi)  # .unmask()
        else:
            raise Exception(f"Method {TIME_REDUCER} not implemented")

        if DISPLAY_REDUCED_ON_MAP == True:
            Map.addLayer(reduced_image, colorized_vis[band], f"{TIME_REDUCER} {band} {year}", opacity=0.7)

        print(f'MODIS {band} and mean both SUCCESS for {year}')
        return all_image, reduced_image
    else:
        print(f'Returning MODIS {band} time series (no time-reducer)')

        # zonalS

        if DISPLAY_REDUCED_ON_MAP == True:
            print('Cannot print time series as a single layer')
        return all_image, None

## Reduce to per_district (spatial) mean/median LAI

def load_modis_band(band, year, aoi, TIME_REDUCER):
    # Define VIs or LandCover bands and their associated MODIS collections
    layers = {'NDVI': 'MOD13Q1', 'EVI': 'MOD13Q1', \
              'Lai': 'MCD15A3H', \
              'LC_Type1': 'MCD12Q1', 'LC_Type2': 'MCD12Q1', 'LC_Type3': 'MCD12Q1'}
    scales = {'NDVI': 0.0001, 'EVI': 0.0001, \
              'Lai': 0.1, \
              'LC_Type1': 1, 'LC_Type2': 1, 'LC_Type3': 1}

    # find modis collection:
    collection = layers[band]
    scale = scales[band]

    collection_all, annual_mean = load_modis_collection(year, aoi, \
                                                        collection=collection, band=band, scale=scale, \
                                                        TIME_REDUCER=TIME_REDUCER,
                                                        DISPLAY_REDUCED_ON_MAP=(TIME_REDUCER != 'None'))
    # print(collection_all.getInfo(), '--'*20)
    # print(annual_mean.getInfo(), '=='*20)

    return collection_all, annual_mean


def load_modis_lc(band, year, aoi, cropclasses=[10]):
    collection_all, annual_mean_lc = load_modis_band(band, year, aoi, 'mean')
    # print(annual_mean_lc.getInfo())

    '''Reclassify Yearly Land Cover into a Grassland only image 'cropland'.
    However, MAIZE should be 'grassland' in MCD12Q1 peoducts, not 'cropland'
        LC_Type1 == 10 (b6ff05) Grasslands: dominated by herbaceous annuals (<2m).
        or LC_Type2 == 10 (b6ff05) Grasslands: dominated by herbaceous annuals (<2m).
        or LC_Type3 == 1 (b6ff05) Grasslands: dominated by herbaceous annuals (<2m) including cereal croplands.
    But NOT LC_Type2 == 12 : Croplands: at least 60% of area is cultivated cropland.
        NOT LC_Type2 == 14 : Cropland/Natural Vegetation Mosaics: mosaics of small-scale cultivation 40-60% with natural tree, shrub, or herbaceous vegetation.
    '''
    if band in ['LC_Type1', 'LC_Type2', 'LC_Type3']:
        raw_class_values = list(range(0, 256))
        new_class_values = [0] * len(raw_class_values)
        for class_index in cropclasses:
            new_class_values[class_index] = 1

            # print(raw_class_values, new_class_values)

    cropland = annual_mean_lc.remap(raw_class_values, new_class_values)
    return cropland, annual_mean_lc


