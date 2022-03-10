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
            'max': 5.5,
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


era5TemperatureVis = {
  'min': 250.0,
  'max': 320.0,
  'palette': [
    "#000080","#0000D9","#4000FF","#8000FF","#0080FF","#00FFFF",
    "#00FF80","#80FF00","#DAFF00","#FFFF00","#FFF500","#FFDA00",
    "#FFB000","#FFA400","#FF4F00","#FF2500","#FF0A00","#FF00FF"]
}


era5PrecipitationVis = {
  'min': 0.0,
  'max': 0.04,
  'palette': [
    "#000080","#0000D9","#4000FF","#8000FF","#0080FF","#00FFFF",
    "#00FF80","#80FF00","#DAFF00","#FFFF00","#FFF500","#FFDA00",
    "#FFB000","#FFA400","#FF4F00","#FF2500","#FF0A00","#FF00FF"]
}

defaultVisualizationVis = {
    'min': 0.0,
    'max': 100.0,
    'palette': ['e1e4b4', '999d60', '2ec409', '0a4b06'],
}

colorized_vis = {
                 'Lai': lai_colorized_vis,\
                 'NDVI': ndvi_colorized_vis,\
                 'Fpar': fpar_colorized_vis,\
                 'EVI': ndvi_colorized_vis,\
                 'LC_Type1': igbpLandCoverVis,\
                 'LC_Type2': igbpLandCoverVis,
                 'LC_Type3': igbpLandCoverVis3,\
                 'FparLai_QC': defaultVisualizationVis,\
                 'Prec': era5PrecipitationVis,\
                 'Temp': era5TemperatureVis
}


def initiate_tamale_map():
    # use Tamale loc of 9.38, -0.68
    tamale_centre = [9.38, -0.68]
    # alternatively, start with a GSSTI farm location with high expected yield of 1500 per acre
    farm_7021YAM = [9.70065, -0.54129]
    Map = geemap.Map(center=farm_7021YAM, zoom=9)
    # Map = Map.add_basemap('HYBRID')

    # tamale_rec = ee.Geometry.Rectangle([-0.75, 9.26, -0.23, 9.71])
    ghana_country = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Ghana'))

    cities = ee.FeatureCollection("FAO/GAUL/2015/level2")
    # Map.addLayer(cities, {}, 'Cities', False)

    ghana_district = cities.filter(ee.Filter.eq('ADM0_NAME', 'Ghana'))
    northern_district = ghana_district.filter(ee.Filter.eq('ADM1_NAME', 'Northern'))
    aoi_bole = northern_district.filter(ee.Filter.eq('ADM2_NAME', 'Bole'))

    outline = ee.Image().byte().paint(**{
        'featureCollection': northern_district,
        'color': 1,
        'width': 2
    })
    Map.addLayer(outline, {}, 'Ghana districts')

    return Map


def refresh_base_map(Map):
    Map = geemap.Map(center=farm_7021YAM, zoom=9)
    Map.addLayer(outline, {}, 'Ghana districts')
    # Map


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


def bitwiseExtract(qa_value, fromBit, toBit=-1):
    if toBit == -1:
        toBit = fromBit
    maskSize = ee.Number(1).add(toBit).subtract(fromBit)
    mask = ee.Number(1).leftShift(maskSize).subtract(1)
    return qa_value.rightShift(fromBit).bitwiseAnd(mask)


def filterqa(image):
    qa = image.select('FparLai_QC')
    good = bitwiseExtract(qa, 0) # returns 0 for good quality
    return image.updateMask(not(good))


def load_modis_collection(year, aoi,
                          collection='MOD13Q1',
                          band='NDVI', scale=0.0001,
                          TIME_REDUCER='None', DISPLAY_REDUCED_ON_MAP=False,
                          onlyGoodQA=False):
    '''This function allows GEE to display a MODIS data collection
    from any year
    that fall within the AOI and cloud tolerance, e.g. 3.0%.
    There are two optional flag:
    When DISPLAY_ON_MAP is 'None', don't display this layer onto Map,
    otherwise, display 'mean', 'median', or 'max', etc.
    '''
    assert year >= 2000
    # print('TIME_REDUCER=',TIME_REDUCER, 'DISPLAY_REDUCED_ON_MAP=', DISPLAY_REDUCED_ON_MAP)

    # if not LC, filter maize planting season only, or whole year f'{year}-01-01', f'{year}-12-31
    if collection == 'MCD12Q1':
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
    else:
        start_date = f'{year}-06-01'  # start of maize growing season
        end_date = f'{year}-11-30'  # end of maize growing season

    all_image = ee.ImageCollection(f"MODIS/006/{collection}") \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .sort('system:time_start')

    if onlyGoodQA:
        all_image = all_image.map(filterqa).select(band)
        print('QA Filter applied.')
    else:
        all_image = all_image.select(band)
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

        #print(f'MODIS {band} and mean both SUCCESS for {year}')
        return all_image, reduced_image
    else:
        print(f'Returning MODIS {band} time series (no time-reducer)')

        # zonalS

        if DISPLAY_REDUCED_ON_MAP == True:
            print('Cannot print time series as a single layer')
        return all_image, None


def load_modis_band(band, year, aoi, TIME_REDUCER, applyQA=False):
    ''' Define VIs or LandCover bands and their associated MODIS collections
    and Reduce to per_district (spatial) mean/median LAI '''
    layers = {'NDVI': 'MOD13Q1', 'EVI': 'MOD13Q1', \
              'Lai': 'MCD15A3H', \
              'FparLai_QC': 'MCD15A3H', \
              'LC_Type1': 'MCD12Q1', 'LC_Type2': 'MCD12Q1', 'LC_Type3': 'MCD12Q1'}
    scales = {'NDVI': 0.0001, 'EVI': 0.0001, \
              'Lai': 0.1, \
              'FparLai_QC': 1, \
              'LC_Type1': 1, 'LC_Type2': 1, 'LC_Type3': 1}

    # find modis collection:
    collection = layers[band]
    scale = scales[band]

    collection_all, annual_mean = load_modis_collection(year, aoi, \
                                                        collection=collection, band=band, scale=scale, \
                                                        TIME_REDUCER=TIME_REDUCER,
                                                        DISPLAY_REDUCED_ON_MAP=False,
                                                        onlyGoodQA=applyQA)
    #DISPLAY_REDUCED_ON_MAP=(TIME_REDUCER != 'None')

    # print(collection_all.getInfo(), '--'*20)
    # print(annual_mean.getInfo(), '=='*20)

    return collection_all, annual_mean


def load_modis_lc(band, year, aoi, cropclasses=[10], applyQA=False):
    collection_all, annual_mean_lc = load_modis_band(band, year, aoi, 'mean', applyQA=applyQA)
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


# Next, we will use Reducer to calculate per district values
# i.e. the spatial MEAN (and histograms) of temporal MAX Lai for each district

def plot_histo_per_county_v0(axes, histo_per_district, year):
    # vi_per_district at 500m scale, to be plotted as histogram

    list_of_histo = histo_per_district.aggregate_array('histogram').getInfo()
    n = len(list_of_histo)
    # print('No. of histograms',n)
    print(list_of_histo)
    list_of_districts = histo_per_district.aggregate_array('ADM2_NAME').getInfo()
    print(list_of_districts)
    assert len(list_of_districts) == n

    for i, histo in enumerate(list_of_histo):
        bins = histo['bucketMeans']
        his = histo['histogram']
        if i == 0: print('bucketWidth=', histo['bucketWidth'])

        axes[i].plot(bins, his)
        axes[i].title.set_text(list_of_districts[i])

        # Next, find median/mean and st dev of histos
        print(i, max(his), histo)
        max_histo_height = max(his)
        # TO DO : normalise the frequency

        max_bin_index = his.index(max_histo_height)
        lai_stdev = statistics.stdev(his)
        lai_cv = lai_stdev / statistics.mean(his)  # coefficient of variation
        # print(max_histo_height, max_bin_index, lai_cv, statistics.stdev(bins))
        axes[i].axvline(x=bins[max_bin_index], linestyle='-')
        axes[i].axvline(x=bins[max_bin_index] + lai_cv, linestyle='--')
        axes[i].axvline(x=bins[max_bin_index] - lai_cv, linestyle='--')


def plot_histo_per_county_v_fixed_histo(axes, fixed_histo_per_district, year):
    # vi_per_district at 500m scale, to be plotted as histogram
    df_lai_model_params = pd.DataFrame(
        columns=['DISTRICT', 'YEAR', 'MAX_HISTO_FREQ', 'MAXLAI_PEAK', 'LAI_STDEV', 'LAI_CV'])

    list_of_histo = fixed_histo_per_district.aggregate_array('histogram').getInfo()
    n = len(list_of_histo)
    # print('No. of histograms',n)
    list_of_districts = fixed_histo_per_district.aggregate_array('ADM2_NAME').getInfo()
    assert len(list_of_districts) == n

    for i, histo in enumerate(list_of_histo):  # for each district
        histo = np.array(histo)

        bins = histo[:, 0]  # 'bucketMeans'
        his = histo[:, 1]  # 'histogram' or frequencies = his/total_pixel
        # if i ==0: print('bucketWidth=',bins[1]-bins[0])
        total_pixel = int(his.sum())
        # a = lambda: 1 if total_pixel>500 else 0.1

        if total_pixel > 5000:
            a = 1.0
        else:
            a = 0.2
        axes[i].plot(bins, his / total_pixel, alpha=a)
        axes[i].set_xlim(bins[0], bins[-1])
        axes[i].title.set_text(f'{list_of_districts[i]}: {total_pixel:.0f} pixels of grassland')

        # Next, find peak and st dev of histos
        max_histo_height = his.max()
        # TO DO : normalise the frequency with total_pixel

        max_bin_index = his.argmax()
        lai_cv = his.std() / his.mean()  # coefficient of variation
        # print(max_histo_height, max_bin_index, lai_cv, statistics.stdev(bins))
        axes[i].axvline(x=bins[max_bin_index], linestyle='-', alpha=a)
        axes[i].axvline(x=bins[max_bin_index] + lai_cv, linestyle='--', alpha=a)
        axes[i].axvline(x=bins[max_bin_index] - lai_cv, linestyle='--', alpha=a)

        # For each county, to return the following parameters:
        new_row = {'DISTRICT': list_of_districts[i], \
                   'YEAR': year, \
                   'MAX_HISTO_FREQ': max_histo_height / total_pixel, \
                   'MAXLAI_PEAK': bins[max_bin_index], \
                   'LAI_STDEV': his.std(), \
                   'LAI_CV': lai_cv}
        df_lai_model_params = df_lai_model_params.append(new_row, ignore_index=True)

    return df_lai_model_params


def modis_ts_vi_per_district(year, vi_band, aoi, SPACE_DERUCER='mean', \
                             CROPLAND_ONLY=True):
    '''STILL DEBUGGING THIS ONE'''
    # Firstly, get all district names of AOI, regardless of CROPLAND_ONLY
    list_of_districts = aoi.aggregate_array('ADM2_NAME').getInfo()
    # print(list_of_districts)

    vi_ts, vi_none = load_modis_band(vi_band, year, aoi, TIME_REDUCER='None')
    print('**************No reducer***vi_ts**************')
    print(vi_ts.first().getInfo())
    vi_ts, vi_annual_max = load_modis_band(vi_band, year, aoi, TIME_REDUCER='max')
    print('**************OVERALL vi_ts before reducer*****************')
    print(vi_ts.first().getInfo())
    print('**************Annual max reducer*****************')
    print(vi_annual_max.getInfo())
    print('**************Date properties of Annual max reducer*****************')
    print(vi_annual_max.get('properties').getInfo())
    print('**************vi_ts.aggregate_stats(Lai)*****************')
    stats = vi_ts.aggregate_stats('Lai')
    print(vi_band, stats.getInfo())

    space_methods = {'mean': ee.Reducer.mean, \
                     'max': ee.Reducer.max, \
                     'min': ee.Reducer.min, \
                     'median': ee.Reducer.median}

    df_vi_ts = pd.DataFrame(columns=['DISTRICT', 'Date', vi_band])

    for dist_name in list_of_districts:
        print(dist_name)
        aoi_one_dist = aoi.filter(ee.Filter.eq('ADM2_NAME', dist_name))

        vi_ts_one_dist, vi_none = load_modis_band(vi_band, year, aoi_one_dist, TIME_REDUCER='None')
        stats = vi_ts_one_dist.aggregate_stats('Lai')
        print(vi_band, stats.getInfo())

        if CROPLAND_ONLY:
            lc_crop_one_dist, lc_all_one_dist = load_modis_lc('LC_Type2', year, aoi_one_dist, cropclasses=[10])
            vi_ts_one_dist = vi_ts_one_dist.map(lambda image: image.mask(lc_crop_one_dist))
        vi_ts_mean = vi_ts_one_dist.map(lambda image: image.reduceRegions(collection=aoi_one_dist, \
                                                                          reducer=space_methods[SPACE_DERUCER](),
                                                                          scale=500))
        list_of_reducedVI_ts = vi_ts_one_dist.aggregate_array(SPACE_DERUCER).getInfo()
        print('LIST', list_of_reducedVI_ts)

        list_test = vi_ts_mean.aggregate_array(SPACE_DERUCER).getInfo()
        print('TEST', list_test)

        # print(vi_ts_mean.getInfo())
        # df_vi_ts.append(vi_ts_mean)
    return df_vi_ts


def modis_annual_vi_per_district(year, vi_band, aoi, TIME_REDUCER='max', SPACE_DERUCER='mean', \
                                 CROPLAND_ONLY=True, ADD_TO_MAP=False, VERBOSE=False):
    if TIME_REDUCER == 'None':
        # generate TS, by calling the following func instead:
        df_vi_ts = modis_ts_vi_per_district(year, vi_band, aoi, \
                                            SPACE_DERUCER=SPACE_DERUCER, CROPLAND_ONLY=CROPLAND_ONLY)
        return df_vi_ts

    else:  # TIME REDUCED (vi_annual_reduced != None), so only look at 'vi_annual_reduced'
        # Firstly, we calculate the mean of the annual average VI per district:
        vi_timeseries, vi_annual_reduced = load_modis_band(vi_band, year, aoi, TIME_REDUCER)
        '''vi_timeseries is an ImageCollection, and vi_annual_reduced is an Image (or None, if no time_reducer) '''
        # print('******', vi_annual_reduced.getInfo()) # this should be a 'Image' or None

        if CROPLAND_ONLY:
            lc_crop, lc_all = load_modis_lc('LC_Type2', year, aoi, cropclasses=[10])
            vi_annual_reduced = vi_annual_reduced.mask(lc_crop)

        if SPACE_DERUCER != 'None':
            space_methods = {'mean': ee.Reducer.mean, \
                             'max': ee.Reducer.max, \
                             'min': ee.Reducer.min, \
                             'median': ee.Reducer.median}

            if SPACE_DERUCER in space_methods:
                vi_per_district_annual = vi_annual_reduced.reduceRegions(collection=aoi, \
                                                                         reducer=space_methods[SPACE_DERUCER](),
                                                                         scale=500)
                list_of_districts = vi_per_district_annual.aggregate_array('ADM2_NAME').getInfo()
                # print(len(list_of_districts), list_of_districts) # this line prints all district names

                list_of_reducedVIs = vi_per_district_annual.aggregate_array(SPACE_DERUCER).getInfo()
                # print(len(list_of_reducedVIs), list_of_reducedVIs) # this line prints all mean VIs

                if VERBOSE:
                    stats = vi_per_district_annual.aggregate_stats('mean')
                    print(year, vi_band, stats.getInfo())
                if ADD_TO_MAP:
                    Map.addLayer(vi_per_district_annual, {'min': 0.0, 'max': 2.75}, \
                                 f'Mean {vi_band} {year} per district')
                    # return np.asarray(list_of_districts), np.asarray(list_of_reducedVIs)
                return vi_per_district_annual
            else:
                raise Exception(f"Method {SPACE_DERUCER} not implemented")

        if ADD_TO_MAP:
            Map.addLayer(vi_annual_reduced, {'min': 0.0, 'max': 2.75}, \
                         f'Mean {vi_band} {year} per district')

        return vi_annual_reduced


def histo_vi_per_district(year, vi_band, aoi, TIME_REDUCER='max', \
                          CROPLAND_ONLY=True, ADD_TO_MAP=False, axes=None, \
                          VERBOSE=False):
    '''Threshold histo (retain 1.5< MaxLAI< 5) and then rel frequency'''
    vi_annual_reduced = modis_annual_vi_per_district(year, vi_band, aoi, \
                                                     TIME_REDUCER=TIME_REDUCER, SPACE_DERUCER='None', \
                                                     CROPLAND_ONLY=CROPLAND_ONLY, ADD_TO_MAP=ADD_TO_MAP,
                                                     VERBOSE=VERBOSE)

    if axes is None:  # setup a new fig
        fig, axes = plt.subplots(math.ceil(26 / 3), 3, figsize=(24, 16), sharex=True, sharey=False)
        fig.suptitle(f'Histograms Per District')
        plt.setp(axes[-1, :], xlabel='Lai')
        plt.setp(axes[:, 0], ylabel='Pixel frequencies')  # not intergers due to rescaled to 500m. counts intersects
        axes = axes.flatten()

    '''#unbounded histo per district
    histo_per_district = vi_annual_reduced.reduceRegions(collection=aoi,\
                                reducer=ee.Reducer.histogram(16),scale=500) 
    #print(histo_per_district.first().get('ADM2_NAME').getInfo())
    plot_histo_per_county_v0(axes, histo_per_district, year)
    '''

    # bounded histo between 1.0 <= MaxLAI <= 5.0
    # binWidth=0.5, so fixedHistogram(1.0, 6.0, 10) or fixedHistogram(1.0, 5.0, 8)
    fixhisto_per_district = vi_annual_reduced.reduceRegions(collection=aoi, \
                                                            reducer=ee.Reducer.fixedHistogram(1.0, 6.0, 10), scale=500)
    df_lai_params = plot_histo_per_county_v_fixed_histo(axes, fixhisto_per_district, year)

    return df_lai_params


def generate_era5_and_yield_DF(df_northern_maize, era5_bands, INCLUDE_LAI=True, VERBOSE=False):
    '''Make a DF of dependent variable, YIELD
    ATTENTION: max_MODIS_LAIs and GAUL_district_names are GLOBAL VARIABLES'''

    df_yield = df_northern_maize[['DISTRICT', 'YEAR', 'YIELD']]
    print(f'{df_yield.size} yield samples found in MOFA record')
    if VERBOSE: print(df_yield.head(5))

    if INCLUDE_LAI:
        '''Make a DF of MAXLai data'''
        df_lai = pd.DataFrame(max_MODIS_LAIs, columns=range(MIN_YEAR, END_YEAR + 1))
        # df_lai.head()
        df_lai.insert(0, 'DISTRICT', GAUL_district_names)
        df_lai = df_lai.melt(id_vars=['DISTRICT']).rename( \
            columns={'variable': 'YEAR', 'value': 'MaxLAI'})
        print(f'{df_lai.size} samples found in MODIS Max LAI record')

        '''ADDING 4 LAI HISTO PARAMETERS '''
        df_histo = df_northern_maize[['DISTRICT', 'MAX_HISTO_FREQ', 'MAXLAI_PEAK', 'LAI_STDEV', 'LAI_CV']]
        df_lai = df_lai.merge(df_histo, how='left', on='DISTRICT')
        if VERBOSE:
            print('*************TEST LAI HISTO PARAMS')
            print(df_lai.head(5))

    '''Make a DF of ERA5 data'''
    for i, era5_band in enumerate(era5_bands):
        print(f'Retreiving ERA5 monthly {era5_band}...')
        weather_district, weather_ts = era5_ts_per_district(range(MIN_YEAR, END_YEAR + 1), \
                                                            northern_district, band=era5_band, CROPONLY=True)
        if VERBOSE: print(weather_district.size, weather_ts.size, len(weather_ts))

        df_weather = pd.DataFrame(weather_ts, columns=range(MIN_YEAR, END_YEAR + 1))
        df_weather.insert(0, 'DISTRICT', weather_district)
        # print(df_weather.head())

        df_weather = df_weather.melt(id_vars=['DISTRICT']).rename( \
            columns={'variable': 'YEAR', 'value': era5_band})
        # print(df_weather.head(50))

        if i == 0:
            df_era5 = df_weather
        else:
            df_era5 = df_era5.merge(df_weather, on=["DISTRICT", "YEAR"])
            # print(df_output.head(50))

    '''Merge LAI and ERA5 DFs'''
    if INCLUDE_LAI:
        df_indp = df_era5.merge(df_lai, on=["DISTRICT", "YEAR"])
    else:
        df_indp = df_era5
    print(f'{df_indp.size} matching samples found in ERA5 record')
    if VERBOSE: print(df_indp.head())

    if df_indp.size > 0:
        df_yield = df_yield.merge(df_indp, on=["DISTRICT", "YEAR"])
        # print(df_yield.head)
        # df_yield = df_yield[['DISTRICT', 'YEAR', 'YIELD']]
        print(f'{df_yield.size} matching yield samples found')
        if VERBOSE: print(df_yield.head)

        if df_yield.size > 0:

            # denpendent variable should be MOFA Yield
            if INCLUDE_LAI:
                era5_bands.append('MaxLAI')
                '''ADDING 4 LAI HISTO PARAMETERS '''
                era5_bands.append('MAX_HISTO_FREQ')
                era5_bands.append('MAXLAI_PEAK')
                era5_bands.append('LAI_STDEV')
                era5_bands.append('LAI_CV')
            print('Indepedent variables are:', era5_bands)
            df_inp = df_yield[era5_bands]
            return df_inp, df_yield['YIELD']
        else:
            print(f'Not enough YIELD data')
            return df_inp, pd.DataFrame()  # return an empty DF
    else:
        print(f'Not enough ERA5 data')
        return pd.DataFrame(), pd.DataFrame()  # return two empty DFs


'''THIS BLOCK CAN REPLACE THE ABOVE BLOCK OF FUNCTION modis_vi_per_district() BY SETTING CROP_ONLY=Flase'''


def modis_vi_per_district_cropmask(year, vi_band, aoi, TIME_REDUCER='max', SPACE_DERUCER='mean', \
                                   CROP_ONLY=True, ADD_TO_MAP=False):
    # Firstly, we calculate the mean VI (annual average VI) per district:
    vi_all, vi_annual_reduced = load_modis_band(vi_band, year, aoi, TIME_REDUCER)

    if CROP_ONLY:
        # we mask the VI images by LC type of croplands only
        lc_crop, lc_all = load_modis_lc('LC_Type2', year, aoi, cropclasses=[10])
        # print(lc_crop.getInfo())
        if ADD_TO_MAP:
            Map.addLayer(lc_crop, {'min': 0, 'max': 1}, f"Croplands {year}", opacity=0.7)

        '''mask croplands only'''
        vi_annual_reduced = vi_annual_reduced.mask(lc_crop)

    if SPACE_DERUCER != 'None':
        space_methods = {'mean': ee.Reducer.mean, \
                         'max': ee.Reducer.max, \
                         'min': ee.Reducer.min, \
                         'median': ee.Reducer.median}

        if SPACE_DERUCER in space_methods:
            vi_per_district = vi_annual_reduced.reduceRegions(collection=aoi, \
                                                              reducer=space_methods[SPACE_DERUCER](), \
                                                              scale=500)
        else:
            raise Exception("Method %s not implemented" % SPACE_DERUCER)

    list_of_districts = vi_per_district.aggregate_array('ADM2_NAME').getInfo()
    list_of_meanVIs = vi_per_district.aggregate_array(SPACE_DERUCER).getInfo()

    if ADD_TO_MAP:
        Map.addLayer(vi_per_district, {'min': 0.0, 'max': 2.5},
                     f'{SPACE_DERUCER} {vi_band} {year} per district (Cropland Only)')

    stats = vi_per_district.aggregate_stats('mean')
    print(year, vi_band, stats.getInfo())

    return np.asarray(list_of_districts), np.asarray(list_of_meanVIs)


def mean_ndvi_per_district(years, aoi, CROPONLY=False):
    for year in years:
        # reduce to per district values (district_names should be constant over years)
        district_names, mean_VI_per_district = modis_annual_vi_per_district(year, 'NDVI', aoi, \
                                                                            CROP_ONLY=CROPONLY, ADD_TO_MAP=False)
        # print(type(mean_VI_per_district))
        if year == years[0]:
            mean_ndvi = mean_VI_per_district
            # print(type(mean_ndvi), mean_ndvi.size)
        else:
            mean_ndvi = np.vstack((mean_ndvi, mean_VI_per_district))

    # print('No. of districts: ', len(district_names))
    # print('No. of years: ', len(mean_ndvi))
    # print(type(mean_ndvi), mean_ndvi.size, len(mean_ndvi))
    mean_ndvi = np.transpose(mean_ndvi)

    plt.figure(figsize=(16, 12))

    for i, district in enumerate(district_names):
        plt.plot(years, mean_ndvi[i], label=district)
    if CROPONLY:
        plt.title('Average NDVI per district (Croplands only)')
    else:
        plt.title('Average NDVI per district (all land covers)')
    plt.legend()

    return district_names, mean_ndvi


def max_lai_per_district(years, aoi, CROPONLY=False):
    for year in years:
        # reduce to per district values (district_names should be constant over years)
        district_names, reduced_VI_per_district = modis_annual_vi_per_district(year, 'Lai', aoi, \
                                                                               TIME_REDUCER='max', SPACE_DERUCER='mean', \
                                                                               CROP_ONLY=CROPONLY, ADD_TO_MAP=False)
        # print(type(mean_VI_per_district))
        if year == years[0]:
            max_lai = reduced_VI_per_district
            # print(type(mean_ndvi), mean_ndvi.size)
        else:
            max_lai = np.vstack((max_lai, reduced_VI_per_district))

    max_lai = np.transpose(max_lai)

    plt.figure(figsize=(16, 12))

    for i, district in enumerate(district_names):
        plt.plot(years, max_lai[i], label=district)
    if CROPONLY:
        plt.title('Max LAI per district (Croplands only)')
    else:
        plt.title('Max LAI per district (all land covers)')
    plt.legend()

    return district_names, max_lai


def load_era_band(band, year, aoi, ADD_TO_MAP=False, VERBOSE=False):
    weather_in_season = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
        .filterBounds(aoi) \
        .select(band) \
        .filterDate(f'{year}-06-01', f'{year}-11-30')  # only during MAize crop season
    size = weather_in_season.size()
    if VERBOSE:
        print(f'In {year}, {size.getInfo()} image(s) found.')

    if ADD_TO_MAP:
        Map.addLayer(weather_in_season, era5_visualization, "Air temperature [K] at 2m height")
    return weather_in_season


def era4_value_per_district(year, aoi, band='temperature_2m', \
                            VERBOSE=False, ADD_TO_MAP=False):
    era5_in_season = load_era_band(band, year, aoi)
    era5_seasonal_accum = era5_in_season.sum()

    accum_per_district = era5_seasonal_accum.reduceRegions(collection=aoi, \
                                                           reducer=ee.Reducer.mean(), \
                                                           scale=11.1 * 1000)  # ERA5 resoltuion = 0.1 deg

    # print('------', accum_per_district.getInfo()) # this should be a 'FeatureCollection'
    list_of_districts = accum_per_district.aggregate_array('ADM2_NAME').getInfo()
    if VERBOSE:
        print(len(list_of_districts), list_of_districts)  # this line prints all district names

    list_of_means = accum_per_district.aggregate_array('mean').getInfo()
    if VERBOSE:
        print(len(list_of_means), list_of_means)  # this line prints all mean values

    if ADD_TO_MAP:
        Map.addLayer(accum_per_district, {'min': 0.0, 'max': 300 * 5}, f'Mean {band} {year} per district')

    stats = accum_per_district.aggregate_stats('mean')
    if VERBOSE:
        print(year, band, stats.getInfo())

    return np.asarray(list_of_districts), np.asarray(list_of_means)


def era5_ts_per_district(years, aoi, band='temperature_2m', CROPONLY=False,
                         PLOT=False):  # resolution so low, no need for CROPONLY
    for year in years:
        # reduce to per district values (district_names should be constant over years)
        district_names, mean_era5_per_district = era4_value_per_district(year, aoi, band, ADD_TO_MAP=False)
        # print(type(mean_era5_per_district))
        if year == years[0]:
            mean_era5 = mean_era5_per_district
            # print(type(mean_ndvi), mean_ndvi.size)
        else:
            mean_era5 = np.vstack((mean_era5, mean_era5_per_district))

    # print('No. of districts: ', len(district_names), '; No. of years: ', len(mean_era5))
    # print(type(mean_ndvi), mean_ndvi.size, len(mean_ndvi))
    mean_era5 = np.transpose(mean_era5)

    if PLOT:
        plt.figure(figsize=(16, 12))

        for i, district in enumerate(district_names):
            plt.plot(years, mean_era5[i], label=district)
        if CROPONLY:
            plt.title(f'Average {band} per district (Croplands only)')
        else:
            plt.title(f'Average {band} per district (all land covers)')
        plt.legend()

    return district_names, mean_era5

