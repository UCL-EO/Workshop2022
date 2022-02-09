import ee
import datetime
import shutil
import imageio
import requests
import numpy as np
from retry import retry
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import matplotlib as mpl
from osgeo import gdal
from pyproj import Proj, Transformer
from PIL import Image
import os
import multiprocessing

mpl.use('Agg')

import pylab as plt

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

def get_s2_files(geometry, start, end):
# create a combined filter from one bounds filter
    # and a timing filter
    criteria = ee.Filter.And( ee.Filter.bounds(geometry), 
                              ee.Filter.date(start, end))

    # load s2 and s2cloud image collection
    s2 = ee.ImageCollection('COPERNICUS/S2')
    s2_files = s2.filter(criteria).aggregate_array('system:index').getInfo()
    
    return s2_files


@retry(tries=10, delay=1, backoff=2)
def download_image(s2_file, geometry, field_name):
    
    image = ee.Image('COPERNICUS/S2/%s'%s2_file)#.filterMetadata('PRODUCT_ID', 'equals', s2_file).first()
    
    image = image.visualize(bands = ['B4', 'B3', 'B2'], min=0, max=2500)
    url = image.getThumbURL({
          'region': geometry,
          'dimensions': 256,
          'format': 'png'})
    
    r = requests.get(url, stream=True)
    print(url)
    if r.status_code != 200:
        r.raise_for_status()
    filename = './data/S2_thumbs/S2_%s_%s.png'%(s2_file, field_name)
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)

#     url = image.getDownloadURL({
#         'region': geometry,
#         'scale': 10, 
#         'format': 'GEO_TIFF',
#         'bands': ['B4', 'B3', 'B2']
#     })
#     print(url)
#     g = gdal.Open('/vsicurl_streaming/' + url)
#     if g is None:
#         print(url)
#         raise
#     else:
#         arr = g.ReadAsArray()

#     epsg_code = 3857
#     ds = gdal.Warp('', g, format = 'MEM', dstSRS='EPSG:%d'%epsg_code, srcNodata=0, dstNodata=0)
#     geo_trans = ds.GetGeoTransform()
    
#     x_min = geo_trans[0]
#     y_min = geo_trans[3]
#     x_max = geo_trans[0] + ds.RasterXSize * geo_trans[1]
#     y_max = geo_trans[3] + ds.RasterYSize * geo_trans[5]

#     coords = np.array([[x_min, y_min], [x_max, y_max]])
    
#     x_coords = [x_min, x_max]
#     y_coords = [y_min, y_max]
    
#     transformer = Transformer.from_crs('epsg:%d'%epsg_code, 'epsg:4326')
#     (x_min, x_max), (y_min, y_max) = transformer.transform(x_coords,y_coords)
    
#     bounds = ((x_min, y_max), (x_max, y_min))
#     # bounds = ((y_max, x_min), (y_min, x_max))
#     img_arr = ds.ReadAsArray() / 10000 * 256 * 4
#     img_mask = np.any(img_arr == 0, axis=0)
    
#     alpha = np.ones(img_mask.shape) * 255
#     alpha[img_mask] = 0
    
#     rgba = np.concatenate([img_arr, [alpha]])
    
    
#     arr = np.clip(rgba, 0, 255).astype(np.uint8).transpose(1,2,0)
#     im = Image.fromarray(arr)
#     date = '-'.join([s2_file[:4], s2_file[4:6], s2_file[6:8]])
#     fname = "./data/S2_thumbs/S2_%s.png"%date
#     im.save(fname)
    
    # home = os.getcwd()
    # cwd = '/files/' + '/'.join(home.split('/')[3:])
    # base_url = my_map.window_url.split('/lab/')[0] + cwd
    # url = base_url + '/data/S2_thumbs/S2_%s.png'%date
    # print(url)
    # print(bounds)
    # image = ImageOverlay(
    #     url=url,
    #     bounds = bounds,
    #     # name = "S2_%s.png"%date
    # )
    # my_map.add_layer(image)   
    
#     print(ds.GetGeoTransform())
#     print(ds.GetProjection())
#     print(ds.ReadAsArray().shape)
#     r = requests.get(url, stream=True)
#     print(url)
#     if r.status_code != 200:
#         r.raise_for_status()
#     filename = './data/S2_thumbs/S2_%s_%s.tif'%(s2_file, field_name)
#     with open(filename, 'wb') as out_file:
#         shutil.copyfileobj(r.raw, out_file)

    
    # image = image.visualize(bands = ['B4', 'B3', 'B2'], min=0, max=2500)
    # url = image.getThumbURL({
    #       'region': geometry,
    #       'dimensions': 256,
    #       'format': 'png'})
    # content = imageio.imread(url)
    # arr = np.asarray(content)
    
#     date = '-'.join([s2_file[:4], s2_file[4:6], s2_file[6:8]])
#     plt.figure()
#     plt.imshow(arr)
#     date = '-'.join([s2_file[:4], s2_file[4:6], s2_file[6:8]])
#     plt.text(180, 20, '%s'%date, color = 'red', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5})
#     filename = './data/S2_thumbs/S2_%s_%s.png'%(s2_file, field_name)
#     plt.axis('off') 
#     plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
#     plt.close()
    
    # return arr

def get_s2_over_field(field_name, geom, start, end):
    s2_files = get_s2_files(geom, start, end)
    par = partial(download_image, geometry=geom, field_name = field_name)
    pool = multiprocessing.Pool(10)
    pool.map(par, s2_files)
    pool.close()
    return s2_files

field_name = '1029ZIN'
start = '2021-01-01'
end = '2021-12-31'
geom = ee.Geometry.Point(-0.615656152367592, 9.38136229854884).buffer(1000)
s2_files = get_s2_over_field(field_name, geom, start, end)

# s2_files = get_s2_files(geom, start, end)
# import time



for s2_file in s2_files:
    bounds = ((9.390428444799689, -0.6064161680076744), (9.372231832296013, -0.6247681500722374))
    date = '-'.join([s2_file[:4], s2_file[4:6], s2_file[6:8]])
    fname = "./data/S2_thumbs/S2_%s.png"%s2_file
    
    home = os.getcwd()
    cwd = '/files/' + '/'.join(home.split('/')[3:])
    base_url = my_map.window_url.split('/lab/')[0] + cwd
    url = base_url + '/data/S2_thumbs/S2_%s.png'%s2_file
    print(url)
    print(bounds)
    image = ImageOverlay(
        url=url,
        bounds = bounds,
        name = "S2_thumbs_%s"%date
    )
    for layer in my_map.layers:
        if 'S2_thumbs_'in layer.name:
            my_map.remove_layer(layer)
            
    my_map.add_layer(image)   
    
#     download_image(s2_file, geometry=geom, field_name = field_name)

                
    # time.sleep(0.5)
    