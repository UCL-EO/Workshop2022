import os
import datetime
import requests
from io import BytesIO
import matplotlib.cm as cm
import numpy as np
import pyproj
# import imageio
import io
import json
import pandas as pd
import pylab as plt
import matplotlib as mpl
from pyproj import Proj, Transformer
from PIL import Image, ImageFont, ImageDraw 
from pygeotile.tile import Tile
from ipyleaflet import ImageOverlay, GeoJSON, WidgetControl
from ipywidgets import Dropdown
from shapely import geometry
import asyncio
from osgeo import gdal, ogr
gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator

def read_wofost_data(fname):
    f = np.load(fname, allow_pickle=True)
    parameters = f.f.parameters
    t_axis = f.f.t_axis
    samples = f.f.samples
    lais = f.f.lais
    yields = f.f.yields
    DVS = f.f.DVS
    print("loading simulations")
    doys = [int(datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%j')) for i in t_axis]
    return parameters, t_axis, samples, lais, yields, DVS, doys

# parameters, t_axis, samples, lais, yields, DVS, doys = read_wofost_data('data/wofost_sims_dvs125.npz')


def get_field_center(field_name, field_lut):
    field = field_lut[field_name]
    field_poly = geometry.Polygon(field['geometry']['coordinates'][0])
    center = field_poly.centroid.y, field_poly.centroid.x
    return center

def create_field_dropdown(codes):
    ### create field selections
    field_dropdown = Dropdown(
        options=codes,
        value=codes[0],
        description="Field ID:",
    )
    field_dropdown_control = WidgetControl(widget=field_dropdown, position="bottomleft")
    return field_dropdown_control, field_dropdown


def load_field_points_polys():
    with open('./data/Biophysical_Data_Collection_Points_V1.geojson', 'r') as f:
        data = json.load(f)
    points = GeoJSON(
        data=data,
        point_style={'radius': 5, 'color': 'blue', 'fillOpacity': 0.5, 'fillColor': 'blue', 'weight': 0.1},
        # style={
        #     'opacity': 1, 'dashArray': '0', 'fillOpacity': 0.2, 'weight': 0.01
        # },
        hover_style={
            'color': 'white', 'dashArray': '0', 'fillOpacity': 1
        },
        # style_callback=random_color
    )
    
    def random_color(feature):
        return {
            'color': 'black',
            'fillColor': np.random.choice(['red', 'yellow', 'green', 'orange']),
        }
    
    with open('./data/Biophysical_Data_Collection_Polygons_V1.geojson', 'r') as f:
        data = json.load(f)

    fields = GeoJSON(
        data=data,
        style={
            'opacity': 1, 'dashArray': '0', 'fillOpacity': 0, 'weight': 1
        },
        hover_style={
            'color': 'white', 'dashArray': '0', 'fillOpacity': 0
        },
        style_callback=random_color
    )

    return points, fields


def add_bing_to_map(my_map):
    zoom = int(my_map.zoom)
    tile = Tile.for_latitude_longitude(*my_map.center, zoom)
    x, y = tile.tms

    for i in range(x- 3, x + 4):
        for j in range(y - 2, y + 3):
            tile = Tile.from_tms(i, j, zoom)
            url = "http://ecn.t3.tiles.virtualearth.net/tiles/a%s.png?g=1"%tile.quad_tree

            image = ImageOverlay(
                url=url,
                bounds = tile.bounds,
                name = 'bing_basemap_%d'%zoom
            )
            my_map.add_layer(image)     
            
            
def get_field_yield():
    df = pd.read_csv('data/Ghana_ground_data_v2.csv')
    yield_df = pd.read_csv('data/Yield_Maize_Biomass_V2.csv').iloc[:, :3]
    codes = np.array([i[:-2] for i in yield_df.FID]).reshape(-1, 3)[:, 0]
    yields = np.array(yield_df.iloc[:, 1]).reshape(-1, 3)
    field_yields = dict(zip(codes, yields.tolist()))
    return field_yields, np.unique(codes).tolist()


def get_lai_color_bar():
    
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.05, 0.8, 0.5, 0.07])
    cmap = plt.cm.YlGn
    norm = mpl.colors.Normalize(vmin=0, vmax=2.5)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    lai_colorbar_f = io.BytesIO()
    plt.savefig(lai_colorbar_f, bbox_inches='tight', format='png', pad_inches=0)
    plt.close()
    return lai_colorbar_f


def get_pixel(location, field_name):
    lat, lon = location
    npz_name = '%s_bios_planet_only_v6.npz'%field_name
    if not os.path.exists('./data/' + npz_name):
        tsen_url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Tsen/'
        url = tsen_url + npz_name
        r = requests.get(url)
        if r.status_code != 200:
            r.raise_for_status()
        with open('./data/' + npz_name, 'wb') as f:
            f.write(r.content)

    f = np.load('./data/' + npz_name, allow_pickle=True)
    projectionRef = str(f.f.projectionRef)
    geo_trans = f.f.geotransform
    doys = f.f.doys
    valid_mask = f.f.valid_mask
    alpha = (valid_mask * 255.).astype(np.uint8)
    lai = f.f.mean_bios_all[:, 4]
    cab = f.f.mean_bios_all[:, 1]

    pj1 = Proj(projectionRef)
    transformer = Transformer.from_crs('EPSG:4326', pj1.crs)
    x, y = transformer.transform(lat, lon)
    inds = np.where(valid_mask)
    pix_x = int((x - geo_trans[0]) / geo_trans[1] + 0.5)
    pix_y = int((y - geo_trans[3]) / geo_trans[5] + 0.5)
    mm = (inds[0] == pix_y) & (inds[1] == pix_x)
    
    inds = f.f.unique_neighbour_inverse_inds[mm][0][:200]
    sels = f.f.unique_neighbour_s2_refs[:, :, inds][[0, 1, 2, 7]]
    bios = f.f.unique_neighbour_orig_bios[:, :, inds][[4]]
    ndvi = (sels[3] - sels[2]) / (sels[3] + sels[2])
    
    sels = np.concatenate([sels, ndvi[None], bios])
    planet_sur = f.f.s2_sur_all[:, :, mm].squeeze()
             
    return doys, lai[mm].ravel(), cab[mm].ravel(), sels, lai, planet_sur

def get_lai_gif(field_name):
    npz_name = '%s_bios_planet_only_v6.npz'%field_name
    if not os.path.exists('./data/' + npz_name):
        tsen_url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Tsen/'
        url = tsen_url + npz_name
        r = requests.get(url)
        if r.status_code != 200:
            r.raise_for_status()
        with open('./data/' + npz_name, 'wb') as f:
            f.write(r.content)
    
    f = np.load('./data/' + npz_name, allow_pickle=True)
    projectionRef = str(f.f.projectionRef)
    geo_trans = f.f.geotransform
    doys = f.f.doys
    valid_mask = f.f.valid_mask
    alpha = (valid_mask * 255.).astype(np.uint8)
    lai = f.f.mean_bios_all[:, 4]
    lai_scale = f.f.mean_bio_scales_all[:, 4]
    
    med_lai = np.nanmedian(lai, axis=0)

    max_lai = np.zeros(valid_mask.shape)
    max_lai[valid_mask] = lai_scale
    
    yld = max_lai * 5000 - 1500
    yld = np.maximum(yld, 0)
    
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_axes([0.05, 0.8, 0.5, 0.07])
    cmap = plt.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=yld[valid_mask].min(), vmax= yld[valid_mask].max())
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    yield_colorbar_f = io.BytesIO()
    plt.savefig(yield_colorbar_f, bbox_inches='tight', format='png', pad_inches=0)
    plt.close()

    
    empirical_yield_min =  yld[valid_mask].min()
    empirical_yield_max =  yld[valid_mask].max()
    
    norm_yld = (yld - yld[valid_mask].min()) / (yld[valid_mask].max() - yld[valid_mask].min())
    cmap = cm.RdYlGn
    greyscale = cmap(norm_yld, bytes=True)
    greyscale[:, :, -1] = alpha
    # greyscale = np.concatenate([greyscale, alpha[:, :, None]])
    
    img = Image.fromarray(greyscale, mode='RGBA')
    
    scale = 256 / img.height
    new_height = int(scale * img.height)
    new_width = int(scale * img.width)

    img = img.resize(( new_width, new_height), resample = Image.NEAREST)
    this_alpha = img.getchannel('A')
    img = img.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
    mask = Image.eval(this_alpha, lambda a: 255 if a <= 5 else 0)

    # Paste the color of index 255 and use alpha as a mask
    img.paste(255, mask)
    img.info['transparency'] = 255

    fname = './data/S2_thumbs/S2_%s_yield.png'%(field_name)
    img.save(fname)
    
    
    pj1 = pyproj.Proj(projectionRef)
    transformer = Transformer.from_crs(pj1.crs, 'EPSG:4326')

    x_min = geo_trans[0]
    y_min = geo_trans[3]
    x_max = geo_trans[0] + valid_mask.shape[1] * geo_trans[1]
    y_max = geo_trans[3] + valid_mask.shape[0] * geo_trans[5]

    coords = np.array([[x_min, y_min], [x_max, y_max]])

    x_coords = [x_min, x_max]
    y_coords = [y_min, y_max]

    (x_min, x_max), (y_min, y_max) = transformer.transform(x_coords,y_coords)

    bounds = ((x_min, y_max), (x_max, y_min))
    print(bounds)


    cmap = cm.YlGn
    frames = []
    for i in range(len(doys)):
        lai_map = np.zeros(valid_mask.shape)
        lai_map[valid_mask] = lai[:, i]
        greyscale = cmap(lai_map / 2.5, bytes=True)
        greyscale[:, :, -1] = alpha
        # greyscale = np.concatenate([greyscale, alpha[:, :, None]])
        date = datetime.datetime(2021, 1, 1) + datetime.timedelta(days=int(doys[i]) -1)

        img = Image.fromarray(greyscale, mode='RGBA')

        scale = 256 / img.height
        new_height = int(scale * img.height)
        new_width = int(scale * img.width)

        img = img.resize(( new_width, new_height), resample = Image.NEAREST)
        draw = ImageDraw.Draw(img, "RGBA")
        #strptimet = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        font = font = ImageFont.load_default()
        text = date.strftime('%Y-%m-%d')
        w, h = font.getsize(text)
        x, y = 5, 5  


        TINT_COLOR = (0, 0, 0)  # Black
        TRANSPARENCY = 0.25  # Degree of transparency, 0-100%
        TRANSPARENCY = 0.75  # Degree of transparency, 0-100%
        
        OPACITY = int(255 * TRANSPARENCY)
        draw.rectangle((x, y, x + w, y + h), fill=TINT_COLOR + (OPACITY,), )
        draw.rectangle((x-2, y-2, x + w+2, y + h+2), outline=(0, 0, 0, 127), width=1)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        this_alpha = img.getchannel('A')
        img = img.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
        mask = Image.eval(this_alpha, lambda a: 255 if a <= 5 else 0)

        # Paste the color of index 255 and use alpha as a mask
        img.paste(255, mask)
        img.info['transparency'] = 255
        
        fname = './data/S2_thumbs/S2_%s_lai_%03d.png'%(field_name, doys[i])
        img.save(fname)
        
        frames.append(img)
        
        # frames.append(np.asarray(img))
        
        
    fp_out = './data/S2_thumbs/S2_%s_lai.gif'%(field_name)
    # imageio.mimsave(fp_out, frames, 'GIF', duration=0.2)
    # url = base_url + '/output.gif'
    # print(url)
    
    frames[0].save(fp_out, save_all=True, append_images=frames[1:], loop=0, duration=200, optimize=False)
    
    return 'data/S2_thumbs/S2_%s_lai.gif'%field_name, bounds, doys, yield_colorbar_f, med_lai, empirical_yield_min, empirical_yield_max
    # print(bounds)
    
def get_field_bounds(field_name):
    npz_name = '%s_bios_planet_only_v6.npz'%field_name
    if not os.path.exists('./data/' + npz_name):
        tsen_url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Tsen/'
        url = tsen_url + npz_name
        r = requests.get(url)
        if r.status_code != 200:
            r.raise_for_status()
        with open('./data/' + npz_name, 'wb') as f:
            f.write(r.content)
    
    f = np.load('./data/' + npz_name, allow_pickle=True)
    projectionRef = str(f.f.projectionRef)
    geo_trans = f.f.geotransform
    doys = f.f.doys
    valid_mask = f.f.valid_mask
    alpha = (valid_mask * 255.).astype(np.uint8)
    lai = f.f.mean_bios_all[:, 4]



    pj1 = pyproj.Proj(projectionRef)
    transformer = Transformer.from_crs(pj1.crs, 'EPSG:4326')

    x_min = geo_trans[0]
    y_min = geo_trans[3]
    x_max = geo_trans[0] + valid_mask.shape[1] * geo_trans[1]
    y_max = geo_trans[3] + valid_mask.shape[0] * geo_trans[5]

    coords = np.array([[x_min, y_min], [x_max, y_max]])

    x_coords = [x_min, x_max]
    y_coords = [y_min, y_max]

    (x_min, x_max), (y_min, y_max) = transformer.transform(x_coords,y_coords)

    bounds = ((x_min, y_max), (x_max, y_min))

    return bounds, doys


def da_pix(sels, planet_sur, u_thresh = 2):
    sel_s2_refs_med = np.nanmedian(sels[:4], axis=2) 
    residuals = planet_sur - sel_s2_refs_med
    residuals_abs = abs(residuals)
    MAD = np.nanmedian(residuals_abs, axis=1)
    S = MAD /  0.6745
    u = residuals_abs / S[:, None]
    u_mask = (u <= u_thresh).all(axis=0)

    sel_s2_refs = sels[:4, u_mask]
    s2_sur = planet_sur[:, u_mask]
    s2_sur_unc = planet_sur[:, u_mask] * 0.10


    # s2_sur[~u_mask] = np.nan

    diff = (s2_sur[:, :, None] - sel_s2_refs)
    diff = np.nansum(diff**2 * s2_sur_unc[:, :, None]**2, axis=(0,1))
    inds = np.argsort(diff)

    sel_nums = 10
    sel_inds = inds[:sel_nums]
    diff = diff[inds][:sel_nums]

    weight = 1 / diff
    # weight = np.exp(-1 * diff / 2)

    weight = weight / weight.sum()

    mean_ref        = np.sum(sel_s2_refs[:, :, sel_inds] * weight[None, None], axis=2) 
    mean_bios       = np.sum(sels[4:][:, :, sel_inds] * weight[None, None], axis=2)

    correction = sel_nums / (sel_nums - 1)
    weight = weight * correction

    std_ref        = np.sqrt(np.sum((sel_s2_refs[:, :, sel_inds] - mean_ref [:, :, None])**2 * weight[None, None], axis=2))
    std_bios       = np.sqrt(np.sum((sels[4:][:, :, sel_inds] - mean_bios[:, :, None])**2 * weight[None, None], axis=2))

    return mean_ref, mean_bios, std_ref, std_bios, sel_inds, u_mask




def load_s2_bios(field_id):
    npz_name = '%s_bios_planet_prior_v1.npz'%field_id
    if not os.path.exists('./data/' + npz_name):
        tsen_url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Tsen/'
        url = tsen_url + npz_name
        r = requests.get(url)
        if r.status_code != 200:
            r.raise_for_status()
        with open('./data/' + npz_name, 'wb') as f:
            f.write(r.content)
    f = np.load('./data/' + npz_name)
    bios = f.f.mean_bios_all.transpose(2,1,0)
    valid_mask = f.f.valid_mask
    temp = np.zeros(bios.shape[:2] + valid_mask.shape) * np.nan
    temp[:, :, valid_mask] = bios
    doys = f.f.doys
    return temp, doys

def get_field_geo_transform_S2_30PYR(field_id):
    temp_file_url = '/vsicurl/https://gws-access.jasmin.ac.uk/public/nceo_ard/S2/30/P/YR/S2B_MSIL1C_20180914T102019_N0206_R065_T30PYR_20180914T173602.SAFE/GRANULE/L1C_T30PYR_A007956_20180914T102951/IMG_DATA/T30PYR_20180914T102019_B02_sur.tif'
    field_polys = 'data/Biophysical_Data_Collection_Polygons_V1.geojson'


    g = ogr.Open(field_polys)
    l = g.GetLayer(0)
    layername = l.GetName()
    csql = "SELECT * FROM %s WHERE Field_ID='%s'"%(layername, field_id)

    gg = gdal.Warp('', temp_file_url, format='MEM', outputType=gdal.GDT_Float32, dstNodata=np.nan, 
                    cutlineDSName=field_polys, 
                    cropToCutline=True, cutlineSQL =csql)
    
    projectionRef, geo_trans = gg.GetProjectionRef(), gg.GetGeoTransform()
    

    
    return projectionRef, geo_trans

def get_s2_bounds(projectionRef, geo_trans, shape):
    pj1 = Proj(projectionRef)
    transformer = Transformer.from_crs(pj1.crs, 'EPSG:4326', always_xy=True)
    
    x_min = geo_trans[0]
    y_max = geo_trans[3]
    x_max = geo_trans[0] + shape[1] * geo_trans[1]
    y_min = geo_trans[3] + shape[0] * geo_trans[5]

    coords = np.array([[x_min, y_min], [x_max, y_max]])

    x_coords = [x_min, x_max]
    y_coords = [y_min, y_max]

    (x_min, x_max), (y_min, y_max) = transformer.transform(x_coords,y_coords)

    bounds = ((y_min, x_min), (y_max, x_max))
    return bounds

def latlon_2_xy(lat, lon, projectionRef, geo_trans):
    
    pj1 = Proj(projectionRef)
    transformer = Transformer.from_crs('EPSG:4326', pj1.crs)
    x, y = transformer.transform(lat, lon)
    # inds = np.where(valid_mask)
    pix_y = int((x - geo_trans[0]) / geo_trans[1])
    pix_x = int((y - geo_trans[3]) / geo_trans[5])
    return pix_x, pix_y


def get_pixel_s2_bios(lat, lon, s2_projectionRef, s2_geo_trans, s2_bios):
    s2_pix_x, s2_pix_y = latlon_2_xy(lat, lon, s2_projectionRef, s2_geo_trans )
    print(s2_pix_x, s2_pix_y)
    s2_pixel_bios = s2_bios[:, :, s2_pix_x, s2_pix_y].T

    s2_bio_names = ['n', 'cab', 'cm', 'cw', 'lai', 'ala', 'cbrown']
    s2_pix_bio_dict = dict(zip(s2_bio_names, s2_pixel_bios))
    return s2_pix_bio_dict

def get_wofost_yield(field_id, empirical_yield_min, empirical_yield_max):
    
    f = np.load('data/pixel_yield.npz')
    yld = f[field_id]
    valid_mask = np.isfinite(yld)
    alpha = (valid_mask * 255.).astype(np.uint8)
    
    yld_max, yld_min = np.nanmax(yld[valid_mask]), np.nanmin(yld[valid_mask])
    
    norm_yld = (yld - yld_min) / (yld_max - yld_min)
    cmap = plt.cm.RdYlGn
    greyscale = cmap(norm_yld, bytes=True)
    
    greyscale[:, :, -1] = alpha
    # greyscale = np.concatenate([greyscale, alpha[:, :, None]])
    
    img = Image.fromarray(greyscale, mode='RGBA')

    scale = 256 / img.height
    new_height = int(scale * img.height)
    new_width = int(scale * img.width)

    img = img.resize((new_width, new_height), resample = Image.NEAREST)
    
    TINT_COLOR = (0, 0, 0)  # Black
    TRANSPARENCY = 0.25  # Degree of transparency, 0-100%
    TRANSPARENCY = 0.75  # Degree of transparency, 0-100%

    
    this_alpha = img.getchannel('A')
    img = img.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
    mask = Image.eval(this_alpha, lambda a: 255 if a <= 5 else 0)

    # Paste the color of index 255 and use alpha as a mask
    img.paste(255, mask)
    img.info['transparency'] = 255

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.05, 0.8, 0.5, 0.07])
    
    norm = mpl.colors.Normalize(vmin=yld_min, vmax=yld_max)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    lai_colorbar_f = io.BytesIO()
    plt.savefig(lai_colorbar_f, bbox_inches='tight', format='png', pad_inches=0)
    plt.close()
    
    fname = 'data/S2_thumbs/S2_%s_yield_wofost.png'%(field_id)
    img.save(fname)
    
    return fname, lai_colorbar_f

def get_wofost_yield_unc(field_id):
    
    f = np.load('data/pixel_yield_sd.npz')
    yld = f[field_id]
    valid_mask = np.isfinite(yld)
    alpha = (valid_mask * 255.).astype(np.uint8)
    
    yld_max, yld_min = np.nanmax(yld[valid_mask]), np.nanmin(yld[valid_mask])
    
    norm_yld = (yld - yld_min) / (yld_max - yld_min)
    cmap = plt.cm.RdPu
    greyscale = cmap(norm_yld, bytes=True)
    
    greyscale[:, :, -1] = alpha
    # greyscale = np.concatenate([greyscale, alpha[:, :, None]])
    
    img = Image.fromarray(greyscale, mode='RGBA')

    scale = 256 / img.height
    new_height = int(scale * img.height)
    new_width = int(scale * img.width)

    img = img.resize((new_width, new_height), resample = Image.NEAREST)
    
    TINT_COLOR = (0, 0, 0)  # Black
    TRANSPARENCY = 0.25  # Degree of transparency, 0-100%
    TRANSPARENCY = 0.75  # Degree of transparency, 0-100%

    
    this_alpha = img.getchannel('A')
    img = img.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
    mask = Image.eval(this_alpha, lambda a: 255 if a <= 5 else 0)

    # Paste the color of index 255 and use alpha as a mask
    img.paste(255, mask)
    img.info['transparency'] = 255

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_axes([0.05, 0.8, 0.5, 0.07])
    
    norm = mpl.colors.Normalize(vmin=yld_min, vmax=yld_max)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    lai_colorbar_f = io.BytesIO()
    plt.savefig(lai_colorbar_f, bbox_inches='tight', format='png', pad_inches=0)
    plt.close()
    
    fname = 'data/S2_thumbs/S2_%s_yield_wofost_unc.png'%(field_id)
    img.save(fname)
    
    return fname, lai_colorbar_f

# from PIL import Image

# def gen_frame(path):
#     im = Image.open(path)
#     alpha = im.getchannel('A')

#     # Convert the image into P mode but only use 255 colors in the palette out of 256
#     im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)

#     # Set all pixel values below 128 to 255 , and the rest to 0
#     mask = Image.eval(alpha, lambda a: 255 if a <=128 else 0)

#     # Paste the color of index 255 and use alpha as a mask
#     im.paste(255, mask)

#     # The transparency index is 255
#     im.info['transparency'] = 255

#     return im


# im1 = gen_frame('frame1.png')
# im2 = gen_frame('frame2.png')        
# im1.save('GIF.gif', save_all=True, append_images=[im2], loop=5, duration=200)