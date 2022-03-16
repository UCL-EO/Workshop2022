from ipywidgets import SelectionRangeSlider, Layout, Accordion, Tab, Text, DatePicker, VBox, Button, Label, FloatSlider, Dropdown
from ipyleaflet import Map, DrawControl, TileLayer, WidgetControl, ImageOverlay, GeoJSON
from ipywidgets import Image as widgetIMG
from ipywidgets import Checkbox
from ipyevents import Event


from pyproj import Proj, Transformer

import json
import os
import ee
import shutil
import requests
import datetime
import numpy as np
from retry import retry
from shapely import geometry
from multiprocessing import Pool
from functools import partial
import pandas as pd

from bqplot import Lines, Figure, LinearScale, DateScale, Axis, Boxplot, DateScale, Scatter, ColorScale, FlexLine
from bqplot.traits import convert_to_date

from ipyleaflet import AwesomeIcon
from ipywidgets import Play, jslink, SelectionSlider, HBox, VBox
import io
import pylab as plt
import matplotlib as mpl
from osgeo import gdal
from glob import glob
import matplotlib.cm as cm
from PIL import Image



import sys
sys.path.insert(0, './python/')
from map_utils import debounce
from wofost_utils import create_ensemble, wofost_parameter_sweep_func, get_era5_gee

ee.Initialize(opt_url = 'https://earthengine-highvolume.googleapis.com')



def get_s2_files(geom, start, end):
    criteria = ee.Filter.And( ee.Filter.geometry(geom), 
                              ee.Filter.date(start, end))

    s2_sur_images = ee.ImageCollection('COPERNICUS/S2_SR')\
                      .filter(criteria).aggregate_array('system:index').getInfo()

    s2_cloud_images = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")\
                        .filter(criteria).aggregate_array('system:index').getInfo()

    s2_ids = sorted(list(set(s2_sur_images) & set(s2_cloud_images)))
    return s2_ids

def get_s2_files(geom, start, end):
    criteria = ee.Filter.And( ee.Filter.geometry(geom), 
                              ee.Filter.date(start, end))

    s2_sur_images = ee.ImageCollection('COPERNICUS/S2_SR')\
                      .filter(criteria).getInfo()
    
    s2_sur_ids = []
    angs = []
    for i in s2_sur_images['features']:
        s2_id = i['properties']['system:index']
        sza = i['properties']['MEAN_SOLAR_ZENITH_ANGLE']
        saa = i['properties']['MEAN_SOLAR_AZIMUTH_ANGLE']
        vza = i['properties']['MEAN_INCIDENCE_ZENITH_ANGLE_B2']
        vaa = i['properties']['MEAN_INCIDENCE_AZIMUTH_ANGLE_B2']

        sza = np.cos(np.deg2rad(sza))
        vza = np.cos(np.deg2rad(vza))
        raa = ((vaa - saa) % 360) / 360

        s2_sur_ids.append(s2_id)
        angs.append([sza, vza, raa])
    s2_angs_dict = dict(zip(s2_sur_ids, angs))
    
    s2_cloud_ids = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")\
                            .filter(criteria).aggregate_array('system:index').getInfo()
    s2_ids = sorted(list(set(s2_sur_ids) & set(s2_cloud_ids)))
    
    return s2_ids, s2_angs_dict


@retry(tries=10, delay=1, backoff=2)
def download_image(inp):
    image_id, bands, filename, geom = inp
    # if not os.path.exists(filename):
    image = ee.Image(image_id)
    download_option = {'name': image_id.split('/')[-1], 
                       'scale': 10, 
                       'bands': bands,
                       'format': "GEO_TIFF",
                       'region': geom,
                       }
    url = image.getDownloadURL(download_option)
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", image_id)
    # else:
    #     print("Using previously downloaded: ", image_id)


def get_s2_over_field(field_name, geom, start, end, dest_dir = 'data/S2_sur_obs/'):
    s2_ids, s2_angs_dict = get_s2_files(geom, start, end)
    
    s2_images = [os.path.join('COPERNICUS/S2_SR', i) for i in s2_ids]
    s2_cloud_images = [os.path.join('COPERNICUS/S2_CLOUD_PROBABILITY', i) for i in s2_ids]

    s2_image_filenames = [os.path.join(dest_dir, 'S2_%s.tif'       % s2_id) for s2_id in s2_ids]
    s2_cloud_filenames = [os.path.join(dest_dir, 'S2_%s_cloud.tif' % s2_id) for s2_id in s2_ids]

    s2_bands = [['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'], ] * len(s2_images)
    s2_bands = [['B4','B8'], ] * len(s2_images)
    s2_bands = [['B2','B3','B4','B5','B6','B7','B8', 'B8A','B11','B12']]* len(s2_images)
    cloud_bands = [['probability'],] * len(s2_cloud_images)

    images_to_downaload = s2_images + s2_cloud_images
    bands_to_download = s2_bands + cloud_bands
    filenames = s2_image_filenames + s2_cloud_filenames
    geoms = [geom, ] * len(filenames)
    inps = np.array([images_to_downaload, bands_to_download, filenames, geoms]).T
    not_need_to_do = np.array([os.path.exists(i) for i in filenames])
    inps = inps[~not_need_to_do]
    inps = inps.tolist()
    
    if len(inps) > 0: 
        pool = Pool(min(10, len(inps)))
        pool.map(download_image, inps)
        pool.close()
        pool.join()

    return filenames, s2_angs_dict



center = [9, 0]
zoom = 15
my_map = Map(center=center, zoom=zoom, max_zoom=17)

draw_control = DrawControl()
draw_control.polygon = {
    "shapeOptions": {
        "fillColor": "#6be5c3",
        "color": "#6be5c3",
        "weight": 1,
        "fillOpacity": 0.2
    },
    "drawError": {
        "color": "#dd253b",
        "message": "Oups!"
    },
    "allowIntersection": False
}
draw_control.circlemarker = {}
draw_control.polyline = {}
# draw_control.circle = {
#     "shapeOptions": {
#         "fillColor": "#efed69",
#         "color": "#efed69",
#         "fillOpacity": 1.0
#     }
# }
# draw_control.rectangle = {
#     "shapeOptions": {
#         "fillColor": "#fca45d",
#         "color": "#fca45d",
#         "fillOpacity": 1.0
#     }
# }

defaultLayout=Layout(width='100%', height='760px')
my_map = Map(center=(9.3771, -0.6062), zoom=zoom, scroll_wheel_zoom=True, max_zoom = 18, layout=defaultLayout)
Google_layer = TileLayer(url = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', name = 'Google satellite')
my_map.add_layer(Google_layer)
my_map.add_control(draw_control)

lai_line_x = np.arange(180, 365)
lai_line_y = lai_line_x * np.nan

x_scale = LinearScale(min = 180, max = 365)
y_scale = LinearScale(min = 0, max = 3)

lai_line  = Lines(x=lai_line_x, y=lai_line_y, scales={"x": x_scale, "y": y_scale}, line_style='dotted', marker='circle', marker_size=4, colors = ['green'])

def add_ee_layer(my_map, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    ee_rater_layer = TileLayer(
        url=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
    )
    my_map.add_layer(ee_rater_layer)
    return ee_rater_layer
my_map.add_ee_layer = add_ee_layer


# 

# dates = [datetime.date(2015, i, 1) for i in range(1, 13)]
# options = [(i.strftime('%b'), i) for i in dates]
# time_slider = SelectionRangeSlider(
#     options=options,
#     index=(0, 11),
#     description='2018',
#     disabled=False
# )

# panel_box = VBox([fig_box, k_slider1, k_slider2, k_slider3], layout = box_layout)

# k_slider = FloatSlider(min=0, max=6, value=2,        # Opacity is valid in [0,1] range
#                orientation='horizontal',       # Vertical slider is what we want
#                readout=True,                # No need to show exact value
#                layout=Layout(width='80%'),
#                description='K: ', 
#                style={'description_width': 'initial'}) 

# panel_box = VBox([fig_box, k_slider], layout = box_layout)




# accordion = Accordion(children=[time_slider, time_slider], titles=('Slider', 'Text'))
# widget_control1 = WidgetControl(widget=accordion, position='bottomleft')
# my_map.add_control(widget_control1)

date_picker1 = DatePicker(
    description='Start Date',
    disabled=False
)
date_picker2 = DatePicker(
    description='End Date',
    disabled=False
)

download_button = Button(
    description='Download',
    disabled=True,
    button_style='primary', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='' # (FontAwesome names without the `fa-` prefix)
)

box_layout = Layout(display='flex',
                flex_flow='column',
                align_items='flex-end',
                width='100%')

info_layout = Layout(display='flex',
                flex_flow='row',
                align_items='flex-start',
                width='100%')

Download_info_label = Button(
    description='1. Draw a field on the map\n2.Pick the date range\n3.Click Donwload',
    disabled=False,
    button_style='primary', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='' # (FontAwesome names without the `fa-` prefix)
)

Download_info_label = VBox([Label('1. Give a neme to the field'), 
                            Label('2. Draw a field on the map'), 
                            Label('3. Pick the date range'), 
                            Label('4. Click Donwload')])

field_name = Text(description='Field name: ')

download_box = VBox([VBox([Download_info_label], layout = info_layout), field_name, date_picker1, date_picker2, download_button], layout = box_layout)

accordion = Accordion(children=[download_box], selected_index=None)
accordion.set_title(0, 'Download S2 images')

widget_control1 = WidgetControl(widget=accordion, position='bottomleft')
my_map.add_control(widget_control1)





def create_ndvi_thumbs(surs, aoi):
    ndvi_cmap = cm.RdYlGn
    dest_folder = os.path.dirname(surs[0])
    ndvi_thumbs = []
    ndvis = []
    for filename in surs:
        data = gdal.Warp('', filename, format = 'MEM', 
                         resampleAlg = gdal.GRIORA_NearestNeighbour, 
                         xRes=10, yRes=10,
                         cropToCutline=False, cutlineDSName=aoi).ReadAsArray() * 1.

        cloud = gdal.Warp('', filename.replace('.tif', '_cloud.tif'), format = 'MEM', 
                          resampleAlg = gdal.GRIORA_NearestNeighbour, 
                          xRes=10, yRes=10,
                          cropToCutline=False, cutlineDSName=aoi).ReadAsArray()

        ndvi = (data[6] - data[2]) / (data[6] + data[2])
        ndvis.append(ndvi)
        valid_mask = (np.isfinite(ndvi)) & (cloud < 40)



        alpha = (valid_mask * 255.).astype(np.uint8)

        greyscale = ndvi_cmap(ndvi / 1., bytes=True)
        greyscale[:, :, -1] = alpha

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

        date = filename.split('/')[-1].split('_')[1][:8]
        
        fname = dest_folder + '/S2_ndvi_%s.png'%(date)
        img.save(fname)
        ndvi_thumbs.append(fname)
    
    ndvi_med = np.nanmedian(ndvis, axis=0)
    valid_mask = np.isfinite(ndvi_med)
    alpha = (valid_mask * 255.).astype(np.uint8)

    greyscale = ndvi_cmap(ndvi_med / 1., bytes=True)
    greyscale[:, :, -1] = alpha

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

    fname = dest_folder + '/S2_ndvi_med.png'
    img.save(fname)
#     ndvi_thumbs.append(fname)

    home = os.getcwd()
    cwd = '/files/' + '/'.join(home.split('/')[3:])
    base_url = my_map.window_url.split('/lab/')[0] + cwd + '/'

    # url = 'data/S2_thumbs/S2_%s_lai_%03d.png'%(field_id, value)
    url = dest_folder + '/S2_ndvi_med.png'
    url = base_url + url

    x_min, y_min, x_max, y_max = bounds

    img_bounds = (y_min, x_max), (y_max, x_min)

    ndvi_med = ImageOverlay(
        url=url,
        bounds = img_bounds,
        name = 'S2 NDVI median'
        )
    my_map.add_layer(ndvi_med)


    return ndvi_thumbs

daily_img = None
def on_change_date_slider(change):

    if (change['name'] == 'value') & (change['type'] == 'change'):
        value = change["new"]
        old = change['old']
        ind = change['owner'].index

        global daily_img, lai_dot

        # value = dates[ind]
        home = os.getcwd()
        cwd = '/files/' + '/'.join(home.split('/')[3:])
        base_url = my_map.window_url.split('/lab/')[0] + cwd + '/'

        # url = 'data/S2_thumbs/S2_%s_lai_%03d.png'%(field_id, value)
        url = ndvi_thumbs[ind]
        url = base_url + url
        
        x_min, y_min, x_max, y_max = bounds

        img_bounds = (y_min, x_max), (y_max, x_min)

        date_str = ndvi_thumbs[ind].split('/')[-1].split('_')[-1][:8]
        date_str = '-'.join([date_str[:4], date_str[4:6], date_str[6:8]])

        if daily_img is None:
            daily_img = ImageOverlay(
            url=url,
            bounds = img_bounds,
            name = 'S2 NDVI: ' + date_str
            )
            my_map.add_layer(daily_img)
            daily_img.url = url
            daily_img.bounds = img_bounds
        else:
            daily_img.url = url
            daily_img.bounds = img_bounds
            daily_img.name = 'S2 NDVI: ' + date_str

def get_colorbar(vmin, vmax, label, colorbar_control = None):
    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_axes([0.05, 0.8, 0.5, 0.07])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm.RdYlGn, norm=norm, orientation='horizontal')
    lai_colorbar_f = io.BytesIO()
    plt.savefig(lai_colorbar_f, bbox_inches='tight', format='png', pad_inches=0)
    plt.close()
    image = lai_colorbar_f.getvalue()
    output = widgetIMG(value=image, format='png',)
    output.layout.object_fit = 'contain'
    # lai_label = Label('$$LAI [m^2/m^2]$$')
    colorbar_label = Label(label)
    colorbar_box = VBox([colorbar_label, output])
    if colorbar_control is None:
        colorbar_control = WidgetControl(widget=colorbar_box, position="bottomright")
    else:
        colorbar_control.widget = colorbar_box
    return colorbar_control


def get_play_slider(dates, play_control = None):
    
    play = Play(
        value=0,
        min=0,
        max=len(dates),
        step=1,
        interval=200,
        description="Press play",
        disabled=False
    )

    date_slider = SelectionSlider(options = dates, description='Date: ', style={'description_width': 'initial'}) 
    date_slider.observe(on_change_date_slider)
    jslink((play, 'value'), (date_slider, 'index'))
    play_label = Label('Click to play movie')
    play_box = HBox([play, date_slider])
    play_box = VBox([play_label, play_box])
    if play_control is None:
        play_control = WidgetControl(widget=play_box, position="bottomright")
    else:
        play_control.widget = play_box
    return play_control


def create_time_series_fig(title, dates, y, y1, min_val, max_val):
    fig_layout = Layout(width='100%', height='100%')
    fig_layout = Layout(width='auto', height='auto', max_height='120px', max_width='360px')

    tick_style = {'font-size': 7}

    s2_dates = [datetime.datetime.strptime(i, '%Y-%m-%d') for i in dates]
    date_x = convert_to_date(s2_dates)
    
    x_scale = DateScale(min = date_x[0], max = date_x[-1])
    y_scale = LinearScale(min = min_val, max = max_val)

    ndvi_line  = Lines(x=date_x, y=y, scales={"x": x_scale, "y": y_scale}, line_style='dotted', marker='circle', marker_size=4, colors = ['green'])
    sndvi_line = Lines(x=date_x, y=y1, scales={"x": x_scale, "y": y_scale})
    
    ndvi_line  = Scatter(x=date_x, y=y, scales={"x": x_scale, "y": y_scale}, default_size=4, colors = ['green'])#, line_style='dotted', marker='circle', marker_size=4, colors = ['green'])
    # sndvi_line = Scatter(x=date_x, y=y1, scales={"x": x_scale, "y": y_scale})
    
    
    tick_values = np.linspace(min_val, max_val, 5)
    ax_y = Axis(label=title,   scale=y_scale, orientation="vertical", side="left", tick_values=tick_values, tick_style=tick_style)
    ax_x = Axis(label="Date", scale=x_scale, num_ticks=5, tick_style=tick_style)


    fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[ndvi_line, sndvi_line], 
                           title=title, 
                           animation_duration=500, 
                           title_style = {'font-size': '8'},
                           fig_margin = dict(top=16, bottom=16, left=16, right=16)
                )
    return fig, ndvi_line, sndvi_line, ax_x, ax_y

def get_pixel_inspector_panel(dates):

    title = 'NDVI'
    y = np.zeros(len(dates))*np.nan
    fig1, ndvi_line, sndvi_line, fig1_ax_x, fig1_ax_y = create_time_series_fig(title, dates, y, y, 0, 1)
    title = 'LAI'
    y = np.zeros(len(dates))*np.nan
    fig2, lai_line, slai_line, fig2_ax_x, fig2_ax_y = create_time_series_fig(title, dates, y, y, 0, 3)
    figs = VBox([fig1, fig2])
    
    return fig1, ndvi_line, sndvi_line, fig1_ax_x, fig1_ax_y, fig2, lai_line, slai_line, fig2_ax_x, fig2_ax_y

filenames = []
bounds = None
ndvi_thumbs = []
dates = None


def download_s2_image():
    global filenames, bounds, ndvi_thumbs, dates, ndvi_line, lai_line, s2_angs_dict
    filed_id = str(field_name.value)
    start_date = date_picker1.value.strftime('%Y-%m-%d')
    end_date = date_picker2.value.strftime('%Y-%m-%d')
    coords = draw_control.last_draw['geometry']['coordinates']
    poly = geometry.Polygon(coords[0])
    bounds = poly.bounds
    bounds_str = '%.05f_%.05f_%.05f_%.05f'%(poly.bounds)
    dest_dir = 'data/S2_sur_obs_%s/'%filed_id # + bounds_str
    # dest_dir = 'data/S2_sur_obs/'
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    with open(dest_dir + '/aoi.geosjon', 'w') as f:
        json.dump(draw_control.last_draw, f)  
        
    geom = ee.Geometry.Polygon(coords)
    filenames, s2_angs_dict = get_s2_over_field(filed_id, geom, start_date, end_date, dest_dir = dest_dir)

play_control = None 
colorbar_control = None 
fig_control = None



def read_wofost_data(lat, lon, year):
    """Reads ensemble from JASMIN"""
    # lat, lon = my_map.center
    lat, lon = (lat // 0.1) * 0.1, (lon // 0.1) * 0.1
    print(lat, lon, year)
    f = create_ensemble(lat, lon, year)
    max_lai = np.nanmax(f.f.LAI, axis=1)
    y = f.f.Yields.astype(float)
    lai = f.f.LAI.astype(float)
    param_names = paras

    param_array = np.array([f[i]
                            for i in param_names]).astype(float)

    pred_yield = max_lai * 1500 - 700
    passer = np.abs(pred_yield - y) < 100.
    sim_yields = y[passer]
    sim_lai = lai[passer, :]
    param_array = param_array[:, passer]
    sim_times = f.f.sim_times

    doys = [int(x.strftime("%j")) for x in sim_times]
    return param_array, sim_times, sim_lai, sim_yields, doys




# Button needs to be centered
assimilate_me_button = Button(
    description='Auto Fit',
    disabled=False,
    button_style='danger', 
    tooltip='Click me',
    icon='',
    layout = Layout(display='flex',
                    flex_flow='horizontal',
                    align_items='center',
                    justify_content="center",
                    width='30%'))

def assimilate_me(b):
    # Needs obs LAI & obs LAI times
    # global pix_lai, doys
    doys = line_axs[-1].x
    pix_lai = line_axs[-1].y
    
#     step = 5
#     bin_doys = np.arange(doys.min(), doys.max(), step)
#     pix_lai = np.interp(bin_doys, doys, pix_lai)
#     doys = bin_doys
    
    bin_lais = []
    bin_doys = []
    step = 10
    for i in np.arange(doys.min(), doys.max()-step, step):
        mm = (doys >= i) * (doys <=i+step)
        bin_lai = np.nanmean(pix_lai[mm])
        if not np.isnan(bin_lai):
            bin_lais.append(bin_lai)
            bin_doys.append(int(i + step / 2))
    pix_lai = np.array(bin_lais)
    doys = np.array(bin_doys)
    print(pix_lai)
    lat, lon = my_map.center
    lat, lon = (lat // 0.1) * 0.1, (lon // 0.1) * 0.1
    year = 2021
    # Read ensemble
    wofost_status_info.description = 'Getting Wofost ensembles...'
    param_array, sim_times, sim_lai, sim_yields, sim_doys = read_wofost_data(lat, lon, year)
    

    t_axis = np.array([datetime.datetime.strptime(f"{year}/{x}", "%Y/%j").date()
              for x in doys])
    
    wofost_status_info.description = 'Fitting to LAI'
    est_yield, est_yield_sd, parameters, _, _ , ensemble_lai_time, lai_fitted_ensembles = ensemble_assimilation(
        param_array, sim_times, sim_lai, sim_yields, pix_lai, t_axis)
    
    for i, para in enumerate(paras):
        wofost_sliders_dict[para].value = np.mean(parameters[i])
    
    ensemble_lai_time = [int(i.strftime('%j')) for i in ensemble_lai_time]
    
    wofost_out_dict['LAI'].marks[3].x = ensemble_lai_time
    wofost_out_dict['LAI'].marks[3].y = lai_fitted_ensembles
    wofost_out_dict['LAI'].marks[3].display_legend = False
    print(est_yield)
assimilate_me_button.on_click(assimilate_me)    

lon = -2.7
lat = 8.20
year = 2021

@debounce(0.2)
def on_change_wofost_slider(change):
    
    global wofost_out_dict
    if (change['type'] == 'change'):
        
        lat, lon = my_map.center
        lat, lon = (lat // 0.1) * 0.1, (lon // 0.1) * 0.1
        print(lat, lon, year)
        wofost_status_info.description = 'Reading ERA5 weather data from local or GEE'
        meteo_file = get_era5_gee(year, lat, lon, dest_folder="data/ERA5_weather/")
        print(meteo_file)
        
        ens_parameters = {}
        paras_to_overwrite = [i for i in paras if 'AMAX_' not in i]
        for para in paras_to_overwrite:    
            ens_parameters[para] = wofost_sliders_dict[para].value
        # ens_parameters['AMAXTB'] = [0, 55.0, 1.5, wofost_sliders_dict['AMAXTB_150'].value]
        scalar =  wofost_sliders_dict["AMAX_SCALAR"].value
        ens_parameters["AMAXTB"]  = [0.0, 70.0*scalar,
                                    1.25, 70.0*scalar,
                                    1.50, 63.0*scalar,
                                    1.75, 49.0*scalar,
                                    2.0, 0.0,
                                    ]
        
        #ens_parameters['AMAXTB'] = [0, wofost_sliders_dict['AMAXTB_000'].value,
        #                           1.25, wofost_sliders_dict['AMAXTB_125'].value,
        #                           1.50, wofost_sliders_dict['AMAXTB_150'].value,
        #                           2.0, 2
        #                           ]
        wofost_status_info.description = 'Running model...'
        df = wofost_parameter_sweep_func(year, ens_parameters = ens_parameters.copy(), meteo=meteo_file)
        print(df)
        
        dates = df.index
        doys = [int(i.strftime('%j')) for i in dates]
        
        # wofost_lai_fig.x = doys
        # wofost_lai_fig.y = np.array(df.LAI)
        wofost_out_paras = ['DVS', 'LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'TRA', 'RD', 'SM', 'WWLOW']
        for wofost_out_para in wofost_out_paras:
            if wofost_out_para != 'TWSO':
                wofost_out_dict[wofost_out_para].marks[0].x = doys
                wofost_out_dict[wofost_out_para].marks[0].y = np.array(df.loc[:, wofost_out_para])
            else:
                twso = np.array(df.loc[:, wofost_out_para])
                real_twso = 1.4553 * np.nanmax(twso) - 1341.81
                real_twso = np.max([0, real_twso])
                twso = real_twso / np.nanmax(twso) * twso
                wofost_out_dict[wofost_out_para].marks[1].x = doys
                wofost_out_dict[wofost_out_para].marks[1].y = np.array(twso)
        
        doys = [int(datetime.datetime.fromtimestamp(i.item() / 10**9).strftime('%j')) for i in lai_line.x]    
        wofost_out_dict['LAI'].marks[2].x = doys
        wofost_out_dict['LAI'].marks[2].y = lai_line.y
        colored_dvs_line.x = doys
        colored_dvs_line.y = wofost_out_dict['DVS'].marks[0].y
        colored_dvs_line.color = wofost_out_dict['DVS'].marks[0].y
        wofost_status_info.description = 'Done'
        # wofost_out_dict['TWSO'].marks[0].x = doys
        # wofost_out_dict['TWSO'].marks[0].y = np.array(df.TWSO)
        
        # lai_fig.marks[1].scales = {'x': LinearScale(max=365.0, min=180.0), 'y': LinearScale(max=3.0, min=0.0)} 
        # lai_fig.axes[1].scale = LinearScale(max=3.0, min=0.0)
        # lai_fig.axes[0].scale = LinearScale(max=365.0, min=180.0)

prior_df = pd.read_csv('data/par_prior_maize_tropical-C.csv')
#paras = ['TDWI', 'SDOY', 'SPAN', 'CVO', 'AMAXTB_000', 'AMAXTB_125', 'AMAXTB_150']
paras = ['TDWI', 'SDOY', 'SPAN',  'AMAX_SCALAR']
all_paras, para_mins, para_maxs = np.array(prior_df.loc[:, ['#PARAM_CODE', 'Min', 'Max']]).T
para_inds = [all_paras.tolist().index(i) for i in paras]
wofost_sliders = []

para_meaning = {'TDWI': 'Initial total crop dry weight [kg ha-1]',
                'SDOY': 'Sowing day of year',
                'SPAN': 'Life span of leaves growing at 35 Celsius [d]',
                #'CVO' : 'Efficiency of conversion into storage org. [kg kg-1]',
                #'AMAXTB_000': 'Max. leaf CO2 assim. rate at development stage of 0',
                #'AMAXTB_125': 'Max. leaf CO2 assim. rate at development stage of 1.25',
                #'AMAXTB_150': 'Max. leaf CO2 assim. rate at development stage of 1.5',
                "AMAX_SCALAR": "Scalar on Max. lead CO2 assim. rate"
               }


for i in range(len(paras)):
    para_ind = para_inds[i]
    para_name = all_paras[para_ind]
    para_min = para_mins[para_ind]
    para_max = para_maxs[para_ind]
    step = (para_max - para_min) / 50
    initial = (para_min + para_max) / 2
    wofost_slider = FloatSlider(min=para_min, max=para_max, value=initial,       # Opacity is valid in [0,1] range
                   step = step,
                   orientation='horizontal',       # Vertical slider is what we want
                   readout=True,                # No need to show exact value
                   layout=Layout(width='80%'),
                   description='%s: '%para_name, 
                   description_tooltip= para_meaning[para_name],
                   style={'description_width': 'initial'}) 
    wofost_slider.observe(on_change_wofost_slider)
    wofost_sliders.append(wofost_slider)

wofost_sliders_dict = dict(zip(paras, wofost_sliders))


wofost_fig_vlines = {}
def get_para_plot(para_name, x, y, xmin = 180, xmax = 330):
    global wofost_fig_vlines
    x = np.array(x)
    y = np.array(y)
    
    para_min_maxs = {'DVS': [0, 2],
                     'LAI': [0, 3],
                     'TAGP': [0, 15000],
                     'TWSO': [0, 5500],
                     'TWLV': [0, 2000],
                     'TWST': [0, 10000],
                     'TWRT': [0, 2000],
                     'TRA':  [0, 0.5],
                     'RD':   [0, 100],
                     'SM':   [0, 0.8],
                     'WWLOW': [0, 100]
                    }

    ymin, ymax = para_min_maxs[para_name]
    
    mm = (x >= xmin) & (x <= xmax)
    x_scale = LinearScale(min = xmin, max = xmax)
    # ymin = np.maximum(np.nanpercentile(y[mm], 2.5) * 0.9, 0)
    # ymax = np.nanpercentile(y[mm], 97.5) * 1.1
    y_scale = LinearScale(min = ymin  , max = ymax)
    
    line = Lines(x=x, y=y, scales={"x": x_scale, "y": y_scale}, labels = 'Wofost %s'%para_name, display_legend=True)
    tick_style = {'font-size': 8}
    tick_values = np.linspace(ymin, ymax, 4)
    tick_values
    
    vline = Lines(x=[xmin, xmin], y=[0, ymax], scales={"x": x_scale, "y": y_scale},
                       line_style='solid', colors=['gray'], stroke_width=1)
    wofost_fig_vlines[wofost_out_para] = vline
    
    ax_x = Axis(label="DOY", scale=x_scale,  num_ticks=5, tick_style=tick_style)
    ax_y = Axis(label=para_name, scale=y_scale, orientation="vertical", side="left", tick_values=tick_values, tick_style=tick_style)
    
    fig_layout = Layout(width='400px', height='160px', max_height='160px', max_width='400px')
    
    para_fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line, vline], 
                       title=para_name, 
                       animation_duration=500, 
                       title_style = {'font-size': '8'},
                       legend_text={'font-size': 7},
                       legend_location = 'top-left',
                       legend_style={'width': '40%', 'height': '30%', 'stroke-width':0},
                       fig_margin = dict(top=16, bottom=16, left=46, right=46))

    return para_fig


def update_wofost_fig_val(wofost_out_para, x, y, xmin = 180, xmax = 330):
    x = np.array(x)
    y = np.array(y)
    mm = (x >= xmin) & (x <= xmax)
    y = y[mm]
    x = x[mm]
    wofost_out_dict[wofost_out_para].marks[0].x = x
    wofost_out_dict[wofost_out_para].marks[0].y = y
    
wofost_out_paras = ['DVS', 'LAI', 'TAGP', 'TWSO', 'TWLV', 'TWST', 'TWRT', 'TRA', 'RD', 'SM', 'WWLOW']
para_figs = []

x = np.arange(180, 330)
y = np.zeros_like(x)
for wofost_out_para in wofost_out_paras:
    para_fig = get_para_plot(wofost_out_para, x, y)
    para_figs.append(para_fig)
wofost_out_dict = dict(zip(wofost_out_paras, para_figs))


# wofost_fig_vlines = {}
# for wofost_out_para in wofost_out_paras:
#     vline = Lines(x=[180, 180], y=[0, wofost_out_dict[wofost_out_para].marks[0].scales['x'].max], scales=wofost_out_dict[wofost_out_para].marks[0], 
#                        line_style='solid', colors=['gray'], stroke_width=1)
#     wofost_fig_vlines[wofost_out_para] = vline
    
# obs_lai_line = Lines(x=line_axs[-1].x, y=line_axs[-1].y, scales=wofost_out_dict['LAI'].marks[0].scales, colors = ['red'])
obs_lai_line  = Scatter(x=lai_line.x, y=lai_line.y, scales=wofost_out_dict['LAI'].marks[0].scales, 
                        default_size=4, colors = ['green'], display_legend=True, labels=['Empirical LAI'])    
ens_lai_line = Lines(x=lai_line.x, y=lai_line.y, scales=wofost_out_dict['LAI'].marks[0].scales, 
                     colors = ['#cccccc'], display_legend=False, labels=['Ensemble LAI'])

ens_lai_line_temp = Lines(x=lai_line.x, y=lai_line.y*np.nan, scales=wofost_out_dict['LAI'].marks[0].scales, 
                     colors = ['#cccccc'], display_legend=True, labels=['Ensemble LAI'])


wofost_out_dict['LAI'].marks[0].display_legend=True

lai_dot_wofost = Lines(x=[180,], y=[0,], scales=wofost_out_dict['LAI'].marks[0].scales,line_style='dotted', marker='circle', marker_size=45, colors = ['red'])
lai_vline = Lines(x=[180, 180], y=[0, 3], scales=wofost_out_dict['LAI'].marks[0].scales, 
                   line_style='solid', colors=['gray'], stroke_width=1)


twso_vline = Lines(x=[0,], y=[0,], scales=wofost_out_dict['TWSO'].marks[0].scales, 
                   line_style='dashed', colors=['gray'], fill='between')

twso_hline = Lines(x=[0,], y=[0,], scales=wofost_out_dict['TWSO'].marks[0].scales, 
                   line_style='solid', colors=['#ffa500'], fill='between', display_legend=True, labels = ['Avg. Field Yield'])

twso_shade = Lines(x=[0,], y=[0,], scales=wofost_out_dict['TWSO'].marks[0].scales, 
                   line_style='solid', colors=['#cccccc'], fill='between', opacities=[1, 1], display_legend=False, labels = ['1σ'])

twso_shade_temp = Lines(x=[0,], y=[0,], scales=wofost_out_dict['TWSO'].marks[0].scales, 
                   line_style='solid', colors=['#cccccc'], fill='between', opacities=[1, 1], display_legend=True, labels = ['1σ'])

dvs_line = wofost_out_dict['DVS'].marks[0]

colors = ['#e5f5e0', '#a1d99b', '#31a354', '#31a354', '#31a354','#a1d99b', '#ffeda0', '#feb24c']
col_line = ColorScale(colors=colors)

scales = dvs_line.scales
scales['color'] = col_line

dvs_yticks = wofost_out_dict['DVS'].axes[1].tick_values
dvs_yax = Axis(
    label="DVS",
    scale=scales['y'],
    orientation="vertical",
    side="right",
    tick_format="0.1f",
    grid_lines = 'none',
    tick_values = dvs_yticks,
    tick_style = {'font-size': 8}
)

colored_dvs_line = FlexLine(x=dvs_line.x, y=dvs_line.y, color=dvs_line.y,
                             scales=scales, labels = ['Wofost DVS'], display_legend=False,)

wofost_out_dict['TWSO'].marks = [twso_shade] +  wofost_out_dict['TWSO'].marks[:2] + [twso_hline, colored_dvs_line,twso_shade_temp]
wofost_out_dict['TWSO'].legend_location = 'bottom-left'
wofost_out_dict['TWSO'].marks[1].display_legend=True

for wofost_out_para in wofost_out_paras:
    wofost_out_dict[wofost_out_para].axes = wofost_out_dict[wofost_out_para].axes + [dvs_yax,]

for wofost_out_para in ['TAGP', 'TWLV', 'TWST', 'TWRT', 'TRA', 'RD', 'SM', 'WWLOW']:
    wofost_out_dict[wofost_out_para].marks = wofost_out_dict[wofost_out_para].marks + [colored_dvs_line,]

    
wofost_out_dict['LAI'].marks = wofost_out_dict['LAI'].marks[:2] + [obs_lai_line, ens_lai_line, lai_dot_wofost, colored_dvs_line, ens_lai_line_temp]
    
    
wofost_output_dropdown1 = Dropdown(
    options=wofost_out_paras,
    value=wofost_out_paras[1],
    layout={'width': 'max-content'}
    
    # description="Field ID:",
)
wofost_output_dropdown2 = Dropdown(
    options=wofost_out_paras,
    value=wofost_out_paras[3],
    layout={'width': 'max-content'}
    # description="Field ID:",
)



wofost_output_dropdowns = HBox([wofost_output_dropdown1, wofost_output_dropdown2], layout = Layout(display='flex',
                                  flex_flow='horizontal',
                                  align_items='flex-start',
                                  width='100%'))


left_output = VBox([wofost_output_dropdown1, wofost_out_dict[wofost_output_dropdown1.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))

right_output = VBox([wofost_output_dropdown2, wofost_out_dict[wofost_output_dropdown2.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))



def on_change_dropdown1(change):
    global wofost_widgets
    
    left_output = VBox([wofost_output_dropdown1, wofost_out_dict[wofost_output_dropdown1.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))
    right_output = VBox([wofost_output_dropdown2, wofost_out_dict[wofost_output_dropdown2.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))
    
    wofost_widgets[-2] = VBox([left_output, right_output])
    wofost_box = VBox(wofost_widgets, 
                  layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='400px'))
    
    wofost_control.widget = wofost_box
    # tab.children = [panel_box, wofost_box]

def on_change_dropdown2(change):
    global wofost_widgets
    
    left_output = VBox([wofost_output_dropdown1, wofost_out_dict[wofost_output_dropdown1.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))
    right_output = VBox([wofost_output_dropdown2, wofost_out_dict[wofost_output_dropdown2.value]], layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='100%'))
    wofost_widgets[-2] = VBox([left_output, right_output])

    wofost_box = VBox(wofost_widgets, 
                  layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='400px'))
    
    wofost_control.widget = wofost_box
    # tab.children = [panel_box, wofost_box]

wofost_status_info = Button(description = '', button_style='info', layout=Layout(width='100%'), disabled=True)
wofost_status_info.style.button_color = '#999999'
wofost_out_panel = VBox([left_output, right_output])
wofost_widgets = wofost_sliders + [assimilate_me_button, wofost_out_panel, wofost_status_info,]
wofost_output_dropdown1.observe(on_change_dropdown1, 'value')
wofost_output_dropdown2.observe(on_change_dropdown2, 'value')


wofost_box = VBox(wofost_widgets, 
                  layout = Layout(display='flex',
                                  flex_flow='column',
                                  align_items='center',
                                  width='400px'))

wofost_control = WidgetControl(widget=wofost_box, position="topright")

my_map.add_control(wofost_control)

def create_thumbs_and_pixel_inspector(filenames, play_control = None, colorbar_control = None, fig_control = None):
    global ndvi_thumbs 
    global lai_line, ndvi_line
    surs = [i for i in filenames if '_cloud' not in i]
    aoi = os.path.dirname(surs[0]) + '/aoi.geosjon'
    ndvi_thumbs = create_ndvi_thumbs(surs, aoi)
    
    dates = [datetime.datetime.strptime(i.split('_')[-1][:8], '%Y%m%d').strftime('%Y-%m-%d') for i in ndvi_thumbs]
    
    play_control = get_play_slider(dates, play_control)
    colorbar_control = get_colorbar(0, 1, 'NDVI', colorbar_control)
    
    # full_dates = [datetime.datetime.strptime(i.split('/')[-1].split('_')[1][:15], '%Y%m%dT%H%M%S') for i in filenames]
    
    # fig1, ndvi_line, sndvi_line, fig1_ax_x, fig1_ax_y, fig2, lai_line, slai_line, fig2_ax_x, fig2_ax_y
    ret = get_pixel_inspector_panel(dates)
    fig1, ndvi_line, sndvi_line, fig1_ax_x, fig1_ax_y, fig2, lai_line, slai_line, fig2_ax_x, fig2_ax_y = ret
    fig_panel = VBox([fig1, fig2])
    if fig_control is None:
        fig_control = WidgetControl(widget=wofost_box, position="topright")
    else:
        fig_control.widget = fig_panel
    return play_control, colorbar_control, fig_control

# def observe_download_ready(change):
    
#     if (change['type'] == 'change') & (change['new'] == 'Finished!'):


@debounce(0.2)
def download(button):
    global play_control, colorbar_control, fig_control
    if draw_control.last_draw['geometry'] is None:
        print('Draw polygon for your field!')
    elif field_name.value == '':
        print('Give a name to your field!')
    elif date_picker1.value is None:
        print('Pick a start date!')
    elif date_picker2.value is None:
        print('Pick a end date!')
    else:
        print('start downloading')
        # download_button.disabled = False
        download_button.disabled = True
        download_button.button_style = 'info'
        download_button.description = 'Downloading...'
        download_s2_image()
        download_button.description = 'Finished!'
        download_button.button_style = 'success'
        
        if (play_control is None) | (colorbar_control is None) | (fig_control is None):
            play_control, colorbar_control, fig_control = create_thumbs_and_pixel_inspector(filenames)
            my_map.add_control(play_control)
            my_map.add_control(colorbar_control)
            
        else:
            create_thumbs_and_pixel_inspector(filenames, play_control, colorbar_control, fig_control)
            
        
def refresh_download_button(change):
    
    if draw_control.last_draw['geometry'] is None:
        print('Draw polygon for your field!')
    elif field_name.value == '':
        print('Give a name to your field!')
    elif date_picker1.value is None:
        print('Pick a start date!')
    elif date_picker2.value is None:
        print('Pick a end date!')
    elif change['type'] == 'change':
        download_button.disabled = False
        download_button.button_style = 'primary'
        download_button.description = 'Download'


import numpy as np
f = np.load('data/nnLai.npz', allow_pickle=True)
arrModel = f.f.model

def affine_forward(x, w, b):
    """
    Forward pass of an affine layer
    :param x: input of dimension (D, )
    :param w: weights matrix of dimension (D, M)
    :param b: biais vector of dimension (M, )
    :return output of dimension (M, ), and cache needed for backprop
    """
    out = np.dot(x, w) + b
    cache = (x, w)
    return out, cache

def relu_forward(x):
    """ Forward ReLU
    """
    out = np.maximum(np.zeros(x.shape).astype(np.float32), x)
    cache = x
    return out, cache

def predict(inputs, arrModel):
    nLayers = int(len(arrModel) / 2)
    r = inputs
    for i in range(nLayers):
        w, b = arrModel[i*2], arrModel[i*2 + 1]
        a, _ = affine_forward(r, w, b) 
        r, _ = relu_forward(a)
    return r

from ipyleaflet import Marker
marker = None
def mouse_click(**kwargs):
    global marker
    if kwargs.get('type') == 'click':
        location=kwargs.get('coordinates')
        point = geometry.Point(location[1], location[0])
        print(point)
        
        if marker is None:
            marker = Marker(location=location, rotation_angle=0, draggable=False, name = 'Pixel location')
            my_map.add_layer(marker) 
        else:
            marker.location = location
        if len(filenames) > 0: 
            lat, lon = marker.location
            g = gdal.Open(filenames[0])
            geo_trans = g.GetGeoTransform()
            projectionRef = g.GetProjectionRef()

            pj1 = Proj(projectionRef)
            transformer = Transformer.from_crs('EPSG:4326', pj1.crs)
            x, y = transformer.transform(lat, lon)

            pix_x = int((x - geo_trans[0]) / geo_trans[1])
            pix_y = int((y - geo_trans[3]) / geo_trans[5])
            surs = [i for i in filenames if '_cloud' not in i]
            ndvis = []
            inputs = []
            selis = []
            for filename in surs:
                sza, vza, raa = s2_angs_dict[filename.split('/')[-1][3:-4]]
                print(filename)
                # try:
                g = gdal.Open(filename)
                pix_val = g.ReadAsArray(pix_x, pix_y, 1, 1).squeeze()
                g = gdal.Open(filename.replace('.tif', '_cloud.tif'))
                cloud_val = g.ReadAsArray(pix_x, pix_y, 1, 1).squeeze()
                if cloud_val < 60:
                    ndvi = (pix_val[6] - pix_val[2]) / (pix_val[6] + pix_val[2])
                    use_bands = [0, 1, 2, 3, 4, 5, 7, 8, 9]
                    seli = (pix_val[7] - pix_val[3]) / (pix_val[7] + pix_val[3])
                    selis.append(seli)
                    # inputs = np.array([pix_val/10000, [np.cos(np.deg2rad(30)), np.cos(np.deg2rad(5)), 120/360.]])
                    inp = np.concatenate([pix_val/10000, [sza, vza, raa]])
                    inputs.append(inp)
                    
                else:
                    inputs.append(np.zeros(13) * np.nan)
                    ndvi = np.nan
                    selis.append(np.nan)
                # except:
                #     ndvi = np.nan
                ndvis.append(ndvi)
                # print(ndvi)
            ndvis = np.array(ndvis)
            selis = np.array(selis)
            lai = 5.405 * selis - 0.114
            # lai = np.log(predict(np.array(inputs), arrModel).ravel()) * -2
            # lai = 3.89 * ndvis - 0.11
            ndvi_line.y = ndvis
            lai_line.y = lai
            doys = [int(datetime.datetime.fromtimestamp(i.item() / 10**9).strftime('%j')) for i in lai_line.x]
            wofost_out_dict['LAI'].marks[2].x = doys
            wofost_out_dict['LAI'].marks[2].y = lai_line.y
            
my_map.on_interaction(mouse_click)

draw_control.observe(refresh_download_button)
field_name.observe(refresh_download_button)
date_picker1.observe(refresh_download_button)
date_picker2.observe(refresh_download_button)

download_button.on_click(download)
# download_button.observe(observe_download_ready)

maize_imgCol = ee.ImageCollection('users/xianda19/classification_result/2021/Ghana/maize_20210501_20211011_100percentSamples')
maize_img = maize_imgCol.mosaic().selfMask()
classification_layer = my_map.add_ee_layer(my_map, ee_image_object = maize_img, vis_params = {'palette': ['green']}, name = 'classification')

# slider = FloatSlider(min=0, max=1, value=1,        # Opacity is valid in [0,1] range
#                orientation='horizontal',       # Vertical slider is what we want
#                readout=False,                # No need to show exact value
#                layout=Layout(height='2em', width='200px')) # Fine tune display layout: make it thinner

# transparency_label = Label('Transparency:')
# transparency_box = HBox([transparency_label, slider])
# my_map.add_control(WidgetControl(widget=transparency_box, position="bottomright"))
# jslink((slider, 'value'), (classification_layer, 'opacity') )

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
    name = 'Field boundaries',
    style_callback=random_color
)


with open('./data/Biophysical_Data_Collection_Points_V1.geojson', 'r') as f:
    data2 = json.load(f)

points = GeoJSON(
    data=data2,
    point_style={'radius': 5, 'color': 'blue', 'fillOpacity': 0.5, 'fillColor': 'blue', 'weight': 0.1},
    # style={
    #     'opacity': 1, 'dashArray': '0', 'fillOpacity': 0.2, 'weight': 0.01
    # },
    hover_style={
        'color': 'white', 'dashArray': '0', 'fillOpacity': 1
    },
    name = 'Field measurement points',
    # style_callback=random_color
)


my_map.add_layer(fields)
my_map.add_layer(points)

def get_layer_control(my_map):
    check_boxs = []
    labels = []
    sliders = []
    for layer in my_map.layers[1:]:
        check_box = Checkbox(value=True, disabled=False,indent=False, layout=Layout(width='20px'))
        check_boxs.append(check_box)
        label = Label(value = layer.name)
        labels.append(label)
        
        if not isinstance(layer, GeoJSON):
            slider = FloatSlider(min=0, max=1, value=layer.opacity, layout=Layout(width='180px'))
            sliders.append(slider)

            jslink((slider, 'value'), (layer, 'opacity'))
            jslink((check_box, 'value'), (layer, 'visible'))
            slider.style.handle_color = '#3399ff'
        else:
            slider = FloatSlider(min=0, max=1, value=1, layout=Layout(width='180px'))
            sliders.append(slider)
        
            slider.disabled=True
            slider.style.handle_color = '#c0c0c0'
            
            # slider.observe(change_geojson_opacity)
    layer_control_widget = HBox([VBox(check_boxs), VBox(labels), VBox(sliders)])
    return layer_control_widget

layers_button = Button(description='Layers',disabled=False,tooltip='Layers',  layout = Layout(width='120px'))

layer_control_title_widget = VBox([layers_button], layout = Layout(align_items='center', justify_content="center",))
layer_control_title_widget_event = Event(source=layer_control_title_widget, watched_events=['mouseenter', 'mouseleave'])


@debounce(0.2)
def layer_control_title_widget_event_handler(event):
    if event['event'] == 'mouseenter':
        layer_control_widget = get_layer_control(my_map)
        layers_button.button_style = 'danger'
        layers_button.style.button_color = '#3399ff'
        layers_button.layout = Layout(width='100%')
        layer_control_title_widget.children =  [layers_button, layer_control_widget]
        
    if event['event'] == 'mouseleave':
        layers_button.style.button_color = '#EEEEEE'
        layers_button.button_style = ''
        layers_button.layout = Layout(width='120px')
        layer_control_title_widget.children =  [layers_button]

layer_control_title_widget_event.on_dom_event(layer_control_title_widget_event_handler)
my_map.add_control(WidgetControl(widget=layer_control_title_widget, position="bottomleft"))