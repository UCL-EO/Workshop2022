import os
import sys
import numpy as np
import requests
import json
import datetime
import pandas as pd
from pygeotile.tile import Tile
from shapely import geometry
from bqplot import Lines, Figure, LinearScale, DateScale, Axis, Boxplot
from ipywidgets import Dropdown, FloatSlider, HBox, VBox, Layout, Label, jslink, Layout, SelectionSlider, Play
from ipyleaflet import Map, WidgetControl, LayersControl, ImageOverlay, GeoJSON, Marker, Icon
from ipywidgets import Image as widgetIMG


sys.path.insert(0, './python/')
from map_utils import get_lai_gif, get_pixel, get_field_bounds, da_pix, get_lai_color_bar

from ipywidgets import Image as ImageWidget

df = pd.read_csv('data/Ghana_ground_data_v2.csv')

yield_df = pd.read_csv('data/Yield_Maize_Biomass_V2.csv').iloc[:, :3]
codes = np.unique([i[:-2] for i in yield_df.FID])
yields = np.array(yield_df.iloc[:, 1]).reshape(-1, 3)
field_yields = dict(zip(codes, yields.tolist()))

# import IPython
# url = 'https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Ghana_workshop2022/imgs/maize.png'
# maize_img_data = IPython.display.Image(url, width = 300)
# maize_img = ImageWidget(
#   value=maize_img_data.data,
#   format='Png', 
#   width='20',
#   height='20',
# )

zoom = 17


defaultLayout=Layout(width='100%', height='640px')
my_map = Map(center=(9.3771, -0.6062), zoom=zoom, scroll_wheel_zoom=True, max_zoom = 19, layout=defaultLayout)



with open('./data/Biophysical_Data_Collection_Polygons_V1.geojson', 'r') as f:
    data = json.load(f)
field_ids = [feat['properties']['Field_ID'] for feat in data['features']]

dropdown = Dropdown(
    options=codes,
    value=codes[0],
    description="Field ID:",
)
field_id = field_ids[0]

field_bounds, doys = get_field_bounds(field_id)
# url, field_bounds, doys, yield_colorbar_f = get_lai_gif(field_id)

# image = yield_colorbar_f.getvalue()
# output = widgetIMG(value=image, format='png',)
# colorbar = WidgetControl(widget=output, position='bottomleft', transparent_bg=False)
# my_map.add_control(colorbar)

poly = geometry.Polygon(data['features'][0]['geometry']['coordinates'][0])
my_map.center = (poly.centroid.y, poly.centroid.x)


x_scale = LinearScale(min = 200, max = 365)
y_scale = LinearScale(min = 0, max = 3)
x = np.arange(200, 365)
y = np.zeros_like(x)

lines = Lines(x=x, y=y, scales={"x": x_scale, "y": y_scale})

tick_style = {'font-size': 8}
ax_x = Axis(label="DOY", scale=x_scale, num_ticks=5, tick_style=tick_style)
ax_y = Axis(label='LAI', scale=y_scale, orientation="vertical", side="left", num_ticks=4, tick_style=tick_style)

figure = Figure(
    axes=[ax_x, ax_y],
    title=field_id,
    marks=[lines],
    animation_duration=500,
    layout={"max_height": "250px", "max_width": "400px"},
)

# widget_control1 = WidgetControl(widget=figure, position="bottomright")
# my_map.add_control(widget_control1)


line = Lines(x=x, y=y, scales={"x": x_scale, "y": y_scale})

fig_layout = Layout(width='auto', height='auto', max_height='120px', max_width='180px')
#fig_layout = Layout(width='1%', height='20%')

figy=[]
for i in range(3):
    figx=[]
    for j in range(2):
        fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line], 
                           title=field_id, 
                           animation_duration=500, 
                           title_style = {'font-size': '8'},
                           fig_margin = dict(top=16, bottom=16, left=16, right=16))
        fig.title = field_id
        figx.append(fig)
    figy.append(HBox(figx))
widget_control2 = WidgetControl(widget=VBox(figy, align_content = 'stretch'), position='topright')
# my_map.add_control(widget_control2)



fig_layout = Layout(width='auto', height='auto', max_height='120px', max_width='180px')
#fig_layout = Layout(width='1%', height='20%')

# doys = np.arange(200, 365)
sels = np.zeros((6, len(doys), 10))

tick_style = {'font-size': 8}
names = ['B1', 'B2', 'B3', 'B4', 'Cab', 'Lai']
line_axs = []
for ii in range(6):
    y_scale = LinearScale(min = sels[ii].T.min(), max = sels[ii].T.max())
    ax_x = Axis(label="DOY", scale=x_scale, num_ticks=5, tick_style=tick_style)
    ax_y = Axis(label=names[ii], scale=y_scale, orientation="vertical", side="left", num_ticks=4, tick_style=tick_style)
    line = Lines(x=doys, y=sels[ii].T, scales={"x": x_scale, "y": y_scale})
    line.colors = ['#81d8d0']
    line.stroke_width = 0.1
    line_axs.append([line, ax_x, ax_y])

ref_lines = []

for ii in range(4):
    line, ax_x, ax_y = line_axs[ii]
    ref_line = Lines(x=doys, y=np.ones_like(doys) * np.nan, scales = line.scales, line_style='dotted', marker='circle', marker_size=4, colors = ['#c0c0c0'])
    ref_lines.append(ref_line)

good_ref_lines = []
line_colors = ['#3399ff', '#008000', '#ff6666', '#990000']
for ii in range(4):
    line, ax_x, ax_y = line_axs[ii]
    good_ref_line = Lines(x=doys, y=np.ones_like(doys) * np.nan, scales = line.scales, line_style='dotted', marker='circle', marker_size=4, colors = [line_colors[ii]])
    good_ref_lines.append(good_ref_line)

figy=[]
for i in range(3):
    figx=[]
    for j in range(2):
        if i*2+j < 4:
            ref_line = ref_lines[i*2+j]
            good_ref_line = good_ref_lines[i*2+j]
            line, ax_x, ax_y = line_axs[i*2+j]
            fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line, ref_line, good_ref_line], 
                               title=field_id, 
                               animation_duration=500, 
                               title_style = {'font-size': '8'},
                               fig_margin = dict(top=16, bottom=16, left=16, right=16))

        else:
            line, ax_x, ax_y = line_axs[i*2+j]
            var_line = Lines(x=doys, y=np.zeros_like(doys), scales = line.scales)
            line_axs.append(var_line)
            fig = Figure(layout=fig_layout, axes=[ax_x, ax_y], marks=[line, var_line], 
                               title=field_id, 
                               animation_duration=500, 
                               title_style = {'font-size': '8'},
                               fig_margin = dict(top=16, bottom=16, left=16, right=16))

        fig.title = names[i*2+j]
        figx.append(fig)
    figy.append(HBox(figx))
# display(VBox(figy, align_content = 'stretch'))

fig_box = VBox(figy, align_content = 'stretch')
# widget_control1 = WidgetControl(widget=VBox(figy, align_content = 'stretch'), position='topright')


cab_fig = figy[2].children[0]
lai_fig = figy[2].children[1]

# field_cab_boxes = Boxplot(x=field_doys[:-1], y=field_cabs[:-1], 
#                           scales=cab_fig.marks[0].scales, box_fill_color='gray')
# field_cab_boxes.auto_detect_outliers=False
# # field_cab_line = Lines(x=field_doys[:-1], y=field_cabs[:-1].T, marker='cross', 
# #                       scales=cab_fig.marks[0].scales, colors=['orange'])
# field_cab_boxes.stroke = 'red'
# field_cab_boxes.box_fill_color = 'blue'
# field_cab_boxes.opacities = [0.5]
# field_cab_boxes.box_width=5
# cab_fig.marks = cab_fig.marks[:2] + [field_cab_boxes,]


field_lai_boxes = Boxplot(x=doys, y=doys[None]*np.nan, 
                          scales=lai_fig.marks[1].scales, box_fill_color='gray')
field_lai_boxes.auto_detect_outliers=False
# field_cab_line = Lines(x=field_doys[:-1], y=field_cabs[:-1].T, marker='cross', 
#                       scales=cab_fig.marks[0].scales, colors=['orange'])
field_lai_boxes.stroke = 'red'
field_lai_boxes.box_fill_color = 'blue'
field_lai_boxes.opacities = [0.4]
field_lai_boxes.box_width=5


lai_dot = Lines(x=doys[:1], y=[0,], scales=lai_fig.marks[1].scales,line_style='dotted', marker='circle', marker_size=45, colors = ['red'])

lai_fig.marks = lai_fig.marks[:2] + [field_lai_boxes, lai_dot]



box_layout = Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%')

k_slider = FloatSlider(min=0, max=6, value=2,        # Opacity is valid in [0,1] range
               orientation='horizontal',       # Vertical slider is what we want
               readout=True,                # No need to show exact value
               layout=Layout(width='80%'),
               description='K: ', 
               style={'description_width': 'initial'}) 

panel_box = VBox([fig_box, k_slider], layout = box_layout)

widget_control1 = WidgetControl(widget=panel_box, position='topright')
my_map.add_control(widget_control1)



slider = FloatSlider(min=0, max=1, value=1,        # Opacity is valid in [0,1] range
               orientation='vertical',       # Vertical slider is what we want
               readout=False,                # No need to show exact value
               layout=Layout(width='2em')) # Fine tune display layout: make it thinner
my_map.add_control(WidgetControl(widget=slider))

tile = Tile.for_latitude_longitude(*my_map.center, zoom)
x, y = tile.tms

for i in range(x-3, x + 4):
    for j in range(y - 3, y + 4):
        tile = Tile.from_tms(i, j, zoom)
        url = "http://ecn.t3.tiles.virtualearth.net/tiles/a%s.png?g=1"%tile.quad_tree
        ul, br = tile.bounds
        image = ImageOverlay(
            url=url,
            bounds = tile.bounds,
            name = 'bing_basemap_%d'%zoom
        )
        my_map.add_layer(image)  


def on_change_zoom(change):
    if change['type'] == 'change' and change['name'] == 'zoom':
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

        for layer in my_map.layers:
            if layer.name == 'bing_basemap_%d'%int(change['old']):
                my_map.remove_layer(layer)

# my_map.observe(on_change_zoom)




field_movie = None
yield_img = None
yield_control = None
daily_img = None
maize_markers = []
def on_click(change):
    global field_id
    global field_bounds
    global doys
    global field_movie, field_lais, field_doys, field_cabs, yield_control, daily_img, maize_markers
    field_id = change["new"]

    ind = field_ids.index(field_id)

    feature = data['features'][ind]
    poly = geometry.Polygon(feature['geometry']['coordinates'][0])
    lon, lat = poly.centroid.coords[0]
    my_map.center = lat, lon
    zoom = 17
    my_map.zoom = zoom
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
    print(field_id)
    play_label.value = 'Click to play LAI movie over field: %s'%field_id
    home = os.getcwd()
    cwd = '/files/' + '/'.join(home.split('/')[3:])
    base_url = my_map.window_url.split('/lab/')[0] + cwd + '/'

    url, bounds, doys, yield_colorbar_f = get_lai_gif(field_id)

    # daily_img = None
#     if daily_img in my_map.layers:
#         my_map.remove_control(daily_img)

    if yield_control is not None:
        my_map.remove_control(yield_control)

    image = yield_colorbar_f.getvalue()
    output = widgetIMG(value=image, format='png',)
#     yield_colorbar = WidgetControl(widget=output, position='bottomleft', transparent_bg=True)
#     yield_colorbar.widget = output
#     my_map.add_control(yield_colorbar)

    ylds = np.array(field_yields[field_id])

    import scipy.stats as st

    cl = st.t.interval(0.95, len(ylds)-1, loc=np.mean(ylds), scale=st.sem(ylds))
    cl = (cl[1] + cl[0]) / 2
    
    yield_label = Label('%s Field yield: %.02f [%.02f, %.02f, %.02f]'%(field_id, np.mean(ylds), ylds[0], ylds[1], ylds[2]))
    yield_label2 = Label('$$Yield [kg/ha]$$')
    # label_box = HBox([play_label, maize_img], align_content = 'stretch', layout=Layout(width='100%', height='50%'))
    yield_box = VBox([yield_label, yield_label2, output], align_content = 'stretch', layout=Layout(width='100%', height='50%'))
    yield_control = WidgetControl(widget=yield_box, position='bottomleft')

    my_map.add_control(yield_control)

    url = 'data/S2_thumbs/S2_%s_yield.png'%(field_id)
    print(url)
    dates = [(datetime.datetime(2021, 1, 1) + datetime.timedelta(days=int(i-1))).strftime('%Y-%m-%d') for i in doys]
    slider2.options = dates
    field_bounds = bounds
    url = base_url + url

    field_mask = df.CODE == field_id

    field_doys = [int(datetime.datetime(2021, int(i.split('/')[1]), int(i.split('/')[0])).strftime('%j')) for i in df[field_mask].DATE]
    field_lais = np.array(df[field_mask].loc[:, 'LAI 1': 'LAI 4'])
    field_cabs = np.array(df[field_mask].loc[:, 'PLT. CHLRO 1': 'PLT. CHLOR 5'])

    # filed_cab_line = OHLC(x=dates2, y=np.array(prices2) / 60, marker='candle', 
    #                       scales={'x': sc_x, 'y': sc_y}, colors=['dodgerblue','orange'])


    # print(url)
#     if field_movie is not None:
#         my_map.remove_layer(field_movie)

#     field_movie = ImageOverlay(
#         url=url,
#         bounds = bounds,
#         name = 'S2_%s_lai_movie'%(field_id)
#     )
    # my_map.add_layer(field_movie)
    global yield_img

    if yield_img is None:
        # print(url)
        # print(field_bounds)
        # print('S2_%s_lai_png'%(field_id))
        yield_img = ImageOverlay(
        url=url,
        bounds = field_bounds,
        name = 'S2_%s_yield_png'%(field_id)
        )
        my_map.add_layer(yield_img)
        yield_img.url = url
        yield_img.bounds = field_bounds
    else:
        my_map.remove_layer(yield_img)
        yield_img = ImageOverlay(
            url=url,
            bounds = field_bounds,
            name = 'S2_%s_yield_png'%(field_id)
        )
        my_map.add_layer(yield_img)
        # daily_img.url = url
        # daily_img.bounds = field_bounds

    jslink((slider, 'value'), (yield_img, 'opacity') )

    try:
        my_map.remove_layer(daily_img)
    except:
        pass
    try:
        my_map.remove_layer(maize_marker)
    except:
        pass
    try:
        for this_marker, lai_ratio in maize_markers:
            my_map.remove_layer(this_marker)
    except:
        pass
    daily_img = None
    maize_marker = None
    maize_markers = []
    #for layer in my_map.layers:
        #if layer.name == 'S2_%s_lai_movie'%field_id:
            # Connect slider value to opacity property of the Image Layer


dropdown.observe(on_click, "value")
widget_control3 = WidgetControl(widget=dropdown, position="bottomleft")
my_map.add_control(widget_control3)


def on_change_slider2(change):

    if (change['name'] == 'value') & (change['type'] == 'change'):
        value = change["new"]
        old = change['old']
        ind = change['owner'].index

        global daily_img, lai_dot

        value = doys[ind]
        home = os.getcwd()
        cwd = '/files/' + '/'.join(home.split('/')[3:])
        base_url = my_map.window_url.split('/lab/')[0] + cwd + '/'

        url = 'data/S2_thumbs/S2_%s_lai_%03d.png'%(field_id, value)
        url = base_url + url
        field_bounds, _ = get_field_bounds(field_id)

#         if daily_img is not None:
#             my_map.remove_layer(daily_img)

#         daily_img = ImageOverlay(
#             url=url,
#             bounds = field_bounds,
#             name = 'S2_%s_lai_png'%(field_id)
#         )

#         my_map.add_layer(daily_img)
        # my_map.remove_layer(maize_marker) 

        # maize_marker.icon.icon_size = [(38*lai_ratio), int(95*lai_ratio)]
        if maize_marker is not None:
            lai_ratio = (pix_lai[ind] - 0) / (field_max - 0)
            
            
            lai_ratio = pix_lai[ind] / pix_lai.max()
            
            max_ind = np.argmax(pix_lai)
            max_ratio = pix_lai.max() / field_max
            
            if ind < max_ind:
                
                img_ind = int(lai_ratio * 5) + 1
                stage_ratio = lai_ratio / (img_ind / 5)
                
                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Ghana_workshop2022/imgs/maize.png', 
                            icon_size=[36.5*lai_ratio, 98.5*lai_ratio], 
                            icon_anchor=[36.5/2*lai_ratio, 98.5*lai_ratio])

                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                            icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                            icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])
                
                for (this_marker, max_ratio) in maize_markers:
 
                    
                    maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                                icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                                icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])

                    this_marker.icon = maize_icon
                    
            elif (lai_ratio > 0.8) & (ind >= max_ind):
                img_ind = 6
                stage_ratio = 1
                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                            icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                            icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])
            
                maize_marker.icon = maize_icon
                
                    
                for (this_marker, max_ratio) in maize_markers:
 
                    
                    maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                                icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                                icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])

                    this_marker.icon = maize_icon
            
                
            else:
                img_ind = np.min([int((1-lai_ratio / 0.8) * 6) + 7, 11])
                
                stage_ratio = (1-lai_ratio) / ((img_ind - 7) / 4)
                
                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Ghana_workshop2022/imgs/maize.png', 
                            icon_size=[36.5*lai_ratio, 98.5*lai_ratio], 
                            icon_anchor=[36.5/2*lai_ratio, 98.5*lai_ratio])

                maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                            icon_size=[28.4 * max_ratio , 82.6 * max_ratio], 
                            icon_anchor=[28.4/2 * max_ratio , 82.6 * max_ratio])
            
                maize_marker.icon = maize_icon
                
                stage_ratio = 1
                for (this_marker, max_ratio) in maize_markers:
 
                    
                    maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/maize_s%d.png'%img_ind, 
                                icon_size=[28.4 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio], 
                                icon_anchor=[28.4/2 * max_ratio * stage_ratio, 82.6 * max_ratio * stage_ratio])

                    this_marker.icon = maize_icon
                
            lai_dot.x = doys[[ind]]
            lai_dot.y = pix_lai[[ind]]

        # maize_marker.icon.icon_size=[38*lai_ratio, 95*lai_ratio] 
        # maize_marker.icon.icon_anchor=[22*lai_ratio,94*lai_ratio]
        # my_map.add_layer(maize_marker) 
        # print(maize_icon)   
        # print(lai_ratio)

        if daily_img is None:
            daily_img = ImageOverlay(
            url=url,
            bounds = field_bounds,
            name = 'S2_%s_lai_png'%(field_id)
            )
            my_map.add_layer(daily_img)
            daily_img.url = url
            daily_img.bounds = field_bounds
        else:
            daily_img.url = url
            daily_img.bounds = field_bounds
            daily_img.name = 'S2_%s_lai_png'%(field_id)
        # print(url)

play = Play(
    value=0,
    min=0,
    max=len(doys),
    step=1,
    interval=200,
    description="Press play",
    disabled=False
)

dates = [(datetime.datetime(2021, 1, 1) + datetime.timedelta(days=int(i-1))).strftime('%Y-%m-%d') for i in doys]
slider2 = SelectionSlider(options = dates, description='Date: ', style={'description_width': 'initial'}) 
slider2.observe(on_change_slider2)
# widget_control2 = WidgetControl(widget=slider2, position="bottomright")
jslink((play, 'value'), (slider2, 'index'))
# label = 
# display(label)
lai_colorbar_f = get_lai_color_bar()
image = lai_colorbar_f.getvalue()
output = widgetIMG(value=image, format='png',)
output.layout.object_fit = 'contain'
lai_label = Label('$$LAI [m^2/m^2]$$')
lai_box = VBox([lai_label, output])


play_label = Label('Click to play LAI movie over field: %s'%field_id)
# label_box = HBox([play_label, maize_img], align_content = 'stretch', layout=Layout(width='100%', height='50%'))
play_box = HBox([play, slider2])

play_box = VBox([play_label, play_box])

widget_control2 = WidgetControl(widget=play_box, position="bottomright")
my_map.add_control(widget_control2)
lai_control = WidgetControl(widget=lai_box, position="bottomright")
my_map.add_control(lai_control)

def random_color(feature):
    return {
        'color': 'black',
        'fillColor': np.random.choice(['red', 'yellow', 'green', 'orange']),
    }

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
    # style_callback=random_color
)



label = Label()
display(label)

maize_marker = None
pix_lai = None
maize_icon = None

def handle_interaction(**kwargs):
    # if kwargs.get('type') == 'mousemove':
    #     label.value = str(kwargs.get('coordinates'))
    if kwargs.get('type') == 'click':

        location=kwargs.get('coordinates')
        # print(location)
        point = geometry.Point(location[1], location[0])

        ind = field_ids.index(field_id)
        feature = data['features'][ind]
        poly = geometry.Polygon(feature['geometry']['coordinates'][0])

        if poly.contains(point):    
            label.value = 'Maize planted in field: %s'%field_id
            # global marker
            # if marker is not None:
            #     my_map.remove_layer(marker)
            # marker = Marker(location=location)

            global sels, planet_sur, doys
            global maize_icon, pix_lai, field_max, field_min, maize_marker, field_lai_boxes, lai_dot, maize_markers
            doys, pix_lai, pix_cab, sels, lais, planet_sur = get_pixel(location, field_id)

            mean_ref, mean_bios, std_ref, std_bios, sel_inds, u_mask = da_pix(sels, planet_sur, u_thresh = k_slider.value)

            show_inds = np.random.choice(range(200), 50)
            sels_to_show = sels[:, :, show_inds]

            pix_cab, pix_lai = mean_bios

            max_ind = np.argmax(pix_lai)

            field_max = lais[:, max_ind].max()
            field_min = lais[:, max_ind].min() 

            lai_ratio = (pix_lai.max() - field_min) / (field_max - field_min)
            lai_ratio = (pix_lai.max() - 0) / (field_max - 0)
            # print(field_max, field_min)
            # print(lai_ratio)

            icon = Icon(icon_url='https://leafletjs.com/examples/custom-icons/leaf-green.png', icon_size=[(38*lai_ratio), int(95*lai_ratio)], icon_anchor=[22,94])
            maize_icon = Icon(icon_url='https://gws-access.jasmin.ac.uk/public/nceo_ard/Ghana/Ghana_workshop2022/imgs/maize.png', 
                        icon_size=[36.5*lai_ratio, 98.5*lai_ratio], 
                        icon_anchor=[36.5/2*lai_ratio, 98.5*lai_ratio])

            maize_marker = Marker(location=location, icon=maize_icon, rotation_angle=0, rotation_origin='22px 94px', draggable=False)
            maize_markers.append([maize_marker, lai_ratio])
            
            my_map.add_layer(maize_marker)                        



            sels_to_show = sels[:, :, sel_inds]
            for ii in range(6):
                line, ax_x, ax_y = line_axs[ii]
                y_scale = LinearScale(min = sels_to_show[ii].T.min(), max = sels_to_show[ii].T.max())
                ax_y.scale = y_scale
                line.x = doys
                line.y = sels_to_show[ii].T
                line.scales = {"x": x_scale, "y": y_scale}

            # cab_fig = figy[2].children[0]
            # lai_fig = figy[2].children[1]

            # field_cab_boxes = Boxplot(x=field_doys[:-1], y=field_cabs[:-1], 
            #                           scales=cab_fig.marks[0].scales, box_fill_color='gray')
            # field_cab_boxes.auto_detect_outliers=False
            # # field_cab_line = Lines(x=field_doys[:-1], y=field_cabs[:-1].T, marker='cross', 
            # #                       scales=cab_fig.marks[0].scales, colors=['orange'])
            # field_cab_boxes.stroke = 'red'
            # field_cab_boxes.box_fill_color = 'blue'
            # field_cab_boxes.opacities = [0.5]
            # field_cab_boxes.box_width=5
            # cab_fig.marks = cab_fig.marks[:2] + [field_cab_boxes,]


            # field_lai_boxes = Boxplot(x=field_doys[:-1], y=field_lais[:-1], 
            #                           scales=lai_fig.marks[1].scales, box_fill_color='gray')
            # field_lai_boxes.auto_detect_outliers=False
            # # field_cab_line = Lines(x=field_doys[:-1], y=field_cabs[:-1].T, marker='cross', 
            # #                       scales=cab_fig.marks[0].scales, colors=['orange'])
            # field_lai_boxes.stroke = 'red'
            # field_lai_boxes.box_fill_color = 'blue'
            # field_lai_boxes.opacities = [0.4]
            # field_lai_boxes.box_width=5
            # lai_fig.marks = lai_fig.marks[:2] + [field_lai_boxes,]

            var_line = line_axs[-2]
            var_line.x = doys
            var_line.y = pix_cab
            var_line.scales = line_axs[4][0].scales
            # field_cab_boxes.scales = var_line.scales

            var_line = line_axs[-1]
            var_line.x = doys
            var_line.y = pix_lai
            var_line.scales = line_axs[5][0].scales
            field_lai_boxes.scales = var_line.scales
            field_lai_boxes.x = field_doys
            field_lai_boxes.y = field_lais
            lai_dot.scales = var_line.scales

            for ii in range(4):
                line, ax_x, ax_y = line_axs[ii]
                ref_line = ref_lines[ii]
                ref_line.scales = line.scales
                ref_line.x = doys[~u_mask]
                ref_line.y = planet_sur[ii][~u_mask]

            # print(planet_sur.shape, u_mask.shape)
            for ii in range(4):
                line, ax_x, ax_y = line_axs[ii]
                good_ref_line = good_ref_lines[ii]
                good_ref_line.scales = line.scales
                good_ref_line.x = doys[u_mask]
                good_ref_line.y = planet_sur[ii][u_mask]

        else:
            label.value = 'Not in field: %s'%field_id
            # print('Not in field: %s'%field_id)


def on_change_k_slider(change):

    if (change['name'] == 'value') & (change['type'] == 'change'):
        value = change["new"]
        old = change['old']
        mean_ref, mean_bios, std_ref, std_bios, sel_inds, u_mask = da_pix(sels.copy(), planet_sur.copy(), u_thresh = value)

        show_inds = np.random.choice(range(200), 50)
        sels_to_show = sels[:, :, show_inds]

        global field_lai_boxes

        sels_to_show = sels[:, :, sel_inds]
        for ii in range(6):
            line, ax_x, ax_y = line_axs[ii]
            y_scale = LinearScale(min = sels_to_show[ii].T.min(), max = sels_to_show[ii].T.max())
            # ax_y.scale = y_scale
            ax_y.scale = ax_y.scale
            line.x = doys
            line.y = sels_to_show[ii].T
            line.scales = {"x": x_scale, "y": y_scale}

        pix_cab, pix_lai = mean_bios

        var_line = line_axs[-2]
        var_line.x = doys
        var_line.y = pix_cab
        var_line.scales = line_axs[4][0].scales


        var_line = line_axs[-1]
        var_line.x = doys
        var_line.y = pix_lai
        var_line.scales = line_axs[5][0].scales

        var_line = line_axs[-2]
        var_line.x = doys
        var_line.y = pix_cab
        var_line.scales = line_axs[4][0].scales
        # field_cab_boxes.scales = var_line.scales

        var_line = line_axs[-1]
        var_line.x = doys
        var_line.y = pix_lai
        var_line.scales = line_axs[5][0].scales
        field_lai_boxes.scales = var_line.scales
        field_lai_boxes.x = field_doys
        field_lai_boxes.y = field_lais

        for ii in range(4):
            line, ax_x, ax_y = line_axs[ii]
            ref_line = ref_lines[ii]
            ref_line.scales = line.scales
            ref_line.x = doys[~u_mask]
            ref_line.y = planet_sur[ii][~u_mask]

        # print(planet_sur.shape, u_mask.shape)
        for ii in range(4):
            line, ax_x, ax_y = line_axs[ii]
            good_ref_line = good_ref_lines[ii]
            good_ref_line.scales = line.scales
            good_ref_line.x = doys[u_mask]
            good_ref_line.y = planet_sur[ii][u_mask]

k_slider.observe(on_change_k_slider)

my_map.on_interaction(handle_interaction)
my_map.add_layer(fields)
my_map.add_layer(points)
control = LayersControl(position='topleft')
my_map.add_control(control)


# my_map
