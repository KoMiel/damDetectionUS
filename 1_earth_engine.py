
### this file downloads .tif files with image, elevation and water layers from US states using Google Earth Engine



### import packages

import ee
import pandas as pd


### parameter definitions

RESOLUTION = 10 # resolution of each pixel: the smaller, the longer it takes
MAX_PIXELS = 30000000000  # a large value to prevent stopping
START = '2017-12-01'  # date range: the larger the longer it takes, but clouds will be filtered out better
END = '2020-11-30'
CLOUD_SHADOW_PRECISION = 20 # cloud filtering parameters
CLOUD_FILTER = 60
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100



### functions to filter out clouds (see https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless)

def get_s2_sr_cld_col(aoi, start_date, end_date, layers):
    # Import and filter S2 SR.
    s2_sr_col = layers.filterBounds(aoi).filterDate(start_date, end_date)\
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))

    # Import and filter s2cloudless.
    s2_cloudless_col = s2_cloudless.filterBounds(aoi).filterDate(start_date, end_date)

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    sr_band_scale = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*sr_band_scale).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/CLOUD_SHADOW_PRECISION)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': CLOUD_SHADOW_PRECISION})
                   .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B4', 'B3', 'B2', 'B12').updateMask(not_cld_shdw)



### main code

# initialize earth engine
ee.Initialize()

# import images and image collections
sentinel = ee.ImageCollection("COPERNICUS/S2_SR")
elevation = ee.Image("JAXA/ALOS/AW3D30/V2_2")
s2_cloudless = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
s2_sr = ee.ImageCollection('COPERNICUS/S2_SR')
states = ee.FeatureCollection('TIGER/2018/States')
surface_water = ee.Image("JRC/GSW1_3/GlobalSurfaceWater")

# import states table
statesTable = pd.read_csv('states.csv', delimiter='\t', dtype=str, header=0)

# loop over all states
for i in range(len(statesTable)):

    # get information on state
    stateID = statesTable['Code '][i]
    state = statesTable['State '][i]

    # get area of state
    roi = states.filter(ee.Filter.eq('STATEFP', stateID.strip()))
    AOI = roi.geometry()
    AOImercator = AOI.transform(proj='EPSG:3395', maxError=0.1)

    # get sentinel data
    s2_sr_cld_col = get_s2_sr_cld_col(AOI, START, END, s2_sr)

    # filter out clouds and calculate median
    s2_sr_median = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median()

    # add elevation
    s2_sr_median_elev = s2_sr_median.addBands(elevation)

    # add surface water
    s2_sr_median_water = s2_sr_median_elev.addBands(surface_water)

    # select the relevant bands
    s2_sr_median_water = s2_sr_median_water.select(['B4', 'B3', 'B2', 'AVE_DSM', 'occurrence', 'B12'])

    # convert to common data type
    s2_sr_median_water = s2_sr_median_water.toFloat()

    # project to Mercator coordinates
    s2_sr_median_water = s2_sr_median_water.reproject('EPSG:3395', scale=RESOLUTION)

    # task configuration
    task_config = {
        'description': state.strip(),
        'folder': 'damImages',
        'scale': RESOLUTION,
        'region': AOImercator,
        'maxPixels': MAX_PIXELS
    }

    # set up task
    task = ee.batch.Export.image.toDrive(s2_sr_median_water, **task_config)

    # start task
    task.start()

    # print message
    print('Export Image %d was submitted, please wait ...' % i)
