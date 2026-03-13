# ================================================================
# Google Earth Engine — Refined Flood Frequency Analysis (No GHSL)
# Uses Relative Thresholding and Essential Masking for Accuracy
# ================================================================
# Improvements Implemented:
# 1. Relative Thresholding: Uses a dynamic threshold (2.5 dB drop) relative to
#    the dry-season VV baseline, improving accuracy over static thresholds.
# 2. Speckle Filtering: Applies a focal median filter to reduce radar noise.
# 3. Masking Layers: Integrates public JRC (Permanent Water) and SRTM (Steep Slopes)
#    masks to exclude known false-positive areas.
# 4. Efficiency: Dry-season baseline is calculated and cached once per year.
#
# FIX: The build_outputs function has been refactored to use .getInfo() sequentially
#      inside the loop, preventing the HttpError 429 (Too many concurrent aggregations).
# ================================================================

import ee
import geemap
import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict

try:
    ee.Initialize(project='your_ee_project')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='your_ee_project')

GEOJSON_PATH = '/content/kilifi_chu_10_cleaned.geojson'

chu_gdf = gpd.read_file(GEOJSON_PATH)
if chu_gdf.crs is None or chu_gdf.crs.to_epsg() != 4326:
    chu_gdf = chu_gdf.to_crs(epsg=4326)


def gdf_to_ee(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """Converts a GeoDataFrame to an Earth Engine FeatureCollection."""
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        props = row.drop(labels=['geometry']).to_dict()
        try:
            if geom.geom_type == 'MultiPolygon':
                rings = [list(part.exterior.coords) for part in geom.geoms]
                ee_geom = ee.Geometry.MultiPolygon(rings)
            elif geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                ee_geom = ee.Geometry.Polygon([coords])
            else:
                continue
            features.append(ee.Feature(ee_geom, props))
        except Exception as e:
            print(f"Skipping invalid geometry: {e}")
    return ee.FeatureCollection(features)


chu_fc: ee.FeatureCollection = gdf_to_ee(chu_gdf)
AOI = chu_fc.geometry().bounds()

DB_DIFFERENCE = 2.5
DRY_MONTHS = [1, 2, 3]

JRC = ee.Image('JRC/GSW1_3/GlobalSurfaceWater').select('occurrence')
SRTM = ee.Image('USGS/SRTMGL1_003')

ROUND_YEAR = {1:2017, 2:2017, 3:2018, 4:2018, 5:2019, 6:2019, 8:2020, 10:2021, 12:2022, 14:2023, 16:2024}
ZONES = ["Kibwabwani","Mutsengo","Viragoni","Mnazimwenga","Mwele-Kisurutini",
         "Tsangatsini","Buni","Kombeni","Kwale","Vishakani"]
GROUP_A_ROUNDS = {1,3,5}
MONTHS_A = {"Kibwabwani":"Dec","Mutsengo":"Jan","Viragoni":"Jan","Mnazimwenga":"Feb","Mwele-Kisurutini":"Feb",
            "Tsangatsini":"Mar","Buni":"Mar","Kombeni":"Apr","Kwale":"Apr","Vishakani":"May"}
MONTHS_B = {"Kibwabwani":"Jun","Mutsengo":"Jun","Viragoni":"Jul","Mnazimwenga":"Jul","Mwele-Kisurutini":"Aug",
            "Tsangatsini":"Aug","Buni":"Sep","Kombeni":"Sep","Kwale":"Oct","Vishakani":"Oct16-Nov15"}
ALL_ROUNDS = [1,2,3,4,5,6,8,10,12,14,16]
MONTH_IDX = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

ZONE_PROP_CANDIDATES = ["CHU","chu_name","Zone","zone","name"]
zone_prop = 'zone_clean'
orig_zone_prop = [c for c in ZONE_PROP_CANDIDATES if c in chu_fc.first().propertyNames().getInfo()][0]


def add_clean_zone(f):
    return f.set({'zone_clean': ee.String(f.get(orig_zone_prop)).trim()})


chu_fc = chu_fc.map(add_clean_zone)


def _month_start_end(year:int, label:str) -> Tuple[ee.Date, ee.Date, str]:
    """Return (start, endExclusive, prettyLabel) for a month label or range."""
    if label == "Oct16-Nov15":
        start = ee.Date(f"{year}-10-16")
        end   = ee.Date(f"{year}-11-16")
        return start, end, "Oct16–Nov15"
    m = MONTH_IDX[label]
    start = ee.Date.fromYMD(year, m, 1)
    end = start.advance(1, 'month')
    return start, end, label


def get_period(round_num:int, zone_name:str):
    """Given a round and zone, return (year, start, end, month_or_range)."""
    year = ROUND_YEAR[round_num]
    label_map = MONTHS_A if round_num in GROUP_A_ROUNDS else MONTHS_B
    month_lbl = label_map[zone_name]
    start, end, pretty = _month_start_end(year, month_lbl)
    return year, start, end, pretty


def get_dry_season_baseline(year: int, geometry: ee.Geometry) -> ee.Image:
    """Calculates the median VV backscatter during the dry season (Jan-Mar) for the year."""
    dry_start = ee.Date.fromYMD(year, DRY_MONTHS[0], 1)
    dry_end = ee.Date.fromYMD(year, DRY_MONTHS[-1], 1).advance(1, 'month')

    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterDate(dry_start, dry_end)
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
           .filter(ee.Filter.eq('instrumentMode','IW'))
           .filterBounds(geometry)
           .select('VV'))

    perm_water_mask = JRC.gte(90).unmask(0).Not()
    median_vv = col.median()

    return ee.Algorithms.If(
        median_vv,
        ee.Image(median_vv).updateMask(perm_water_mask).rename('baseline'),
        ee.Image(-25).rename('baseline')
    )


def frequency_image(start:ee.Date, end:ee.Date, baseline:ee.Image) -> ee.Image:
    """Build per-pixel mean flood frequency image for [start, end) using relative threshold."""

    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterDate(start, end)
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
           .filter(ee.Filter.eq('instrumentMode','IW'))
           .filterBounds(AOI)
           .select('VV'))

    slope = ee.Terrain.slope(SRTM)
    slope_mask = slope.lt(10)
    perm_water_mask = JRC.gte(90).unmask(0).Not()
    total_mask = slope_mask.And(perm_water_mask)

    def to_flood(img: ee.Image) -> ee.Image:
        img_filtered = img.focal_median(1).rename('VV_filtered')
        baseline_img = ee.Image(baseline)
        threshold_img = baseline_img.subtract(DB_DIFFERENCE)
        flood = img_filtered.lt(threshold_img).rename('flood')
        flood = flood.updateMask(total_mask)
        return flood.updateMask(img.mask()).unmask(0)

    fcol = col.map(to_flood)
    sum_img = fcol.sum()
    cnt_img = fcol.count()
    freq = sum_img.divide(cnt_img.updateMask(cnt_img.gt(0))).unmask(0).rename('freq')

    return freq.set({'nScenes': fcol.size(),
                     'threshold_method': f"relative_{DB_DIFFERENCE}dB_drop_w_SRTM_JRC"})


def build_outputs(scale:int=10) -> Tuple[ee.FeatureCollection, pd.DataFrame]:
    """
    Computes flood frequency for all rounds/zones, using annual baselines.
    This version computes features sequentially using getInfo() to avoid HTTP 429 errors.
    Returns: (ee.FeatureCollection for export, pd.DataFrame for local preview)
    """

    baseline_cache: Dict[int, ee.Image] = {}
    freq_cache: Dict[str, ee.Image] = {}
    python_rows = []
    ee_features_to_export = []

    for r in ALL_ROUNDS:
        year = ROUND_YEAR[r]
        if year not in baseline_cache:
            print(f"Calculating dry-season baseline for year {year}...")
            baseline_cache[year] = get_dry_season_baseline(year, AOI)

        current_baseline = baseline_cache[year]

        for zone in ZONES:
            _, start, end, label = get_period(r, zone)
            key = f"R{r}_{year}_{label}"
            if key not in freq_cache:
                print(f"Calculating frequency image for {key}...")
                freq_cache[key] = frequency_image(start, end, current_baseline)

            freq_img = freq_cache[key]
            z_fc = chu_fc.filter(ee.Filter.eq(zone_prop, zone))
            z_feat = z_fc.first()

            stats = ee.Dictionary({})
            stats = ee.Algorithms.If(
                z_feat,
                ee.Image(freq_img).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee.Feature(z_feat).geometry(),
                    scale=scale,
                    maxPixels=1e13
                ),
                ee.Dictionary({})
            )
            stats = ee.Dictionary(stats)
            nScenes = ee.Number(freq_img.get('nScenes'))

            try:
                computed_stats = stats.getInfo()
                nScenes_val = nScenes.getInfo()
                mean_freq_val = computed_stats.get('freq')
                flag = ''
                if nScenes_val == 0:
                    flag = 'no_imagery'
                elif nScenes_val < 2:
                    flag = 'low_imagery'

                python_row_data = {
                    'round': f"R{r}",
                    'round_num': r,
                    'year': year,
                    'zone': zone,
                    'month_or_range': label,
                    'start': start.format('YYYY-MM-dd').getInfo(),
                    'end': end.format('YYYY-MM-dd').getInfo(),
                    'mean_flood_freq': mean_freq_val,
                    'mean_flood_freq_pct': (mean_freq_val * 100) if mean_freq_val is not None else None,
                    'nScenes': nScenes_val,
                    'flag': flag,
                    'method': freq_img.get('threshold_method').getInfo()
                }
                python_rows.append(python_row_data)
                ee_features_to_export.append(ee.Feature(None, python_row_data))

            except ee.EEException as e:
                print(f"EE Error during calculation for {key} in {zone}: {e}")
                python_rows.append({
                    'round': f"R{r}", 'round_num': r, 'year': year, 'zone': zone,
                    'month_or_range': label, 'start': 'N/A', 'end': 'N/A',
                    'mean_flood_freq': None, 'mean_flood_freq_pct': None,
                    'nScenes': 0, 'flag': 'ee_compute_error', 'method': 'N/A'
                })

    results_fc = ee.FeatureCollection(ee_features_to_export)
    df = pd.DataFrame(python_rows)
    return results_fc, df


results_fc, df = build_outputs(scale=10)
print(f"Total rows successfully computed: {df.shape[0]}")

export_task = ee.batch.Export.table.toDrive(
    collection=results_fc,
    description='CHU_FloodFrequency_Relative_SRTM_JRC_Masked',
    folder='EarthEngine_Exports',
    fileFormat='CSV'
)
export_task.start()
print("Export started: check Tasks tab in the Code Editor or ee.batch.Task.list() in Python.")

try:
    df.columns = ['Round', 'RoundNum', 'Year', 'Zone', 'MonthOrRange', 'Start', 'End', 'MeanFloodFreq', 'MeanFloodFreq_%', 'nScenes', 'Flag', 'Method']
    round_order = (df[['Round','RoundNum']].drop_duplicates().sort_values('RoundNum'))['Round'].tolist()

    print("\n--- Mean Flood Frequency (%) ---")
    pivot_df = (df.pivot_table(index='Zone', columns='Round', values='MeanFloodFreq_%', aggfunc='mean')
                .reindex(columns=round_order))
    display(pivot_df)

    print("\n--- Scenes Used (Count) ---")
    scenes_pivot = (df.pivot_table(index='Zone', columns='Round', values='nScenes', aggfunc='max')
                    .reindex(columns=round_order))
    display(scenes_pivot)
except Exception as e:
    print("Preview tables skipped:", e)

try:
    first_row = df.iloc[0]
    ex_round = first_row['RoundNum']
    ex_year  = first_row['Year']
    ex_label = first_row['MonthOrRange']
    ex_start = ee.Date(first_row['Start'])
    ex_end   = ee.Date(first_row['End'])

    ex_baseline = get_dry_season_baseline(ex_year, AOI)
    example_img = frequency_image(ex_start, ex_end, ex_baseline).select('freq')

    Map = geemap.Map()
    Map.centerObject(chu_fc, 9)
    Map.addLayer(example_img, {'min':0, 'max':1, 'palette':['#ffffff','#d0e4f7','#73b2e7','#005dae','#00008b']}, f"Freq R{ex_round}_{ex_year}_{ex_label}")
    Map.addLayer(chu_fc, {'color': 'FF0000', 'fillColor': '00000000'}, 'CHU Boundaries')
    Map
except Exception as e:
    print("Map preview skipped:", e)
