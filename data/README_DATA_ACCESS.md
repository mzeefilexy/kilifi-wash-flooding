# Data Access

This repository contains analysis code only. No individual-level or CHU-level data are included.

## CHU-level surveillance data

The aggregated Community Health Unit (CHU) level dataset on safe water access, functional latrine access, and household characteristics was derived from the Kaloleni Rabai Health and Demographic Surveillance System (KRHDSS) in Kilifi County, Kenya. This dataset is available from the corresponding author (Dr Felix Oluoch, oluoch.felix@aku.edu) on reasonable request, subject to approval by the KRHDSS governance structures and the Aga Khan University Institutional Ethics Review Committee.

## Publicly available geospatial datasets

All geospatial datasets used in this study are freely available from their original providers:

| Dataset | Source | Access |
|---------|--------|--------|
| Sentinel-1 GRD (C-band SAR) | European Space Agency via Google Earth Engine | `ee.ImageCollection('COPERNICUS/S1_GRD')` |
| ESA WorldCover 10 m 2021 | ESA / Zenodo | https://zenodo.org/record/7254221 |
| SRTM GL1 (30 m DEM) | NASA / USGS via Google Earth Engine | `ee.Image('USGS/SRTMGL1_003')` |
| JRC Global Surface Water | European Commission Joint Research Centre via Google Earth Engine | `ee.Image('JRC/GSW1_4/GlobalSurfaceWater')` |
| Relative Wealth Index | Meta Data for Good | https://dataforgood.facebook.com/dfg/tools/relative-wealth-index |

## Reproducing the analysis

1. Run `flood_exposure_gee_relative_threshold.py` in Google Earth Engine (requires an Earth Engine account and access to the CHU boundary shapefile).
2. Run `resilience_model_main_analysis_from_notebook.py` in Python with the CHU-level panel dataset.
3. Run `sensitivity_analysis_colab_ready.py` for bounded-outcome sensitivity analyses.

See `requirements.txt` for the Python package dependencies.
