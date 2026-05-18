## Project summary

This repository contains the analytical code used to derive Community Health Unit level flood exposure from Sentinel 1 synthetic aperture radar imagery, fit longitudinal mixed effects models of water and sanitation outcomes, construct comparative resilience scores, and run supplementary sensitivity analyses for bounded outcome modelling. The study examines how repeated flood exposure is associated with safe water access and functional latrine access across ten Community Health Units in Kilifi County, Kenya, using repeated CHU level observations collected between 2017 and 2024.

## Repository contents

This repository is organized into code, documentation, and metadata components to support transparency and reproducibility.

### Suggested folder structure

- `code/`  
  Contains the main analysis scripts.
  - `flood_exposure_gee_relative_threshold.py`  
    Google Earth Engine workflow for deriving CHU level flood frequency from Sentinel 1 data.
  - `resilience_model_main_analysis_from_notebook.py`  
    Main longitudinal modelling and resilience score construction workflow.
  - `sensitivity_analysis_colab_ready.py`  
    Supplementary bounded outcome sensitivity analyses.


## Software requirements

The code was developed in Python and Google Earth Engine. The following packages are required for local execution of the Python workflows:

- earthengine-api
- geemap
- pandas
- geopandas
- numpy
- statsmodels
- scipy
- matplotlib
- scikit-learn
- patsy
- jupyter

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


## Citation
If you use this code, please cite:
Oluoch F, Ondiek RI, Gudda F, et al. Satellite-Derived Longitudinal Evidence of
Flood Effects on Community Water and Sanitation Resilience in Coastal Kenya.
BMC Global and Public Health. 2026. [DOI to be added]

Code archive: [https://doi.org/10.5281/zenodo.19108284]
