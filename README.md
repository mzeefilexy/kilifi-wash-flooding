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

- `docs/`  
  Contains supporting documentation.
  - `methods_reproducibility_notes.md`
  - `reviewer_response_summary.md`
  - `figures_tables_manifest.md`

- `data/`  
  Contains metadata, data dictionaries, and placeholders only unless permissions allow data sharing.
  - `README_data.md`
  - `example_or_metadata_only/`

- `outputs/`  
  Recommended location for model outputs, tables, and figures generated from the code.

- `requirements.txt`  
  Python package requirements.

- `LICENSE`  
  Repository license.

- `CITATION.cff`  
  Citation metadata for the repository.

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

A suggested `requirements.txt` file is:

```text
earthengine-api
geemap
pandas
geopandas
numpy
statsmodels
scipy
matplotlib
scikit-learn
patsy
jupyter
