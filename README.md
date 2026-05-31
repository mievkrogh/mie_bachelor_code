# Bachelor Project Code: GNSS-Based Ionospheric Disturbance Mapping in the Arctic

This repository contains Python code developed for a bachelor project on grid-based mapping of space-weather-related ionospheric disturbances in the Arctic using GNSS observations.

The project investigates how GNSS-derived observations can be represented spatially using ionospheric pierce points (IPPs) and a hierarchical grid structure in an Arctic Lambert Azimuthal Equal Area (LAEA) projection. The purpose is to support transparent visualisation of ionospheric disturbance levels together with information about observational coverage.

## Project overview

The code supports the following main tasks:

- reading and processing GNSS scintillation monitoring data from ISMR files
- reading precise satellite orbit information from SP3 files
- computing satellite-receiver geometry and ionospheric pierce points (IPPs)
- comparing IPPs derived from different geometry sources
- assigning IPPs to a hierarchical Arctic grid
- computing a simplified AIMS disturbance index, \(I_T\)
- aggregating disturbance information per grid cell
- estimating coverage quality for grid cells
- producing static and interactive visualisations of IPPs, grid cells, disturbance levels, and coverage information

The mapping framework is based on the idea that ionospheric disturbance observations should only be shown where GNSS observations are available. This avoids filling data gaps through interpolation and makes the resulting maps more transparent in regions with sparse or uneven receiver coverage.


## Data

Large raw GNSS data files are not included in this repository. This includes SP3 orbit files and ISMR receiver files. The expected data structure is described in `data/README.md`.

The notebooks use relative paths and should not contain local absolute paths such as `C:\Users\...`.


## How to use

The main processing workflow is demonstrated in the notebooks in the `notebooks/` folder. The functions used for reading input data, computing IPP locations, assigning observations to grid cells, calculating disturbance indices and producing visualisations are stored in the `Functions/` folder.

Some notebooks contain interactive Bokeh/Plotly outputs. If GitHub cannot render a notebook preview, the notebook should be downloaded and opened locally in Jupyter Notebook or JupyterLab.

Large input data files are not included. To run the notebooks, the required SP3 and ISMR files must be placed according to the structure described in `data/README.md`.

## Repository structure

```text
bachelor-gnss-ionosphere-code/
├── README.md
├── .gitignore
├── Functions/
│   ├── collective0.py
│   ├── ipp_on_map.py
│   ├── ToD_grid.py
│   └── AIMS_IT_value.py
│   └── sp3_reader.py
│   └── sp3_spline.py
│   └── ismr_files.py
│   └── get_stations.py
├── notebooks/
│   └── ipp_pipeline.ipynb
│   └── Interactive_Product.ipynb
│   └── Design_2D_gridd.ipynb
└── data/
    └── README.md
```
## Author

Mie Vassard Krogh  
Bachelor project, Geophysics and Space Technology  
Technical University of Denmark, DTU Space  
2026
