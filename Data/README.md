# Data Directory

This directory describes the expected data structure for the bachelor project code.

Large raw GNSS data files are not included in this repository. This includes ISMR receiver files, SP3 precise orbit files, extracted intermediate data files, and large processed data products. These files are excluded because of file size, data ownership, and reproducibility considerations.

The code and notebooks are written to use relative paths where possible. Local absolute paths, such as `C:\Users\...`, should not be stored in the repository.

## Expected data structure

The recommended local data structure is:

```text
data/
├── README.md
├── ISMR/
│   ├── SWADO/
│   │   └── ...
│   └── CHAIN/
│       └── ...
├── SP3/
│   └── ...
├── stations/
│   ├── SWADO_stations.csv
│   └── CHAIN_stations.csv
└── processed/
    └── ...
