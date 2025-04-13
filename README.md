# Synergistic Impact of Marine Heatwaves and Rapid Intensification Exacerbates Tropical Cyclone Destructive Power Worldwide

This repository contains Python scripts developed for the analysis presented in the Science Advances paper titled *"Synergistic impact of marine heatwaves and rapid intensification exacerbates tropical cyclone destructive power worldwide."* The scripts support statistical, geospatial, and machine learning analyses to investigate the compounding effects of marine heatwaves (MHWs) and rapid intensification (RI) on tropical cyclone (TC) hazards and damages.

## Cite

If you use the codes, data, ideas, or results from this project, please cite the following paper:

**Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Sen Gupta, A., and Foltz, G. (2025). Synergistic impact of marine heatwaves and rapid intensification exacerbates tropical cyclone destructive power worldwide. Science Advances.**

- **DOI Link:** [https://doi.org/10.1126/sciadv.adkxxxx](https://doi.org/10.1126/sciadv.adkxxxx) *(placeholder)*

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation

To run the code in this repository, you'll need the following dependencies:

### Python Dependencies
- Python 3.7 or higher
- numpy
- pandas
- matplotlib
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- statsmodels
- tqdm
- seaborn
- cartopy
- basemap-python

Install all dependencies using pip:
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost lightgbm shap statsmodels tqdm seaborn cartopy basemap-python
```

## Usage

Each Python script includes a full header that clearly describes the objectives, outputs, and context within the study. Please refer to those for individual script details.

## File Structure
```bash
├── scripts/
│   ├── [Cleaned Python Scripts]
├── LICENSE
└── README.md
```

## Data

The study integrates multiple global datasets from publicly available sources:

1. **Tropical Cyclone Best Track Data (IBTrACS)**  
   Source: NOAA NCEI  
   Format: CSV (3-hourly resolution, global coverage)  
   Access: [https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/)

2. **Sea Surface Temperature (OISST v2.1)**  
   Source: NOAA ERDDAP  
   Format: NetCDF  
   Access: [https://www.ncei.noaa.gov/erddap/info/index.html](https://www.ncei.noaa.gov/erddap/info/index.html)

3. **Precipitation (ERA5)**  
   Source: ECMWF Copernicus Climate Data Store  
   Format: Reanalysis at 0.25° spatial resolution  
   Access: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)

4. **Economic Loss Data (EM-DAT)**  
   Source: Emergency Events Database  
   Access: [https://www.emdat.be/](https://www.emdat.be/)

5. **Built-Up Volume Data (GHSL)**  
   Source: Copernicus Emergency Management Service  
   Format: 30 arcsecond global raster grids (1980–2025)  
   Access: [https://human-settlement.emergency.copernicus.eu/download.php](https://human-settlement.emergency.copernicus.eu/download.php)

## Results

The repository contains figures and models supporting all key results in the manuscript, including:

- Marine heatwave trends  
- RI onset statistics  
- Basin-level machine learning performance  
- Cost quantile regressions  
- Copula-based TC risk visualizations  
- Predictor importance analysis using SHAP  

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements or questions.

## License

This project is licensed under the MIT License.

## Acknowledgments

This research is supported by the Coastal Hydrology Lab and the Center for Complex Hydrosystems Research at the University of Alabama. Funding was awarded to Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement with The University of Alabama (NA22NWS4320003). Partial support was also provided by NSF award #2223893.

## Contact

For questions, please contact Soheil Radfar at [sradfar@ua.edu].
