# Global Predictability of Marine Heatwave-Induced Rapid Intensification of Tropical Cyclones

This repository contains Python scripts developed for the analysis and prediction of marine heatwave-induced rapid intensification (RI) of tropical cyclones (TCs) using a global machine learning (ML) approach. The study aims to improve RI forecasts by integrating marine heatwave (MHW) characteristics into predictive models. The results and methods are detailed in the manuscript: *Global Predictability of Marine Heatwave-Induced Rapid Intensification of Tropical Cyclones*.

## Cite

If you use the codes, data, ideas, or results from this project, please cite the following paper:

**Radfar, S., Foroumandi, E., Moftakhari, H., Moradkhani, H., Sen Gupta, A., and Foltz, G. (2024). Global predictability of marine heatwave-induced rapid intensification of tropical cyclones. Earth’s Future.**

- **Link to the Published Paper:** [Earth’s Future Journal](https://doi.org/10.1029/xxxxxxx)

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

To run the code in this repository, you'll need to have the following dependencies installed:

### Python Dependencies
- Python 3.7 or higher
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- Joblib
- SHAP
- Basemap

You can install the required Python packages using pip:
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost lightgbm joblib shap basemap
```

## Usage

Each script file has a description on top that clearly describes the objectives of that code and expected outputs.

## File Structure
```bash
├── scripts/
│   ├── [Cleaned Python Scripts]
├── LICENSE
└── README.md
```

## Data

All data supporting the findings of this study are publicly accessible and available for download. The analysis covers the time period from September 1, 1981, to October 19, 2023. The datasets used include:

1. **Tropical Cyclone Best Track Data**:
   - TC best track data were obtained from the **International Best Track Archive for Climate Stewardship (IBTrACS)** dataset (Knapp et al., 2018). This dataset is freely available in CSV format through the National Centers for Environmental Information (NCEI) website. Access the data here: [IBTrACS CSV Format](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/).

2. **Sea Surface Temperature (SST) Data**:
   - SST data were sourced from the **NOAA Optimum Interpolation Sea Surface Temperature (OISST) version 2.1** dataset (Huang et al., 2021). The dataset is provided in NetCDF format and can be accessed through the ERDDAP data server. Access the data here: [NOAA OISST v2.1](https://www.ncei.noaa.gov/erddap/info/index.html?page=1&itemsPerPage=1000).

These datasets are publicly available, ensuring the transparency and reproducibility of the results presented in this study.

## Results
The main output of this analysis is a set of visualizations and machine learning models examining the impact of marine heatwaves on the rapid intensification of tropical cyclones globally. The results are provided in the cited manuscript.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the Apache License.

## Acknowledgments
This research is supported by the Coastal Hydrology Lab and the Center for Complex Hydrosystems Research at the University of Alabama. Funding was awarded to Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement with The University of Alabama (NA22NWS4320003). Partial support was also provided by NSF award # 2223893.

## Contact
For any questions or inquiries, please contact the project maintainer at [sradfar@ua.edu].
