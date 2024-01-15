To run code, execute in the noted order

1_earth_engine.py (requires Google Earth Engine)

Manually download the data from your Google Drive to folder 

2_lakes_rivers.R (requires Hydrorivers(https://www.hydrosheds.org/products/hydrorivers) and Hydrolakes (https://www.hydrosheds.org/products/hydrolakes) datasets)
3_dataset_creation.py (requires Cartographic Boundary Files (https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html) and NID dataset (https://www.fema.gov/emergency-managers/risk-management/dam-safety/national-inventory-dams))
4_subselection_rivers.py
5_cross_validation_splits.py
6_model_fitting_a1.py (has to be repeated for other cross validations and dam subsets, see file)
7_evaluation.py
