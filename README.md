# Zambia Drought Analysis

## TODO

1. Calculate payout percentages and run k-means on that (setting equal to admin zones and optimized)
2. Come up with a more intuitive way of calculating payout (a likelihood where we say the probability of getting a payout is X%)
3. Attempt g-means and hope it works

### For next weel
-Talk about which ones to report, and which ones are the most intepretable (what to keep)

## Steps

**1a. Run `Rainfall Extraction.ipynb`**
- Extracts rainfall data at 5-day intervals (e.g., Jan 1, Jan 6, etc.) for each pixel
- Output files are named like:
  
  `19900101_y1990m01p01_Precipitation.csv`  
  `19900106_y1990m01p02_Precipitation.csv`

- Also generates two additional CSV files containing pixel center coordinates:

  `AOI_latitude.csv`  
  `AOI_longitude.csv`

- All files are saved to:  
  `data/pixel data/`

**1b. Run `Boundary Extraction.ipynb`**

- Loads FAO GAUL Level 2 administrative district boundaries
- Loads Zambia's Agro-Ecological Regions from:

  `data/Agro-ecological_regions_of_Zambia.geojson`

- Filters administrative districts where at least 70% of the area falls within Agro-Ecological Regions I, IIA, or IIB

- Exports the following GeoJSON files to `data/boundaries/`:

  `zambia_boundary.geojson`  
  `zambia_admin2_boundaries.geojson`  
  `mod_agro_eco_outer_boundary.geojson`


**2 Run `Loading and Preprocessing Pixels `**

- Reads in pixel rainfall data and `AOI_latitude.csv` and `AOI_longitude.csv`

- Creates xarray dataset

- Masks precipitation to AOI in `mod_agro_eco_outer_boundary.geojson`

- Converts xarray to geodataframes

- Preprocess data

  - Aggregate Monthly Precipitation for Each Pixel
  - Extracts Precipitation for Rain Season Months
  - Aggregates to Zambia's Agricultural Season

- Exports `seasonal_precipitation_touse.feather` to `data/precipitation data` (**TODO**: Change name)

**3 Run `K-Means Clustering Analysis `**

- Loads `seasonal_precipitation_touse.feather` 

**4 Run `X-Means Clustering Analysis `**

- Loads `seasonal_precipitation_touse.feather` 

------

## Initial Results 

**Note**: Change drought indicator to payout 

| Model               | $D_{\text{avg}} $    | Notes                              |
|---------------------|----------------------|-------------------------------------|
| Traditional Admins (n=34)  | 35.83%                    | Based on administrative boundaries  |
| Non-Optimized K-Means (n=34)      | 35.09%                    | K-means equals number of administrative districts           |
| Optimized K-Means (n=3)     | 22.41%                    | K-means optimized from Silloute Score         |
| Optimized X-Means (n=34)     | 27.14%                    | X-means optimized          |
| Drought Indicator Optimized K-Means (n=33)     | 40.72%                    | K-means optimized from Silloute Score on Drought Indicator    |
| Drought Indicator X-Means (n=34)     |      40.72%               | X-means on Drought Indicator    |


