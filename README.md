# Zambia Drought Analysis

---

## Payout Balance Metric

To evaluate how well the clustering minimizes **basis risk**, we first compute the **absolute deviation** from a 50% payout rate for each cluster-year pair:

$$
D_t^k = |P_t^k - 50|
$$

Where:

- $P_t^k$ = Percentage of pixels receiving a payout in **cluster** $k$ during **year** $t$  
- $D_t^k$ = Absolute deviation from the 50% balance point — **higher values indicate greater basis risk**


### Payout Balance Index (PBI)

We then transform this deviation into a **Payout Balance Index (PBI)** scaled between 0 and 100:

$$
PBI_{t,k} = \left(1 - \frac{D_t^k}{50} \right) \times 100
$$

- A value of **100** means perfect balance: all or none of the farmers in the cluster received payouts  
- A value of **0** indicates a 50/50 split — the **worst-case scenario** for basis risk


### Overall Clustering Quality

To assess clustering quality across the entire dataset, we compute the **average PBI**:

$$
PBI_{\text{avg}} = \frac{1}{N} \sum_{t,k} PBI_{t,k}
$$

Where $N$ is the total number of (year × cluster) combinations.


### Basis Risk Interpretation

- A **PBI close to 100** reflects **low basis risk**: clusters are internally consistent — farmers in the same area experience similar insurance outcomes  
- A **PBI near 0** reflects **high basis risk**: payout decisions vary significantly within clusters, indicating poor targeting

This index provides a **clear, interpretable, and policy-relevant** measure of how well your spatial clustering aligns with actual drought impact.


------

##  Results 

### Primary Analysis

The primary analysis includes all available data. Optimization methods, where applied, use the Silhouette Score to determine the optimal number of clusters.

| **Zoning Method**                               | **$PBI_{\text{avg}}$** | **Notes**                                                                 |
|------------------------------------|--------------------|-----------------------------------------------------------------------|
| Traditional Admins (n=34)          | 91.25%              | Based on existing administrative district boundaries.                 |
| Non-Optimized K-Means (n=34)       | 90.52%             | K-Means clustering with number of clusters fixed to match admin units.|
| Optimized K-Means (n=3)            | 80.57%             | K-Means clustering with number of clusters selected via Silhouette Score. |
| Payout Non-Optimized K-Means (n=34)| 90.16%             | Pixels clustered by K-Means; evaluated using the insurance payout formula. |
| Payout Optimized K-Means (n=3)     | 81.16%             | Pixels clustered by optimized K-Means; evaluated using the insurance payout formula. |

### Filtered Data 

This sub-analysis filters out clusters in which fewer than 50% of pixels experienced a drought, focusing only on clusters with substantial drought exposure.


| **Zoning Method**                               | **$PBI_{\text{avg}}$** | **Notes**                                                                 |
|------------------------------------|--------------------|-----------------------------------------------------------------------|
| Traditional Admins (n=34)         | 79.36%             | Based on existing administrative district boundaries.                 |
| Non-Optimized K-Means (n=34)       | 78.12%             | K-Means clustering with number of clusters fixed to match admin units.|
| Optimized K-Means (n=3)            | 65.75%             | K-Means clustering with number of clusters selected via Silhouette Score. |
| Payout Non-Optimized K-Means (n=34)| 77.77%             | Pixels clustered by K-Means; evaluated using the insurance payout formula. |
| Payout Optimized K-Means (n=3)     | 61.88%             | Pixels clustered by optimized K-Means; evaluated using the insurance payout formula. |


---

## Steps

**TODO**: Finish this 

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

**3 Run ` Clustering Analysis `**

- Loads 
  
  - `seasonal_precipitation_touse.feather`
  - `zambia_admin2_boundaries.geojson`

**4 Run `K-Means Clustering Analysis `**

- Loads `seasonal_precipitation_touse.feather` 

**5 Run `Payout Clustering Analysis `**

- Loads `seasonal_precipitation_touse.feather` 



---- 

## TODO

1. Calculate payout percentages and run k-means on that (setting equal to admin zones and optimized)
2. Come up with a more intuitive way of calculating payout (a likelihood where we say the probability of getting a payout is X%)
3. Attempt g-means and hope it works

### For next week
-Talk about which ones to report, and which ones are the most intepretable (what to keep)

**Notes 4-15-25**

- We can go to about 10th as many places, only get a zonal basis risk increase by 10% (aka going from 90% to 80%)
- Helps determination of trade off of basis risk and decreasing the number of data driven clusters 
- None of the otpimization based methods give us better results than the admin boundaries -> goes counter to Bemani found in Kenya where they found improvements -> some of the boundaries might already fall along natural features with differecnes in weather patterns -> objectively it's quite good
- Worth mentioning adding in NDVI or crop cuts assuming there is already modelled data
-To include my assingment/draft
-Show table, maps side by side, how different are the k-means clusters vs the admin boundaries visiually
-Practical is that optimization methods like x-means g-means don't work for this
-Social Science ideas -> why we are seeing the results we do (aka why admin boundaries work so well) and presenting a statistic that has a economically meaningful interpretaiton
-Address the type of basis risk I am addressing, and show the time series stuff I show (aka the heat maps)
-Social science look at this from the perspective of a designer and motivating is literature from Dan's team, Cater's team

**One More Thing**

- Look at the 50% above cut off 

---


