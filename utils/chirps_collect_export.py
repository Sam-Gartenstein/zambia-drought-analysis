import time
import os
import numpy as np
import pandas as pd
import ee  # Google Earth Engine API
import geemap  # For converting EE images to NumPy


def collect_and_export_chirps(start_date, end_date, output_folder, aoi, data_type="PENTAD", max_retries=5, wait_time=10):
    """
    Exports CHIRPS precipitation data (Daily or Pentad) to CSV files for a given date range and region.
    Implements a restart mechanism to handle timeouts or failures.

    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - output_folder (str): Path to the folder where CSV files will be saved.
    - aoi (ee.Geometry): The area of interest (AOI) for filtering the CHIRPS data.
    - data_type (str): "DAILY" or "PENTAD" to specify which dataset to use.
    - max_retries (int): Maximum number of retry attempts before exiting (default: 5).
    - wait_time (int): Time (seconds) to wait before retrying after a failure (default: 10).

    Returns:
    - None (Exports files to the specified output folder)
    """
    start_time = time.time()

    # Ensure Earth Engine is initialized
    for attempt in range(max_retries):
        try:
            ee.Initialize()
            break  # Exit retry loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Earth Engine not initialized: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All retries failed. Exiting.")
                return

    # Validate data_type input
    data_type = data_type.upper()
    if data_type not in ["DAILY", "PENTAD"]:
        print("Invalid data_type! Choose 'DAILY' or 'PENTAD'.")
        return

    dataset = "UCSB-CHG/CHIRPS/DAILY" if data_type == "DAILY" else "UCSB-CHG/CHIRPS/PENTAD"

    # Define the CHIRPS image collection and filter it for the given date range
    CHIRPSCollection = (
        ee.ImageCollection(dataset)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    def makeBandLabel(img):
        year = ee.Number(img.get('year')).int().format()
        month = ee.Number(img.get('month')).int().format('%02d')

        if data_type == "PENTAD":
            pentad = ee.Algorithms.If(img.get('pentad'), ee.Number(img.get('pentad')).int().format('%02d'), 'XX')
            label = ee.String('y').cat(year).cat('m').cat(month).cat('p').cat(pentad).cat('_Precipitation')
        else:  # DAILY
            day = ee.Algorithms.If(img.get('day'), ee.Number(img.get('day')).int().format('%02d'), 'XX')
            label = ee.String('y').cat(year).cat('m').cat(month).cat('d').cat(day).cat('_Precipitation')

        return img.rename([label])

    chirpsExportImage = CHIRPSCollection.map(makeBandLabel).toBands()

    for attempt in range(max_retries):
        try:
            bandNames = chirpsExportImage.bandNames().getInfo()
            break  # Exit retry loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Error retrieving band names: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All retries failed. Exiting.")
                return

    os.makedirs(output_folder, exist_ok=True)

    for b in bandNames:
        for attempt in range(max_retries):
            try:
                band_start_time = time.time()
                
                precipImage = chirpsExportImage.select(b)
                chirps_arr = geemap.ee_to_numpy(precipImage, region=aoi)[:,:,0]
                
                if chirps_arr is None or chirps_arr.size == 0:
                    print(f"No data for band {b}. Skipping export.")
                    break
                
                local_file_path = os.path.join(output_folder, f"{b}.csv")
                np.savetxt(local_file_path, chirps_arr, delimiter=",")
                
                if not os.path.exists(local_file_path):
                    print(f"Failed to create {local_file_path}. Skipping export.")
                else:
                    print(f"File {local_file_path} created successfully.")
                
                band_end_time = time.time()
                print(f"Exported {b} in {round(band_end_time - band_start_time, 2)} seconds.")
                break  # Exit retry loop if successful
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} - Error processing band {b}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Skipping band {b} after {max_retries} failed attempts.")

    lonlatimage = ee.Image.pixelLonLat().reproject('EPSG:4326', None, 5565.97)
    for coord in ['longitude', 'latitude']:
        for attempt in range(max_retries):
            try:
                lonlat_arr = geemap.ee_to_numpy(lonlatimage.select(coord), region=aoi)[:,:,0]
                local_file_path = os.path.join(output_folder, f"AOI_{coord}.csv")
                np.savetxt(local_file_path, lonlat_arr, delimiter=",")
                print(f"Exported {coord} to {local_file_path}")
                break  # Exit retry loop if successful
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} - Error exporting {coord}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Skipping {coord} after {max_retries} failed attempts.")

    end_time = time.time()
    print(f"Total time taken: {round(end_time - start_time, 2)} seconds.")
    
    
    
################################################################################################################

# Fecthing Administrative 2 Rainfall Data

################################################################################################################


def fetch_precipitation_data_admin2(admin2_name):
    """
    Fetch monthly precipitation data for a given Admin Level 2 region in Zambia for 2000 to 2023,
    and return a restructured DataFrame with the following columns:
    - year, month, date, region, admin2_name, precipitation.
    """
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
    startyear, endyear = 1981, 2024
    startdate, enddate = ee.Date.fromYMD(startyear, 1, 1), ee.Date.fromYMD(endyear, 12, 31)

    # Define the region from Zambia (Admin Level 2)
    region = ee.FeatureCollection('FAO/GAUL/2015/level2') \
              .filter(ee.Filter.eq('ADM0_NAME', 'Zambia')) \
              .filter(ee.Filter.eq('ADM2_NAME', admin2_name)).first()

    def MonthlySum(year):
        """
        Sum precipitation data for each month of a given year.
        """
        def monthSum(month):
            # Filter the CHIRPS dataset for the specific month and year
            monthly_sum = chirps.filterDate(startdate, enddate) \
                                .filter(ee.Filter.calendarRange(year, year, 'year')) \
                                .filter(ee.Filter.calendarRange(month, month, 'month')) \
                                .sum() \
                                .reduceRegion(ee.Reducer.mean(), geometry=region.geometry(), scale=5000, maxPixels=1e8)

            # Return the precipitation data and additional info
            return ee.Feature(None, {
                'year': year,
                'month': month,
                'date': ee.Date.fromYMD(year, month, 1).format(),
                'region': 'Zambia',
                'admin2_name': admin2_name,
                'precipitation': monthly_sum.get('precipitation')
            })
        return ee.List.sequence(1, 12).map(monthSum)

    years = ee.List.sequence(startyear, endyear)
    monthlyPrecip = years.map(MonthlySum).flatten()
    monthlyPrecipCollection = ee.FeatureCollection(monthlyPrecip)

    properties_list = monthlyPrecipCollection.getInfo()

    if properties_list['features']:
        data = [feature['properties'] for feature in properties_list['features']]
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame()

    def restructure_dataframe(df):
        """
        Restructure the DataFrame to match the desired column order:
        ['year', 'month', 'date', 'region', 'admin2_name', 'precipitation']
        """
        desired_order = ['year', 'month', 'date', 'region', 'admin2_name', 'precipitation']
        return df[desired_order]

    return restructure_dataframe(df)


def fetch_and_combine_precipitation_data(admin2_names):
    """
    Fetch and combine monthly precipitation data for all Admin Level 2 regions in Zambia
    from 2000 to 2023, and return a single combined DataFrame.

    Parameters:
    admin2_names (list): List of Admin Level 2 region names.

    Returns:
    pd.DataFrame: Combined DataFrame containing precipitation data for all regions.
    """
    all_precip_data = []

    for admin2_name in admin2_names:
        print(f"Fetching precipitation data for {admin2_name}...")
        df = fetch_precipitation_data_admin2(admin2_name)
        all_precip_data.append(df)

    combined_df = pd.concat(all_precip_data, ignore_index=True)
    return combined_df


