import time
import os
import numpy as np
import ee  # Google Earth Engine API
import geemap  # For converting EE images to NumPy


def collect_and_export_chirps(start_date, end_date, output_folder, aoi, data_type="PENTAD"):
    """
    Exports CHIRPS precipitation data (Daily or Pentad) to CSV files for a given date range and region.

    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - output_folder (str): Path to the folder where CSV files will be saved.
    - aoi (ee.Geometry): The area of interest (AOI) for filtering the CHIRPS data.
    - data_type (str): "DAILY" or "PENTAD" to specify which dataset to use.

    Returns:
    - None (Exports files to the specified output folder)
    """
    # Start the timer for the entire process
    start_time = time.time()

    # Ensure Earth Engine is initialized
    try:
        ee.Initialize()
    except Exception as e:
        print("Earth Engine not initialized. Run `ee.Authenticate()` first.")
        return

    # Validate data_type input
    data_type = data_type.upper()
    if data_type not in ["DAILY", "PENTAD"]:
        print("Invalid data_type! Choose 'DAILY' or 'PENTAD'.")
        return

    # Select the appropriate CHIRPS dataset
    dataset = "UCSB-CHG/CHIRPS/DAILY" if data_type == "DAILY" else "UCSB-CHG/CHIRPS/PENTAD"

    # Define the CHIRPS image collection and filter it for the given date range
    CHIRPSCollection = (
        ee.ImageCollection(dataset)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    # Convert each image in the collection to a band with a custom label
    def makeBandLabel(img):
        year = ee.Number(img.get('year')).int().format()
        month = ee.Number(img.get('month')).int().format('%02d')

        # Safe check to prevent null values
        if data_type == "PENTAD":
            pentad = ee.Algorithms.If(img.get('pentad'), ee.Number(img.get('pentad')).int().format('%02d'), 'XX')
            label = ee.String('y').cat(year).cat('m').cat(month).cat('p').cat(pentad).cat('_Precipitation')
        else:  # DAILY
            day = ee.Algorithms.If(img.get('day'), ee.Number(img.get('day')).int().format('%02d'), 'XX')
            label = ee.String('y').cat(year).cat('m').cat(month).cat('d').cat(day).cat('_Precipitation')

        return img.rename([label])

    # Apply the band labeling and combine into a single image
    chirpsExportImage = CHIRPSCollection.map(makeBandLabel).toBands()

    # Get the band names
    try:
        bandNames = chirpsExportImage.bandNames().getInfo()
    except Exception as e:
        print("Error retrieving band names. The dataset may be empty for the given date range.")
        return

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through the bands, convert to numpy, and save directly in the output folder
    for b in bandNames:
        # Start the timer for each band export
        band_start_time = time.time()

        # Convert the precipitation image to a numpy array
        try:
            precipImage = chirpsExportImage.select(b)
            chirps_arr = geemap.ee_to_numpy(precipImage, region=aoi)[:,:,0]
        except Exception as e:
            print(f"Error processing band {b}. Skipping.")
            continue

        if chirps_arr is None or chirps_arr.size == 0:
            print(f"No data for band {b}. Skipping export.")
            continue

        # Save the numpy array as a CSV in the output folder
        local_file_path = os.path.join(output_folder, f"{b}.csv")
        np.savetxt(local_file_path, chirps_arr, delimiter=",")

        # Check if the file was created successfully
        if not os.path.exists(local_file_path):
            print(f"Failed to create {local_file_path}. Skipping export.")
            continue
        else:
            print(f"File {local_file_path} created successfully.")

        # Print the time it took for the current band
        band_end_time = time.time()
        print(f"Exported {b} in {round(band_end_time - band_start_time, 2)} seconds.")

    # Export latitude and longitude grids
    lonlatimage = ee.Image.pixelLonLat().reproject('EPSG:4326', None, 5565.97)
    for coord in ['longitude', 'latitude']:
        try:
            lonlat_arr = geemap.ee_to_numpy(lonlatimage.select(coord), region=aoi)[:,:,0]
            local_file_path = os.path.join(output_folder, f"AOI_{coord}.csv")
            np.savetxt(local_file_path, lonlat_arr, delimiter=",")
            print(f"Exported {coord} to {local_file_path}")
        except Exception as e:
            print(f"Error exporting {coord}. Skipping.")

    # End the timer for the entire process and print the time taken
    end_time = time.time()
    print(f"Total time taken: {round(end_time - start_time, 2)} seconds.")

