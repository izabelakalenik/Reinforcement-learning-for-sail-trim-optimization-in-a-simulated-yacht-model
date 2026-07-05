import xarray as xr
import numpy as np
from main.config import DEFAULT_WIND_CSV, WIND_RAW_NC


def calculate_wind_speed_direction(u10, v10):
    """
    Calculate wind speed and direction from u10, v10 components.
    
    Args:
        u10: u component (west-east)
        v10: v component (south-north)
    
    Returns:
        speed: wind speed [m/s]
        direction: wind direction [°], 0°=N, 90°=E, 180°=S, 270°=W
    """
    speed = np.sqrt(u10**2 + v10**2)

    # direction: from which side the wind blows
    direction = (np.arctan2(u10, v10) * 180 / np.pi) % 360
    return speed, direction


def convert_netcdf_to_csv(input_path, output_path, latitude: float = None, longitude: float = None):
    print(f"Opening file: {input_path}")
    ds = xr.open_dataset(input_path)
    
    print(f"Structure:\n{ds}\n")
    
    df = ds.to_dataframe().reset_index()

    print("Calculating wind speed and direction")
    df['wind_speed'], df['wind_direction'] = calculate_wind_speed_direction(df['u10'], df['v10'])
    
    if latitude is not None and longitude is not None:
        print(f"Filtering data for: lat={latitude}, lon={longitude}")
        df = df[(df['latitude'] == latitude) & (df['longitude'] == longitude)]
        if len(df) == 0:
            print(f"No data available for these coordinates")
            print(f"Available latitudes: {sorted(ds.latitude.values)}")
            print(f"Available longitudes: {sorted(ds.longitude.values)}")
            return
    
    df = df.sort_values('valid_time').reset_index(drop=True)
    
    columns_to_keep = ['valid_time', 'latitude', 'longitude', 'u10', 'v10', 'wind_speed', 'wind_direction']
    df = df[columns_to_keep]

    print(f"Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df)} rows")
    print(f"\nSample data:\n{df.head()}")
    print(f"\nStatistics:\n{df[['wind_speed', 'wind_direction']].describe()}")
    
    return df


def list_available_locations(input_path):
    ds = xr.open_dataset(input_path)
    print("Available locations:")
    print(f"  Latitudes: {sorted(ds.latitude.values)}")
    print(f"  Longitudes: {sorted(ds.longitude.values)}")


if __name__ == '__main__':
    # # 1: convert all data
    # print("=" * 60)
    # print("All data")
    # print("=" * 60)
    # df_all = convert_netcdf_to_csv(input_path='dataset/era5_dataset.nc', output_path='dataset/wind_data_all.csv')
    
    # 2: only specific location (Gdansk ~ 54.5°N, 18.5°E)
    print("\n" + "=" * 60)
    print("Data for specific location (Gdansk)")
    print("=" * 60)
    list_available_locations(WIND_RAW_NC)
    
    # adjust coordinates to available data
    # close to Gdansk: lat=54.5, lon=18.5 or 18.75
    df_local = convert_netcdf_to_csv(
        input_path=WIND_RAW_NC,
        output_path=DEFAULT_WIND_CSV,
        latitude=54.5,
        longitude=18.5
    )
    
    print("\n" + "=" * 60)
    print("Conversion completed")
    print("=" * 60)
