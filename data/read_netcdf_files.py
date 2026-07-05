import netCDF4

f = netCDF4.Dataset('./data/datasets/era5_dataset.nc')

print("=== Variables ===")
for name, var in f.variables.items():
    units = getattr(var, 'units', 'no units')
    long_name = getattr(var, 'long_name', '')
    print(f"  {name:20s} shape={str(var.shape):20s} units={units}  {long_name}")

print()

# access the coordinate variables
# use [:] to load the actual data into a NumPy array
lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]

print(f"Latitude range: {lats.min()} to {lats.max()}  [{getattr(f.variables['latitude'], 'units', '?')}]")
print(f"Longitude range: {lons.min()} to {lons.max()}  [{getattr(f.variables['longitude'], 'units', '?')}]")

print("\nFirst 5 Latitude values:", lats[:5])
print("First 5 Longitude values:", lons[:5])

# check the total number of points (shape)
print(f"\nShape of latitude array: {lats.shape}")
print(f"Shape of longitude array: {lons.shape}")