import numpy as np

# Constants
PI = np.pi


# Step 1: Compute the Illumination Angle
def compute_illumination_angle(altitude):
    # Convert altitude to zenith angle
    zenith_deg = 90.0 - altitude
    # Convert zenith angle to radians
    zenith_rad = zenith_deg * PI / 180.0
    return zenith_rad


# Step 2: Compute the Illumination Direction
def compute_illumination_direction(azimuth):
    # Convert azimuth angle from geographic to mathematical unit
    azimuth_math = 360.0 - azimuth + 90.0
    if azimuth_math >= 360.0:
        azimuth_math = azimuth_math - 360.0
    # Convert the mathematical azimuth angle to radians
    azimuth_rad = azimuth_math * PI / 180.0
    return azimuth_rad


# Step 3: Compute Slope and Aspect for a cell
def compute_slope_aspect(cell_values, cellsize, z_factor):
    a, b, c, d, e, f, g, h, i = cell_values

    # Compute rate of change in x direction
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cellsize)
    # Compute rate of change in y direction
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cellsize)

    # Compute slope in radians
    slope_rad = np.arctan(z_factor * np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    # Determine aspect in radians
    if dz_dx != 0:
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        if aspect_rad < 0:
            aspect_rad = 2 * PI + aspect_rad
    else:
        if dz_dy > 0:
            aspect_rad = PI / 2
        elif dz_dy < 0:
            aspect_rad = 2 * PI - PI / 2
        else:
            aspect_rad = 0  # Set aspect to zero when terrain is flat

    return slope_rad, aspect_rad


# Step 4: Compute Hillshade for a cell
def compute_hillshade(zenith_rad, azimuth_rad, slope_rad, aspect_rad):
    hillshade = 255.0 * ((np.cos(zenith_rad) * np.cos(slope_rad)) +
                         (np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)))
    return max(hillshade, 0)  # If hillshade value is less than 0, return 0


def compute_hillshade_for_grid(elevation_grid, cellsize=1, z_factor=1, altitude=45, azimuth=315):
    # Get the dimensions of the elevation grid
    rows, cols = elevation_grid.shape

    # Initialize the hillshade matrix with zeros
    hillshade_matrix = np.zeros_like(elevation_grid)

    # Compute zenith and azimuth in radians
    zenith_rad = compute_illumination_angle(altitude)
    azimuth_rad = compute_illumination_direction(azimuth)

    # Helper function to get 3x3 window around a cell
    def get_window(r, c):
        # Handle boundary cases by mirroring
        r_start = max(r - 1, 0)
        r_end = min(r + 2, rows)
        c_start = max(c - 1, 0)
        c_end = min(c + 2, cols)

        return elevation_grid[r_start:r_end, c_start:c_end].flatten()

    # Iterate over each cell in the elevation grid
    for r in range(rows):
        for c in range(cols):
            # Get the 3x3 window around the current cell
            cell_values = get_window(r, c)

            # If we don't have a full 3x3 window (i.e., for boundary cells), continue to next cell
            if len(cell_values) != 9:
                continue

            # Compute slope and aspect for the cell
            slope_rad, aspect_rad = compute_slope_aspect(cell_values, cellsize, z_factor)

            # Compute the hillshade value for the cell
            hillshade_value = compute_hillshade(zenith_rad, azimuth_rad, slope_rad, aspect_rad)

            # Store the hillshade value in the output matrix
            hillshade_matrix[r, c] = hillshade_value

    return hillshade_matrix
