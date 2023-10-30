import numpy as np
import matplotlib.colors as mcol
import cv2

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

def export_results(visibility, vmin, vmax, color_choice, color_factor, equalize = False):
    def adjust_color(color, color_factor, h_color):
        r, g, b = color

        # Calculate grayscale intensity (average of RGB values)
        intensity = (r + g + b) / 3.0

        # Adjust the coloring factor based on intensity
        adjusted_color_factor = color_factor * (1 - intensity)

        # Apply the adjusted coloring factor to the blue component
        if h_color == 0:
            r += adjusted_color_factor * (1 - r)  # Increase blue but ensure it doesn't exceed 1
            return min(r, 1), g, b  # Ensure doesn't exceed 1
        elif h_color == 1:
            g += adjusted_color_factor * (1 - g)  # Increase blue but ensure it doesn't exceed 1
            return r, min(g,1), b  # Ensure doesn't exceed 1
        elif h_color == 2:
            b += adjusted_color_factor * (1 - b)  # Increase blue but ensure it doesn't exceed 1
            return r, g, min(b,1)  # Ensure doesn't exceed 1


    # Normalize data
    normalized_data = (visibility - vmin) / (vmax - vmin)

    # Original colors
    colors = [(0, 0, 0),(255, 255, 255)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]

    # Adjust colors
    colors_adjusted = [adjust_color(color, color_factor, color_choice) for color in colors_scaled]

    # Create colormap
    custom_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_adjusted, N=256)

    # Apply a colormap from matplotlib (e.g., 'viridis')
    colored_data = custom_cmap(normalized_data)

    # Convert the RGB data to uint8 [0, 255]
    img = (colored_data[:, :, :3] * 255).astype(np.uint8)

    if equalize:
        img = apply_clahe_color(img)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img_bgr


def apply_clahe_color(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Split the LAB image into L, A and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Define the CLAHE algorithm
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to L channel
    clahe_img = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge([clahe_img, a_channel, b_channel])

    # Convert back to RGB color space
    final_img = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)

    return final_img