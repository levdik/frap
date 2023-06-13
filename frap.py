import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

from collections.abc import Iterable
import os
import re

from scipy.constants import k as k_b
from scipy.constants import zero_Celsius


__author__ = "Evgeniy Levdik"
__copyright__ = "Copyright (C) 2023 Evgeniy Levdik"
__license__ = "Public Domain"
__version__ = "1.0"


def gaussian_distribution(x, y0, a, x0, w):
    return y0 + a * np.exp(-2 * ((x - x0) ** 2) / (w ** 2))


def process_diffusion(w, t, linear_points=None, remove_outliers=False,
                      room_temperature_celsius=23., solvant_viscosity=0.9358e-3):
    """Calculates diffusion coefficient and hydrodynamic diameter of particles based on FRAP data.

    Parameters
    ----------
    w : iterable
        Array of widths at half minimum intensity of bleached line
    t : float, iterable
        Either a time per frame or an array of time delays inbetween frames
    linear_points : int, optional
        Number of points from which the slope is determined. If None, all points are considered (None by default)
    remove_outliers : bool, iterable, optional
        List of indices of outliers to be removed from `w`
        If False, no points are removed (False by default)
        If True, outliers are removed automatically (not implemented yet)
    room_temperature_celsius : float, optional
        Room temperature in Celsius, 23 by default
    solvant_viscosity : float, optional
        Solvant viscosity in Pa * s, 0.9358e-3 by default (water at 23 degrees Celsius)

    Returns
    ----------
    float : diffusion coefficient, m^2 / s
    float : hydrodynamic radius, nm
    tuple : linearization data (t, w2, (k, b))
    """

    if isinstance(t, float):
        t = np.arange(len(w)) * t

    if isinstance(remove_outliers, Iterable):
        w = np.delete(w, remove_outliers)
        t = np.delete(t, remove_outliers)
    elif remove_outliers:
        # TODO: remove outliers automatically
        pass

    w2 = np.array(w) ** 2

    if linear_points is None:
        linear_points = len(w)

    linearization_params = np.polyfit(t[:linear_points - 1], w2[:linear_points - 1], 1)
    k, b = linearization_params

    diffusion_coeff_m2s = k * 1e-12 / 8
    room_temperature_kelvin = room_temperature_celsius + zero_Celsius
    diameter_nm = 1e9 * k_b * room_temperature_kelvin / (3 * np.pi * solvant_viscosity * diffusion_coeff_m2s)
    linearization_data = (t, w2, linearization_params)
    return diffusion_coeff_m2s, diameter_nm, linearization_data


def process_frap_images(directory_path, scan_size_um,
                        baseline_correction='linear',
                        frames_without_bleaching=2, crop=(0.3, 0.03)):
    """
    Processes microscopic images of diffusion kinetics of a bleached horizontal line (FRAP)

    Parameters
    ----------
    directory_path : str
        Path to directory with microscopic images
    scan_size_um : float
        Scanning area size, μm
    baseline_correction : str, optional
        Type of baseline correction. 'linear' (by default) and None are possible
    frames_without_bleaching : int, optional
        Number of images taken before bleaching, 2 by default
    crop : iterable, optional
        Relative size of edges to crop, by default 0.3 vertically, 0.03 horizontally

    Returns
    ----------
    list : list of full width at half minimum intensity of bleached line in micrometers
    list : list of tuples (x_um, experimental_data, gaussian_parameters) for each frame
    """

    image_filenames = os.listdir(directory_path)
    number_of_images = len(image_filenames) - frames_without_bleaching
    extension = image_filenames[0].split('.')[-1]
    prefix_end_index = re.search(f'\\d+.{extension}', image_filenames[0]).start()
    image_filename_prefix = directory_path + '/' + image_filenames[0][:prefix_end_index]
    index_length = len(image_filenames[0]) - prefix_end_index - len('.jpg')

    def filename_postfix(index):
        index = str(index)
        return '0' * (index_length - len(index)) + index + '.' + extension

    with Image.open(image_filename_prefix + filename_postfix(0)) as initial_image:
        initial_image = initial_image.convert('L')
        initial_data = np.array(initial_image).astype(float) / 255
    image_size_px = initial_data.shape[0]
    crop_x = int(crop[0] * image_size_px)
    crop_y = int(crop[1] * image_size_px)
    initial_data = initial_data[crop_x:-crop_x, crop_y:-crop_y]
    initial_data = np.mean(initial_data, 1)

    x = np.linspace(0, len(initial_data) * scan_size_um / image_size_px, len(initial_data))

    if baseline_correction is not None:
        baseline_correction = baseline_correction.lower()
        if baseline_correction == 'linear':
            k, b = np.polyfit(x, initial_data, 1)
            baseline = k * x + b
        else:
            raise ValueError(f'No such type of baseline correction: {baseline_correction}')

    w = []
    all_images_data = []
    for i in range(number_of_images):
        with Image.open(image_filename_prefix + filename_postfix(i + frames_without_bleaching)) as image:
            image = image.convert('L')
            data = np.array(image).astype(float) / 255
        data = data[crop_x:-crop_x, crop_y:-crop_y]
        data = np.mean(data, 1)
        if baseline_correction is not None:
            data /= baseline

        guess_params = data[0], min(data) - data[0], x[np.argmin(data)], 0.2 * image_size_px
        params, cov = curve_fit(gaussian_distribution, x, data, p0=guess_params)
        w.append(params[-1])

        all_images_data.append((x, data, params))

    return w, all_images_data
