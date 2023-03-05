import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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


def process_diffusion(w, t, linear_points=None, remove_outliers=False, show_plot=False,
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
    show_plot : bool, optional
        If True, shows plot of :math:`w^2(t)`, False by default
    room_temperature_celsius : float, optional
        Room temperature in Celsius, 23 by default
    solvant_viscosity : float, optional
        Solvant viscosity in Pa * s, 0.9358e-3 by default (water at 23 degrees Celsius)

    Returns
    ----------
    float : diffusion coefficient, m^2 / s
    float : hydrodynamic radius, nm
    """

    if isinstance(t, float):
        t = np.arange(len(w)) * t

    w2 = np.array(w) ** 2
    if isinstance(remove_outliers, Iterable):
        w = np.delete(w, remove_outliers)
        t = np.delete(t, remove_outliers)
    elif remove_outliers:
        # TODO: remove outliers automatically
        pass

    if linear_points is None:
        linear_points = len(w)

    k, b = np.polyfit(t[:linear_points - 1], w2[:linear_points - 1], 1)

    if show_plot:
        plt.plot(t, w2, marker='o', linewidth=0)
        plt.plot(t, t * k + b)
        plt.xlabel('Time, s')
        plt.ylabel('w², μm²')
        plt.grid()
        plt.show()

    diffusion_coeff = k * 1e-12 / 8  # m^2 / s
    room_temperature_kelvin = room_temperature_celsius + zero_Celsius
    diameter = k_b * room_temperature_kelvin / (3 * np.pi * solvant_viscosity * diffusion_coeff)
    return diffusion_coeff, diameter


def process_frap_images(directory_path, scan_size_um,
                        show_curves=0, show_baseline=False,
                        frames_without_bleaching=2, crop=(0.3, 0.03)):
    """
    Processes microscopic images of diffusion kinetics of a bleached horizontal line (FRAP)

    Parameters
    ----------
    directory_path : str
        Path to directory with microscopic images
    scan_size_um : float
        Scanning area size, μm
    show_curves : int, optional
        Number of first intensity distributions to show, 0 by default
    show_baseline : bool, optional
        If True, plotted baseline is shown, False by default
    frames_without_bleaching : int, optional
        Number of images taken before bleaching, 2 by default
    crop : iterable, optional
        Relative size edges to crop, by default 0.3 vertically, 0.03 horizontally

    Returns
    ----------
    list : list of full width at half minimum intensity of bleached line
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

    k, b = np.polyfit(x, initial_data, 1)
    baseline = k * x + b

    if show_baseline:
        plt.plot(x, initial_data)
        plt.plot(x, baseline)
        plt.xlabel('x, μm')
        plt.ylabel('Intensity')
        plt.show()

    w = []
    for i in range(number_of_images):
        with Image.open(image_filename_prefix + filename_postfix(i + frames_without_bleaching)) as image:
            image = image.convert('L')
            data = np.array(image).astype(float) / 255
        data = data[crop_x:-crop_x, crop_y:-crop_y]
        data = np.mean(data, 1)
        data /= baseline

        def gaussian_distribution(x, y0, a, x0, w):
            return y0 + a * np.exp(-2 * ((x - x0) ** 2) / (w ** 2))
        guess_params = data[0], min(data) - data[0], x[np.argmin(data)], 0.2 * image_size_px
        params, cov = curve_fit(gaussian_distribution, x, data, p0=guess_params)
        standard_deviation = np.sqrt(np.diag(cov))
        # TODO: reject outliers with high standard_deviation
        w.append(params[-1])

        if i < show_curves:
            plot_color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(x, data, linewidth=0.2, color=plot_color, label='_nolegend_')
            plt.plot(x, gaussian_distribution(x, *params), color=plot_color)

    if show_curves > 0:
        plt.xlabel('x, μm')
        plt.ylabel('Intensity')
        plt.show()
    return w
