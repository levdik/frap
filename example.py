import matplotlib.pyplot as plt

from frap import process_frap_images, process_diffusion
from frap import gaussian_distribution


if __name__ == '__main__':
    time_between_frames = 0.392
    scan_size_um = 425

    # approximate frap images with gaussian distribution and get w(t)
    w, frap_raw_data = process_frap_images(directory_path='test_data', scan_size_um=425)

    # linearize w²(t) and get hydrodynamic diameter
    D, diameter, linearization_data = process_diffusion(w, t=time_between_frames)
    print(f'Hydrodynamic diameter: {round(diameter, 3)} nm')

    # visualization
    # intensity and gaussian approximation
    n_frames_to_plot = 5
    for i in range(n_frames_to_plot):
        x, experimental_data, gaussian_params = frap_raw_data[i]
        plot_color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.plot(x, experimental_data, linewidth=0.2, color=plot_color, label='_nolegend_')
        plt.plot(x, gaussian_distribution(x, *gaussian_params), color=plot_color)
    plt.title('Bleached line intensity approximated with Gaussian')
    plt.xlabel('x, μm')
    plt.ylabel('Intensity')
    plt.legend([f'frame {i + 1}' for i in range(n_frames_to_plot)])
    plt.show()

    # diffusion linearization
    t, w2, linearization_params = linearization_data
    k, b = linearization_params
    plt.plot(t, w2, marker='o', linewidth=0)
    plt.plot(t, t * k + b)
    plt.xlabel('Time, s')
    plt.ylabel('w², μm²')
    plt.show()
