from frap import process_frap_images, process_diffusion


if __name__ == '__main__':
    w = process_frap_images(directory_path='test_data',
                            scan_size_um=425,
                            show_baseline=True,
                            show_curves=3)
    D, diameter = process_diffusion(w,
                                    t=0.392,
                                    show_plot=True)
    print(f'Hydrodynamic diameter: {round(diameter * 1e9, 3)} nm')
