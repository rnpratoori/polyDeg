import os
import netCDF4
import numpy as np
import pyvista as pv
import vtk
from tqdm import tqdm

def create_lookup_table(color='red'):
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(0, 1)
    for i in range(256):
        x = i / 255.0
        if color == 'red':
            r, g, b = 0.5 + 0.5 * x, 0.0, 0.0
        elif color == 'green':
            r, g, b = 0.0, 0.5 + 0.5 * x, 0.0
        elif color == 'blue':
            r, g, b = 0.0, 0.0, 0.5 + 0.5 * x
        else:
            r = g = b = x
        a = x
        lut.SetTableValue(i, r, g, b, a)
    lut.Build()
    return lut

def create_custom_colormap(color='red'):
    if color == 'red':
        base_color = [1, 0, 0]
    elif color == 'green':
        base_color = [0, 1, 0]
    elif color == 'blue':
        base_color = [0, 0, 1]
    else:
        base_color = [1, 1, 1]

    def custom_cmap(x):
        rgba = np.zeros((len(x), 4))
        rgba[:, :3] = np.array(base_color)
        rgba[:, 3] = x
        return rgba

    return custom_cmap

def main(exodus_filename, output_gif_filename, output_img_filename):
    ds = netCDF4.Dataset(exodus_filename)
    times = ds.variables['time_whole'][:]
    nt = len(times)
    X = np.ma.getdata(ds.variables['coordx'][:])
    Y = np.ma.getdata(ds.variables['coordy'][:])
    Z = np.zeros_like(X)
    points = np.vstack([X, Y, Z]).T
    elem_node = np.ma.getdata(ds.variables['connect1'][:]) - 1

    base_mesh = pv.UnstructuredGrid({vtk.VTK_QUAD: elem_node}, points)
    c1_all = ds.variables['vals_nod_var1'][:]
    c2_all = ds.variables['vals_nod_var2'][:]
    c3_all = 1 - c1_all - c2_all
    ds.close()

    mesh_c1 = base_mesh.copy(deep=True)
    mesh_c2 = base_mesh.copy(deep=True)
    mesh_c3 = base_mesh.copy(deep=True)
    mesh_c1.point_data['c1'] = c1_all[0]
    mesh_c2.point_data['c2'] = c2_all[0]
    mesh_c3.point_data['c3'] = c3_all[0]

    cmap_red = create_custom_colormap('red')
    cmap_green = create_custom_colormap('green')
    cmap_blue = create_custom_colormap('blue')

    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(output_gif_filename)
    plotter.add_mesh(mesh_c1, scalars='c1', clim=(0, 1),
                     show_scalar_bar=False, opacity=0.7,
                     cmap=cmap_red, render=False)
    plotter.add_mesh(mesh_c2, scalars='c2', clim=(0, 1),
                     show_scalar_bar=False, opacity=0.7,
                     cmap=cmap_green, render=False)
    plotter.add_mesh(mesh_c3, scalars='c3', clim=(0, 1),
                     show_scalar_bar=False, opacity=0.7,
                     cmap=cmap_blue, render=False)
    plotter.view_xy()

    for i in tqdm(range(nt), desc="Generating frames"):
        mesh_c1.point_data['c1'] = c1_all[i]
        mesh_c2.point_data['c2'] = c2_all[i]
        mesh_c3.point_data['c3'] = c3_all[i]
        plotter.write_frame()
        
        # Save the final timestep as PNG
        if i == nt-1:
            plotter.screenshot(output_img_filename)

    plotter.close()

if __name__ == "__main__":
    input_dir = "results/output_dump_3p"
    for exodus_file in sorted(os.listdir(input_dir)):
        if exodus_file.endswith('.e'):
            exodus_path = os.path.join(input_dir, exodus_file)
            output_gif = os.path.join(input_dir, exodus_file.replace('.e', '.gif'))
            output_png = os.path.join(input_dir, exodus_file.replace('.e', '_last.png'))
            print(f"Processing {exodus_file}...")
            main(exodus_path, output_gif, output_png)
            print(f"Created {output_gif} and {output_png}")
