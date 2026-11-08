#!/usr/bin/env python3

import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
import pyvista as pv

import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

from scipy.spatial import cKDTree

def sparse_streamlines(
    mesh,
    xmin, xmax,
    ymin, ymax,
    z_plane,
    nx=100,
    ny=100,
    min_dist_fraction=0.02,
):
    """
    Generate streamlines from a uniform XY seed grid,
    rejecting streamlines that come too close to previously
    accepted ones.
    """

    domain_size = max(xmax - xmin, ymax - ymin)
    min_dist = min_dist_fraction * domain_size

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    accepted_pts = []
    accepted_lines = []

    for y in ys:
        for x in xs:

            seed = np.array([[x, y, z_plane]])

            src = pv.PolyData(seed)

            sl = mesh.streamlines_from_source(
                src,
                vectors="velocity",
                surface_streamlines=True,
                max_length=5.0,
                integration_direction="both",
                initial_step_length=0.01,
            )

            if sl.n_points < 5:
                continue

            pts = sl.points

            keep = True

            if accepted_pts:

                tree = cKDTree(
                    np.vstack(accepted_pts)
                )

                dmin = tree.query(
                    pts,
                    k=1
                )[0].min()

                if dmin < min_dist:
                    keep = False

            if keep:
                accepted_lines.append(sl)
                accepted_pts.append(pts)

    if not accepted_lines:
        return None

    result = accepted_lines[0]

    for sl in accepted_lines[1:]:
        result = result.merge(sl)

    print(
        f"Accepted {len(accepted_lines)} streamlines "
        f"from {nx*ny} candidate seeds"
    )

    return result

def fit_camera(plotter, margin=0.05):
    """
    Fit visible geometry into current view.

    margin=0.05 -> 5% border
    """

    plotter.reset_camera()
    plotter.reset_camera_clipping_range()

    if plotter.camera.parallel_projection:
        plotter.camera.parallel_scale *= (1.0 + margin)

def fit_mesh(plotter, mesh, margin=0.95):
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    plotter.camera.focal_point = (cx, cy, cz)

    if plotter.camera_position == "xz":
        extent = max(dx, dz)
    elif plotter.camera_position == "xy":
        extent = max(dx, dy)
    else:
        extent = max(dy, dz)

    plotter.camera.parallel_scale = 0.5 * extent / margin

def latest_vtk():
    vtk_files = sorted(
        glob.glob("postProcessing/midXYPlane/*/*.vtk"),
        key=lambda x: Path(x).parent.name
    )

    if not vtk_files:
        raise RuntimeError(
            "No VTK files found under postProcessing/midXYPlane/*/*.vtk"
        )

    return vtk_files[-1]


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("  plotStreamlines.py U")
        sys.exit(1)

    field_name = sys.argv[1]

    vtk_file = latest_vtk()

    print(f"Reading: {vtk_file}")

    mesh = pv.read(vtk_file)

    if field_name not in mesh.point_data:
        print(f"Available fields:")
        for f in mesh.point_data.keys():
            print("   ", f)
        raise RuntimeError(f"Field '{field_name}' not found")

    vel = mesh.point_data[field_name]

    if vel.shape[1] != 3:
        raise RuntimeError(
            f"'{field_name}' is not a vector field"
        )

    mesh["velocity"] = vel

    bounds = mesh.bounds

    xmin, xmax = bounds[0], bounds[1]
    ymin, ymax = bounds[2], bounds[3]
    zmin, zmax = bounds[4], bounds[5]

    nx = 10
    ny = 30
    nz = 1

    streamlines = sparse_streamlines( mesh, xmin, xmax, ymin, ymax, z_plane=(zmin + zmax)/2,
        nx=nx,   ny=ny,   min_dist_fraction=0.01   )

    p = pv.Plotter(off_screen=True)
    p.view_xy()
    p.camera.parallel_projection = True
    fit_mesh(p, mesh, margin=0.95)

    p.add_mesh(
        mesh,
        scalars=np.linalg.norm(vel, axis=1),
        cmap="viridis",
        opacity=0.4,
        show_scalar_bar=True,
        scalar_bar_args={
                "title": field_name,
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.1,
                "height": 0.8,
                "width": 0.08,
        }        
    )

    p.add_mesh(  streamlines.tube(radius=0.002*(xmax-xmin)),  color="black"  )

    p.show_axes()
    #p.show()

    safe_field_name = field_name.replace(".", "")
    output_file = f"{safe_field_name}.png"
    p.screenshot(output_file)
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()