#!/usr/bin/env python3

"""
Requirements
============

This script reads VTK files produced by an OpenFOAM sampling
function object (for example midXZPlane or midXYPlane),
extracts an iso-contour of a specified field and plots the
contours from all available times onto a single figure.

The generated figure contains:

- contour profiles for all times
- simulation time legend
- x and z coordinates
- automatic PNG output

The script is intended for headless environments such as:

- GitHub Codespaces
- HPC clusters
- remote Linux servers

No graphical display is required.

Python packages:

    pip install pyvista matplotlib numpy

Usage
=====

Plot alpha.solids=0.5 contours:

    python plotContours.py --field alpha.solids --iso 0.5 --dir postProcessing/midXZPlane

Plot alpha.air=0.9 contours:

    python plotContours.py --field alpha.air --iso 0.9 --dir postProcessing/midXZPlane

Plot alpha.particles=0.95 contours with particle-velocity vectors:

    python plotContours.py --field particles --iso 0.95 --dir postProcessing/midXYPlane

Output:

    alpha.solids_contours.png

Example
=======

    python plotContours.py \
        --field alpha.solids \
        --iso 0.5

"""

import numpy as np

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")     # Required for headless systems

import matplotlib.pyplot as plt
import pyvista as pv

MIN_PLOTTED_VELOCITY = 1e-5


def time_key(path):
    """
    Sort VTK files by simulation time.
    """

    try:
        return float(path.parent.name)
    except ValueError:
        return -1.0


def plot_vectors(ax, contour, mesh, field_name, color):

    mesh_bounds = mesh.bounds 
    ax.set_xlim(mesh_bounds[0], mesh_bounds[1])
    ax.set_ylim(mesh_bounds[2], mesh_bounds[3])
    ax.set_aspect('equal', adjustable='box')

    sampled_contour = contour.sample(mesh)
    if field_name in sampled_contour.point_data:
        points = sampled_contour.points
        vectors = np.asarray(sampled_contour.point_data[field_name])
    elif field_name in sampled_contour.cell_data:
        points = sampled_contour.cell_centers().points
        vectors = np.asarray(sampled_contour.cell_data[field_name])
    else:
        raise RuntimeError(
            f"Vector field '{field_name}' could not be sampled onto the alpha contour."
        )

    if vectors.ndim != 2 or vectors.shape[1] < 2 or len(points) != len(vectors):
        raise RuntimeError(f"Vector field '{field_name}' has an invalid shape.")

    visible_speed = np.linalg.norm(vectors[:, :2], axis=1)
    visible = visible_speed >= MIN_PLOTTED_VELOCITY
    points = points[visible]
    vectors = vectors[visible]
    if not len(points):
        return

    stride = max(1, int(np.ceil(len(points) / 400)))
    points = points[::stride]
    vectors = vectors[::stride]
    ax.quiver(points[:, 0],points[:, 1],vectors[:, 0],vectors[:, 1],
        color=color,  alpha=0.6,  angles="xy",  scale_units="xy" )


def max_velocity_magnitude(mesh, field_name):
    if field_name in mesh.point_data:
        vectors = np.asarray(mesh.point_data[field_name])
    elif field_name in mesh.cell_data:
        vectors = np.asarray(mesh.cell_data[field_name])
    else:
        raise RuntimeError(f"Vector field '{field_name}' has no point or cell data.")

    if vectors.ndim != 2 or vectors.shape[1] < 2 or not len(vectors):
        raise RuntimeError(f"Vector field '{field_name}' has an invalid shape.")

    magnitudes = np.linalg.norm(vectors, axis=1)
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    if not len(finite_magnitudes):
        raise RuntimeError(f"Vector field '{field_name}' contains no finite values.")
    return float(finite_magnitudes.max())


def plot_max_velocity_reference(ax, magnitude):
    ax.annotate(
        "",
        xy=(1.04, 0.88),
        xytext=(1.04, 0.76),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.5},
        annotation_clip=False,
    )
    ax.text(
        1.08,
        0.82,
        f"max |U| = {magnitude:.3g} m/s",
        transform=ax.transAxes,
        ha="left",
        va="center",
        clip_on=False,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--field",
        required=True,
        help="Field used for contour extraction."
    )

    parser.add_argument(
        "--iso",
        type=float,
        default=0.5,
        help="Iso-value."
    )

    parser.add_argument(
        "--dir",
        default="postProcessing/midXZPlane",
        help="Directory containing VTK files."
    )

    args = parser.parse_args()

    vtk_files = sorted(
        Path(args.dir).rglob("*.vtk"),
        key=time_key
    )

    if not vtk_files:
        raise RuntimeError(
            f"No VTK files found in '{args.dir}'."
        )

    fig, ax = plt.subplots(figsize=(12, 8))

    n_contours = 0
    maximum_velocity = None

    colors = plt.cm.viridis( np.linspace(0, 1, len(vtk_files)) )

    for color, vtk_file in zip(colors, vtk_files):

        time_name = vtk_file.parent.name

        mesh = pv.read(vtk_file)

        phase_alpha = f"alpha.{args.field}"
        phase_velocity = f"U.{args.field}"
        is_phase = (
            phase_alpha in mesh.array_names
            and phase_velocity in mesh.array_names
        )
        if is_phase:
            mesh_maximum_velocity = max_velocity_magnitude(mesh, phase_velocity)
            maximum_velocity = (
                mesh_maximum_velocity
                if maximum_velocity is None
                else max(maximum_velocity, mesh_maximum_velocity)
            )
        contour_field = phase_alpha if is_phase else args.field

        if contour_field not in mesh.array_names:
            print(f"WARNING: '{contour_field}' not found in {vtk_file}")
            continue

        contour = mesh.contour(isosurfaces=[args.iso], scalars=contour_field)

        for cell_id in range(contour.n_cells):
            cell = contour.get_cell(cell_id)
            pts = cell.points

            ax.plot(
                pts[:,0],
                pts[:,1],
                linewidth=1.0,
                color=color,
                label=time_name if cell_id == 0 else None
            )

        if is_phase:
            plot_vectors(ax, contour, mesh, phase_velocity, color)

        n_contours += 1

    if n_contours == 0:
        raise RuntimeError(
            "No contours could be extracted."
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title(
        f"{contour_field} = {args.iso}"
    )

    ax.grid(True)

    legend_anchor = (1.02, 0.65) if maximum_velocity is not None else (1.02, 1.0)
    ax.legend(
        title="Time",
        bbox_to_anchor=legend_anchor,
        loc="upper left"
    )

    if maximum_velocity is not None:
        plot_max_velocity_reference(ax, maximum_velocity)

    plt.tight_layout()

    output_file = (
        f"{args.field.replace('.', '_')}_contours.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()