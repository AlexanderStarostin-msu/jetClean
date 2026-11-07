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


def time_key(path):
    """
    Sort VTK files by simulation time.
    """

    try:
        return float(path.parent.name)
    except ValueError:
        return -1.0


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

    colors = plt.cm.viridis( np.linspace(0, 1, len(vtk_files)) )

    for color, vtk_file in zip(colors, vtk_files):

        time_name = vtk_file.parent.name

        mesh = pv.read(vtk_file)
 
        if args.field not in mesh.array_names:
            print(  f"WARNING: '{args.field}' not found in {vtk_file}"  )
            continue

        contour = mesh.contour( isosurfaces=[args.iso],  scalars=args.field )

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

        # pts = contour.points

        # if len(pts) == 0:
        #     continue

        # ax.plot(
        #     pts[:, 0],
        #     pts[:, 1],
        #     linewidth=1.5,
        #     label=time_name
        # )

        n_contours += 1

    if n_contours == 0:
        raise RuntimeError(
            "No contours could be extracted."
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title(
        f"{args.field} = {args.iso}"
    )

    ax.grid(True)

    ax.legend(
        title="Time",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left"
    )

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