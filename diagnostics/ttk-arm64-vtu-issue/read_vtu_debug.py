#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 2:
        print("usage: read_vtu_debug.py file.vtu", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]

    import vtk

    # Capture VTK error output in stderr naturally; also print basic info.
    print("VTK:", vtk.vtkVersion.GetVTKVersion())

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    ok = r.Update()  # will trigger the parse
    out = r.GetOutput()

    print("Update() returned:", ok)
    if out is None:
        print("Output is None")
    else:
        print("points:", out.GetNumberOfPoints(), "cells:", out.GetNumberOfCells())

if __name__ == "__main__":
    main()
