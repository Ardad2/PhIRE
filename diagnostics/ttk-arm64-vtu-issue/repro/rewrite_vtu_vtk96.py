import sys
import vtk

inp = sys.argv[1]
out = sys.argv[2]

r = vtk.vtkXMLUnstructuredGridReader()
r.SetFileName(inp)
r.Update()
ug = r.GetOutput()

if ug is None or ug.GetNumberOfPoints() == 0:
    print("ERROR: input read failed (0 points).")
    sys.exit(2)

w = vtk.vtkXMLUnstructuredGridWriter()
w.SetFileName(out)
w.SetInputData(ug)

# Write in a conservative mode for compatibility
if hasattr(w, "SetDataModeToAscii"):
    w.SetDataModeToAscii()
elif hasattr(w, "SetDataModeToBinary"):
    w.SetDataModeToBinary()

# Turn off appended-base64 if possible (older readers sometimes get picky)
if hasattr(w, "EncodeAppendedDataOff"):
    w.EncodeAppendedDataOff()
elif hasattr(w, "SetEncodeAppendedData"):
    w.SetEncodeAppendedData(0)

ok = w.Write()
print("Wrote:", out, "ok=", ok, "points=", ug.GetNumberOfPoints(), "cells=", ug.GetNumberOfCells())
