import glob, os, csv
import vtk

def read_pd(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    ug = r.GetOutput()

    npts = ug.GetNumberOfPoints()
    pdata = ug.GetPointData()

    # TTK often provides "Persistence" array
    pers = pdata.GetArray("Persistence")
    if pers is not None:
        lo, hi = pers.GetRange()
    else:
        lo = hi = None

    return npts, lo, hi

rows = []
for p in sorted(glob.glob("ttk_outputs/pd/**/*.vtu", recursive=True)):
    npts, lo, hi = read_pd(p)
    rows.append([p, npts, lo, hi])

os.makedirs("ttk_outputs/summary", exist_ok=True)
out = "ttk_outputs/summary/pd_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "num_points", "persistence_min", "persistence_max"])
    w.writerows(rows)

print("Wrote:", out, "rows:", len(rows))
