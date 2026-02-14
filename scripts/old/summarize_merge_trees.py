import glob, os, csv
import vtk

def read_tree(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    ug = r.GetOutput()

    nodes = ug.GetNumberOfPoints()
    arcs = ug.GetNumberOfCells()

    pdata = ug.GetPointData()
    pers = pdata.GetArray("Persistence")
    if pers is not None:
        lo, hi = pers.GetRange()
    else:
        lo = hi = None

    return nodes, arcs, lo, hi

rows = []
for p in sorted(glob.glob("ttk_outputs/mt/**/*.vtu", recursive=True)):
    # only port_0/port_1 are .vtu; port_2 is .vti
    if "_port_0.vtu" not in p:
        continue
    nodes, arcs, lo, hi = read_tree(p)
    rows.append([p, nodes, arcs, lo, hi])

os.makedirs("ttk_outputs/summary", exist_ok=True)
out = "ttk_outputs/summary/mt_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "nodes", "arcs", "persistence_min", "persistence_max"])
    w.writerows(rows)

print("Wrote:", out, "rows:", len(rows))
