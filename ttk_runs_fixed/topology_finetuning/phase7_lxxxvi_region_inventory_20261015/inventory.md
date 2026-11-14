# Phase 7 — preliminary WTK geographic and temporal inventory

No meteorological fields, SR outputs, scores or model test-set splits generated.

- Source: `/nrel/wtk-us.h5`; grid shape (61368, 1602, 2976); coordinate format compound-HW
- Frozen old-168 extraction rectangle: `(720, 1220, 2102, 2602)`; reference alignment PASS
- Required buffer: 80 WTK grid cells (only an index-space separation criterion)
- Provisional candidate set: `gulf_inland, upper_midwest, pacific_inland, southwest_borderland`; subject to manual meteorology and split review

| Candidate | Intended regime | WTK center row,col | Full 500² | Separated from old area |
|---|---|---|---|---|
| rocky_mountain_inland | Rocky Mountain / intermountain | (933,912) | True | True |
| great_plains_south | Southern Great Plains | (699,1291) | True | True |
| gulf_inland | Gulf-influenced inland/coastal transition | (455,1721) | True | True |
| upper_midwest | Upper Midwest / Great Lakes transition | (1099,1565) | True | True |
| pacific_inland | Western basin and range | (1070,565) | True | True |
| northern_plains | Northern Great Plains | (1270,1231) | True | True |
| appalachian_inland | Appalachian terrain | (733,2010) | True | False |
| southwest_inland | Southwest high plateau | (640,860) | True | True |
| central_plains | Central Great Plains | (831,1383) | True | True |
| southeast_inland | Southeast inland | (537,1993) | True | False |
| northern_rockies | Northern Rockies | (1265,895) | True | True |
| pacific_northwest | Pacific Northwest transition | (1287,590) | True | True |
| southwest_borderland | Southern Arizona / borderland transition (domain suitability unverified) | (423,822) | True | True |
| east_texas_inland | East Texas / inland Gulf transition (domain suitability unverified) | (447,1464) | True | True |

## Season candidates (not finalized)

| Requested first UTC hour | First WTK index | Last WTK index | Valid 168 hourly steps |
|---|---:|---:|---|
| 2010-02-01T00:00:00 | 27048 | 27215 | True |
| 2010-05-01T00:00:00 | 29184 | 29351 | True |
| 2010-08-01T00:00:00 | 31392 | 31559 | True |
| 2010-11-01T00:00:00 | 33600 | 33767 | True |
| 2012-02-01T00:00:00 | 44568 | 44735 | True |
| 2012-05-01T00:00:00 | 46728 | 46895 | True |
| 2012-08-01T00:00:00 | 48936 | 49103 | True |
| 2012-11-01T00:00:00 | 51144 | 51311 | True |

## Important boundaries

- A disjoint 500×500 native grid rectangle does not prove statistical meteorological independence.
- Original PhIRE pretrained-weight train/test membership remains unknown; this inventory guarantees neither pretraining novelty nor temporal novelty.
- Geographic regime labels reflect proposed targets, not verified physical descriptors. Validate terrain, coastline, speed/gradient distributions before finalizing.
- Do not tune models or inspect locked test-region results while choosing losses.
- WTK timestamps are read from metadata; check local/global temporal overlap with archived training windows before test-set freezing.
- The full-field/patch topological evaluation protocol is a separate following phase.
