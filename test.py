from cropharvest.eo import LocalDiskExporter
from cropharvest.countries import BBox

from datetime import date

local_disk_exporter = LocalDiskExporter()

bbox = BBox(min_lon=16.43, min_lat=-18.70, max_lon=16.44, max_lat=-18.69) #(minx, miny, maxx, maxy)
# bbox = 16.43, -18.70, 16.44, -18.69 #(minx, maxx, miny, maxy)


local_disk_exporter.export_for_bbox(
    bbox=bbox,
    start_date=date(2020, 7, 1),
    end_date=date(2020, 10, 1),
    pixel_size=30,
    file_name="somewhere_nigeria"
)
