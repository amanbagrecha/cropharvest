import math
import os
import numpy as np
import pandas as pd
import rasterio
from datetime import timedelta, date

try:
    import ee
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The Earth Engine API is not installed. "
        + "Please install it with `pip install earthengine-api`."
    )

from .ee_boundingbox import EEBoundingBox
from .sentinel1 import (
    get_single_image as get_single_s1_image,
    get_image_collection as get_s1_image_collection,
)
from .sentinel2 import get_single_image as get_single_s2_image
from .era5 import get_single_image as get_single_era5_image
from .srtm import get_single_image as get_single_srtm_image

from .utils import make_combine_bands_function
from cropharvest.bands import DYNAMIC_BANDS
from cropharvest.countries import BBox
from cropharvest.config import (

    DAYS_PER_TIMESTEP,
)

from typing import Dict, List, Optional


DYNAMIC_IMAGE_FUNCTIONS = [get_single_s2_image, get_single_era5_image]
STATIC_IMAGE_FUNCTIONS = [get_single_srtm_image]


class LocalDiskExporter:
    output_folder_name = "eo_data"

    def __init__(
        self,
        credentials: Optional[str] = None,
    ) -> None:

        try:
            if credentials:
                ee.Initialize(credentials=credentials)
            else:
                ee.Initialize()
        except Exception:
            print("This code doesn't work unless you have authenticated your earthengine account")

    def _export_for_polygon(
        self,
        bbox,
        polygon: ee.Geometry.Polygon,
        start_date: date,
        end_date: date,
        pixel_size,
        crs,
        file_name,
        identifier,
        days_per_timestep: int = DAYS_PER_TIMESTEP,
    ) -> bool:

        image_collection_list: List[ee.Image] = []
        cur_date = start_date
        cur_end_date = cur_date + timedelta(days=days_per_timestep)

        # first, we get all the S1 images in an exaggerated date range
        vv_imcol, vh_imcol = get_s1_image_collection(
            polygon, start_date - timedelta(days=31), end_date + timedelta(days=31)
        )

        while cur_end_date <= end_date:
            image_list: List[ee.Image] = []

            # first, the S1 image which gets the entire s1 collection
            image_list.append(
                get_single_s1_image(
                    region=polygon,
                    start_date=cur_date,
                    end_date=cur_end_date,
                    vv_imcol=vv_imcol,
                    vh_imcol=vh_imcol,
                )
            )
            for image_function in DYNAMIC_IMAGE_FUNCTIONS:
                image_list.append(
                    image_function(region=polygon, start_date=cur_date, end_date=cur_end_date)
                )
            image_collection_list.append(ee.Image.cat(image_list))

            cur_date += timedelta(days=days_per_timestep)
            cur_end_date += timedelta(days=days_per_timestep)

        # now, we want to take our image collection and append the bands into a single image
        imcoll = ee.ImageCollection(image_collection_list)
        combine_bands_function = make_combine_bands_function(DYNAMIC_BANDS)
        img = ee.Image(imcoll.iterate(combine_bands_function))

        # finally, we add the SRTM image seperately since its static in time
        total_image_list: List[ee.Image] = [img]
        for static_image_function in STATIC_IMAGE_FUNCTIONS:
            total_image_list.append(static_image_function(region=polygon))

        img = ee.Image.cat(total_image_list)


        input_bands = img.bandNames().getInfo()
        df = self.create_df(img, polygon, pixel_size=pixel_size, crs="EPSG:4326")
        # len_x , len_y = self.generate_pts(bbox, pixel_size)
        side_shape = df.shape[0]
        
        data_matrix = np.array(df.loc[:, input_bands].values)
        # .reshape(
            # side_shape, side_shape, len(input_bands)
        # )
        data_matrix = np.flip(data_matrix, axis=0)
        transform = rasterio.transform.from_origin(
            bbox.min_lon, bbox.max_lat, pixel_size, pixel_size
        )

        self.save_tif(f"{file_name}_{identifier}", data_matrix, transform, crs)
        print(f"file saved as {file_name}_{identifier}.tif")
        return True

    def create_df(self, img_col, feature, pixel_size, crs):
        imgcol = ee.ImageCollection(img_col)
        polygon = ee.FeatureCollection(feature).geometry()
        df = pd.DataFrame(imgcol.getRegion(polygon, pixel_size, crs).getInfo())
        df, df.columns = df[1:], df.iloc[0]
        df = df.drop(["id", "time"], axis=1)

        return df

    def generate_pts(self, bbox, pixel_size):
        x_pt = ee.List.sequence(bbox.min_lon, bbox.max_lon, pixel_size)
        y_pt = ee.List.sequence(bbox.min_lat, bbox.max_lat, pixel_size)

        return len(x_pt.getInfo()), len(y_pt.getInfo())

    def save_tif(self,file_name, data_array, transform, crs):
        

        options = {
            "driver": "Gtiff",
            "height": data_array.shape[0],
            "width": data_array.shape[1],
            "count": 1,
            "dtype": np.float32,
            "crs": crs,
            "transform": transform,
        }
        with rasterio.open(f"{file_name}.tif", "w", **options) as src:
            src.write(data_array,1)
            # for band in range(count):
                # src.write(data_array[:, :, band], band + 1)

        return True

    def export_for_bbox(
        self,
        bbox: BBox,
        start_date: date,
        end_date: date,
        pixel_size,
        file_name,
        metres_per_polygon: Optional[int] = 500,
    ) -> Dict[str, bool]:
        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        ee_bbox = EEBoundingBox.from_bounding_box(bounding_box=bbox, padding_metres=0)

        if metres_per_polygon is not None:
            regions = ee_bbox.to_polygons(metres_per_patch=metres_per_polygon)
            ids = [f"batch_{i}_{i}" for i in range(len(regions))]
        else:
            regions = [ee_bbox.to_ee_polygon()]
            ids = ["batch_0"]

        import pyproj
        from pyproj import Transformer

        def latlon_to_utm_zone(lat, lon):
            """
            Calculate the UTM zone for the given latitude and longitude.
            Returns UTM zone number and hemisphere (N or S).
            """
            zone_number = math.floor((lon + 180) / 6) + 1
            hemisphere = "N" if lat >= 0 else "S"
            return zone_number, hemisphere

        def latlon_to_utm_crs(lat, lon):
            """
            Convert latitude and longitude to UTM CRS.
            Returns the pyproj CRS object.
            """
            zone_number, hemisphere = latlon_to_utm_zone(lat, lon)
            utm_crs = pyproj.CRS(
                f"EPSG:326{zone_number}" if hemisphere == "N" else f"EPSG:327{zone_number}"
            )
            return utm_crs

        lon = sum(bbox[::2]) / 2
        lat = sum(bbox[1::2]) / 2
        # Convert to UTM CRS
        utm_crs = latlon_to_utm_crs(lat, lon)

        transformer = Transformer.from_crs("EPSG:4326", utm_crs)
        # Convert the bounding box coordinates
        minx, miny = transformer.transform(bbox[1], bbox[0])
        maxx, maxy = transformer.transform(bbox[3], bbox[2])

        bbox_proj = BBox(min_lon=minx, min_lat=miny, max_lon=maxx, max_lat=maxy)

        return_obj = {}
        for identifier, region in zip(ids, regions):
            return_obj[identifier] = self._export_for_polygon(
                bbox=bbox_proj,
                polygon=region,
                start_date=start_date,
                end_date=end_date,
                pixel_size=pixel_size,
                crs=utm_crs,
                file_name=file_name,
                identifier=identifier
            )
        return return_obj
