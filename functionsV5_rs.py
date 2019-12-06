from gdalconst import *
from osgeo import gdal


def loadImage(img_path):
    band_data = []
    band_name = []

    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    if dataset is None:
        print("Unable to open image file.")
        return band_data
    else:
        print("Open image file success.\n")

        # 读取地理变换参数
        param_geoTransform = dataset.GetGeoTransform()
        print("GeoTransform info:\n", param_geoTransform, "\n")

        # 读取投影信息
        param_proj = dataset.GetProjection()
        print("Projection info:\n", param_proj, "\n")

        # 读取波段数及影像大小
        bands_num = dataset.RasterCount
        print("Image height:" + dataset.RasterYSize.__str__() + " Image width:" + dataset.RasterXSize.__str__())
        print(bands_num.__str__() + " bands in total.")

        # 依次读取波段数据
        for i in range(bands_num):
            # 获取影像的第i+1个波段
            band_i = dataset.GetRasterBand(i + 1)

            # 获取影像第i+1个波段的描述(名称)
            name = band_i.GetDescription()
            band_name.append(name)

            # 读取第i+1个波段数据
            data = band_i.ReadAsArray(0, 0, band_i.XSize, band_i.YSize)
            band_data.append(data)

            print("band " + (i + 1).__str__() + " read success.")
            if name != "":
                print("Name:", name)
        return band_data, band_name, param_geoTransform, param_proj


def getImgBriefInfo(img_path):
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    if dataset is None:
        print("Unable to open image file.")
    else:
        print("=" * 15, "Information Summary of Image", "=" * 15)

        # 输出影像路径
        print("File name:", img_path)

        # 读取波段数及影像大小
        bands_num = dataset.RasterCount
        print("Image height(pixel):", dataset.RasterYSize.__str__())
        print("Image width(pixel):", dataset.RasterXSize.__str__())
        print("Number of bands:", bands_num.__str__())

        # 读取地理变换参数
        param_geoTransform = dataset.GetGeoTransform()
        if param_geoTransform == "":
            print("GeoTransform info: None")
        else:
            print("GeoTransform info:", param_geoTransform)

        # 读取投影信息
        param_proj = dataset.GetProjection()
        if param_proj == "":
            print("Projection info: None")
        else:
            print("Projection info:", param_proj)

        # 依次读取波段数据
        for i in range(bands_num):
            # 获取影像的第i+1个波段
            band_i = dataset.GetRasterBand(i + 1)

            # 获取影像第i+1个波段的描述(名称)
            name = band_i.GetDescription()
            if name != "":
                print("Name of band", (i + 1), ":", name)
        print("=" * 15, "Information Summary of Image", "=" * 15)


def loadSingleBand(img_path, index=1):
    """
    以只读方式打开遥感影像，用于加载单波段影像到内存中

    :param img_path: 需要加载的影像路径
    :param index: 波段的索引，默认为1，即加载第一个波段
    :return: 影像数据

    """
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    if dataset is None:
        print("Unable to open image file.")
        return False
    else:
        print("Open image file success.")

        # 读取地理变换参数
        param_geoTransform = dataset.GetGeoTransform()

        # 读取投影信息
        param_proj = dataset.GetProjection()

        # 读取波段数及影像大小
        bands_num = dataset.RasterCount

        if 1 <= index <= bands_num:
            print("Read band", index.__str__(), "data...")
        else:
            print("Out of band index,read the first band.")
            index = 1

        # 获取影像的第i+1个波段
        band_i = dataset.GetRasterBand(index)

        # 获取影像第i+1个波段的描述(名称)
        name = band_i.GetDescription()
        if name != "":
            print("Band name:", name)

        # 读取第i+1个波段数据
        data = band_i.ReadAsArray(0, 0, band_i.XSize, band_i.YSize)

        return data, name, param_geoTransform, param_proj


def writeImage(save_path, bands, names=None, geotrans=None, proj=None):
    projection = [
        # WGS84坐标系(EPSG:4326)
        """GEOGCS["WGS 84", DATUM["WGS_1984", SPHEROID["WGS 84", 6378137, 298.257223563, AUTHORITY["EPSG", "7030"]], AUTHORITY["EPSG", "6326"]], PRIMEM["Greenwich", 0, AUTHORITY["EPSG", "8901"]], UNIT["degree", 0.01745329251994328, AUTHORITY["EPSG", "9122"]], AUTHORITY["EPSG", "4326"]]""",
        # Pseudo-Mercator、球形墨卡托或Web墨卡托(EPSG:3857)
        """PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs"],AUTHORITY["EPSG","3857"]]"""
    ]

    if bands is None or bands.__len__() == 0:
        return False
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        band1 = bands[0]
        # 设置影像保存大小、波段数
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(save_path, img_width, img_height, num_bands, datatype)

        if dataset is not None:
            # 写入仿射变换参数
            if geotrans is not None:
                dataset.SetGeoTransform(geotrans)

            # 写入投影参数
            if proj is not None:
                if proj is 'WGS84' or \
                        proj is 'wgs84' or \
                        proj is 'EPSG:4326' or \
                        proj is 'EPSG-4326' or \
                        proj is '4326':
                    dataset.SetProjection(projection[0])  # 写入投影
                elif proj is 'EPSG:3857' or \
                        proj is 'EPSG-3857' or \
                        proj is '3857':
                    dataset.SetProjection(projection[1])  # 写入投影
                else:
                    dataset.SetProjection(proj)  # 写入投影

            # 逐波段写入数据
            for i in range(bands.__len__()):
                raster_band = dataset.GetRasterBand(i + 1)

                # 设置没有数据的像素值为0
                raster_band.SetNoDataValue(0)

                if names is not None:
                    # 设置波段的描述(名称)
                    raster_band.SetDescription(names[i])

                # 写入数据
                raster_band.WriteArray(bands[i])
            print("save image success.")
            return True
