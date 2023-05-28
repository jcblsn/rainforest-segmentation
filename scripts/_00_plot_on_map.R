install.packages(c("raster", "leaflet"))

library(raster)
library(leaflet)
library(tidyverse)
library(sf)

# -------------------- test individual image
raster_data <- raster("~/Documents/Files/School/grad/ST456 Deep Learning/group-project/project-2023-group-11/data/download/AMAZON/Training/image/S2A_MSIL2A_20200111T142701_N0213_R053_T20NQG_20200111T164651_01_07.tif")

leaflet() %>%
    addTiles() %>%
    addRasterImage(raster_data, opacity = 0.8, colors = "YlOrRd")

# ----------------- plot all images on map

# get paths
image_dir <- "~/Documents/Files/School/grad/ST456 Deep Learning/group-project/project-2023-group-11/data/download/AMAZON/"

image_sub_dirs <- list.dirs(image_dir, recursive = TRUE)
image_paths <- list()
for (sub_dir in image_sub_dirs) {
    image_paths <- c(image_paths, list.files(sub_dir, full.names = TRUE))
}
image_paths <- image_paths[grepl(".tif", image_paths)]
image_paths <- image_paths[!grepl("mask", image_paths)]
image_paths <- image_paths[!grepl("label", image_paths)]

# read in images
raster_data <- lapply(image_paths, function(image_path) raster(image_path))

# -------------------- plot actual images
leaflet() %>%
    addTiles() %>%
    {
        map <- .
        for (i in seq_along(raster_data[1:10])) {
            map <- map %>%
                addRasterImage(raster_data[[i]], opacity = 0.8, colors = "YlOrRd", layerId = i, group = paste("Image", i))
        }
        map
    } %>%
    addLayersControl(
        overlayGroups = sapply(seq_along(raster_data[1:10]), function(i) paste("Image", i)),
        options = layersControlOptions(collapsed = FALSE)
    )

# ---------------- plot points

# check
# lapply(raster_data, function(r) {
#     crs(r)
# })

raster_data <- lapply(raster_data, function(r) {
    projectRaster(r, crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
})

center_coordinates <- lapply(raster_data, function(r) {
    e <- extent(r)
    lat <- (e@ymax + e@ymin) / 2
    lng <- (e@xmax + e@xmin) / 2
    xy <- data.frame(x = lng, y = lat)

    # Use PROJ string representation for the CRS
    crs_proj_string <- sprintf("+proj=utm +zone=20 +datum=WGS84 +units=m +type=crs")

    transformed_points <- st_as_sf(xy, coords = c("x", "y"), crs = crs_proj_string) %>%
        st_transform(crs = 4326)

    # Extract latitude and longitude from geometry column
    coords <- st_coordinates(transformed_points)
    data.frame(lat = coords[, 2], lng = coords[, 1])
})

center_coordinates_df <- do.call(rbind, center_coordinates)

# plot the map with circle markers
leaflet(center_coordinates_df) %>%
    addTiles() %>%
    addCircleMarkers(lng = ~lng, lat = ~lat, fillColor = "red", fillOpacity = 0.8, stroke = FALSE, radius = 5)
