
### import packages

require(sf)


### main code

set.seed(0) # random seed for reproducibility

# open file, select US lakes
shape <- read_sf(dsn = 'HydroLAKES_points_v10_shp/', layer = 'HydroLAKES_points_v10')
shape <- shape[shape$Country == 'United States of America',]

# filter data according to position
xy <- st_coordinates(shape)
xy <- xy[xy[,2] < 50,]
xy <- xy[xy[,2] > 20,]
xy <- xy[xy[,1] < -50,]
xy <- xy[xy[,1] > -130,]

# select LAT, LONG
colnames(xy) <- c('LATITUDE', 'LONGITUDE')

# shuffle
xy <- xy[sample(nrow(xy)), ]

# write to file
write.csv(xy, 'lakes_latlong.csv', row.names = FALSE)

# open file for rivers
shape <- read_sf(dsn = 'HydroRIVERS_v10_na_shp/', layer = 'HydroRIVERS_v10_na')

# filter data according to position
xy <- st_coordinates(shape)
xy <- xy[xy[,2] < 50,]
xy <- xy[xy[,2] > 20,]
xy <- xy[xy[,1] < -50,]
xy <- xy[xy[,1] > -130,]

# select LAT, LONG
xy <- xy[,1:2]
colnames(xy) <- c('LATITUDE', 'LONGITUDE')

# shuffle, select 120k rivers
xy <- xy[sample(nrow(xy)), ]
xy <- xy[1:120000, ]

# write to file
write.csv(xy, 'rivers_latlong.csv', row.names = FALSE)
