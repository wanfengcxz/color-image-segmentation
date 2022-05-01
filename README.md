# color-image-segmentation

A Multi-Objective Evolutionary Algorithm for Color Image Segmentation

## NSGA 2

NSGA-II is a modification of NSGA [Deb et al., 2002a]. NSGA-II computes the cost
of an individual x by taking into account not only the individuals that dominate
it, but also the individuals that it dominates. For each individual, we also compute
a crowding distance by finding the distance to the nearest individuals along each
objective function dimension.
