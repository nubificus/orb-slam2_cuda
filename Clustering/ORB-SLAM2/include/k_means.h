#ifndef K_MEANS
#define K_MEANS

#include "ORBextractor.h"

void filter_points(ORB_SLAM2::GpuPoint *d_datapoints, ORB_SLAM2::GpuPoint *final_points_buffer, int2 *d_centroids, int *d_clust_sizes, uint *N, int *K, int2 *initial_centroids, int *mnFeatrues, int maxLevel, int points_offset, int cluster_offset, cudaStream_t stream);

#endif