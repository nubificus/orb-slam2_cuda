#include "ORBextractor.h"
#include "k_means.h"

#define TDB 1024
#define N_ITER 1

__device__ inline float distance(ORB_SLAM3::GpuPoint x1, int2 x2)
{
    const float diffx = (float)x2.x-x1.x;
    const float diffy = (float)x2.y-x1.y;
	return (diffx*diffx)+(diffy*diffy);
}


__global__ void kMeansClusterAssignment(ORB_SLAM3::GpuPoint *d_datapoints_, ORB_SLAM3::GpuPoint *final_points_, /*int *d_clust_assn_,*/ int2 *d_centroids_, uint *N_, int *K_, int maxLevel, int points_offset, int cluster_offset)
{
	//get idx for this datapoint
	const int level = blockIdx.y;
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const uint N = N_[level];

	//bounds check
	if (idx >= N) return;

	const int K = K_[level];

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	ORB_SLAM3::GpuPoint *final_points = &(final_points_[points_offset*level]);
	final_points[idx].clust_assn = -1;
	int2 *d_centroids = &(d_centroids_[cluster_offset*level]);

	//find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c=0; c<K; c++)
	{
		float dist = distance(d_datapoints[idx],d_centroids[c]);

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid = c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_datapoints[idx].clust_assn = closest_centroid;
}


// template <int T>
// __global__ void kMeansCentroidUpdate(ORB_SLAM3::GpuPoint *d_datapoints_, /*int *d_clust_assn_,*/ int2 *d_centroids_, int *d_clust_sizes_, uint *N_, int *K_, int maxLevel, int points_offset, int cluster_offset)
// {
// 	const int level = blockIdx.y;
// 	//get idx of thread at grid level
// 	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

// 	if (level >= maxLevel) return;

// 	const uint N = N_[level];

// 	//bounds check
// 	if (idx >= N) return;

// 	const int K = K_[level];

// 	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset]);
// 	// int *d_clust_assn = &(d_clust_assn_[points_offset]);
// 	int2 *d_centroids = &(d_centroids_[cluster_offset]);
// 	int *d_clust_sizes = &(d_clust_sizes_[cluster_offset]);

// 	//get idx of thread at the block level
// 	const int s_idx = threadIdx.x;

// 	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
// 	__shared__ int2 s_datapoints[T];
// 	s_datapoints[s_idx]= make_int2(d_datapoints[idx].x, d_datapoints[idx].y);

// 	__shared__ int s_clust_assn[T];
// 	s_clust_assn[s_idx] = d_datapoints[idx].clust_assn;

// 	extern __shared__ int b_clust_datapoint_sums_x[];
// 	int *b_clust_datapoint_sums_y = &b_clust_datapoint_sums_x[cluster_offset];
// 	int *b_clust_sizes = &b_clust_datapoint_sums_y[cluster_offset];

// 	for (int i=0; i+s_idx<cluster_offset; i=i+T) {
// 		b_clust_datapoint_sums_x[i+s_idx] = 0;
// 		b_clust_datapoint_sums_y[i+s_idx] = 0;
// 		b_clust_sizes[i+s_idx] = 0;
// 	}

// 	__syncthreads();

// 	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
// 	if(s_idx==0)
// 	{
// 		// int b_clust_datapoint_sums_x[K]={0};
// 		// int b_clust_datapoint_sums_y[K]={0};
// 		// int b_clust_sizes[K]={0};

// 		for(int j=0; j< blockDim.x; ++j)
// 		{
// 			const int clust_id = s_clust_assn[j];
// 			b_clust_datapoint_sums_x[clust_id]+=s_datapoints[j].x;
// 			b_clust_datapoint_sums_y[clust_id]+=s_datapoints[j].y;
// 			b_clust_sizes[clust_id]+=1;
// 		}

// 		//Now we add the sums to the global centroids and add the counts to the global counts.
// 		for(int z=0; z < K; ++z)
// 		{
// 			atomicAdd(&(d_centroids[z].x),b_clust_datapoint_sums_x[z]);
// 			atomicAdd(&(d_centroids[z].y),b_clust_datapoint_sums_y[z]);
// 			atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
// 		}
// 	}

// 	__syncthreads();

// 	//currently centroids are just sums, so divide by size to get actual centroids
// 	if(idx < K){
// 		d_centroids[idx].x = d_centroids[idx].x/d_clust_sizes[idx]; 
// 		d_centroids[idx].y = d_centroids[idx].y/d_clust_sizes[idx]; 
// 	}

// }


__global__ void get_max_score(ORB_SLAM3::GpuPoint *d_datapoints_, uint *N_, int *k_scores_, int maxLevel, int points_offset, int cluster_offset){
	const int level = blockIdx.y;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const uint N = N_[level];

	if (idx >= N) return;

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	int *k_scores = &(k_scores_[cluster_offset*level]);

	const int k = d_datapoints[idx].clust_assn;
	const int score = d_datapoints[idx].score;

	atomicMax(&(k_scores[k]), score);
}

__global__ void get_max_points(ORB_SLAM3::GpuPoint *d_datapoints_, uint *N_, int *k_scores_, ORB_SLAM3::GpuPoint *final_points_, int maxLevel, int points_offset, int cluster_offset){
	const int level = blockIdx.y;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const uint N = N_[level];

	if (idx >= N) return;

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	ORB_SLAM3::GpuPoint *final_points = &(final_points_[points_offset*level]);
	int *k_scores = &(k_scores_[cluster_offset*level]);

	const int k = d_datapoints[idx].clust_assn;
	const int score = d_datapoints[idx].score;

	const int max_score = k_scores[k];

	if (max_score == score) {
		final_points[k] = d_datapoints[idx];
	} else {
		d_datapoints[idx].clust_assn = -1;
	}

}

__global__ void copy_points(ORB_SLAM3::GpuPoint *d_datapoints_, /*int *d_clust_assn_,*/ int *K_, uint *N_, ORB_SLAM3::GpuPoint *final_points_, int maxLevel, int points_offset){
	const int level = blockIdx.y;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//in questo caso idx si riferisce a k

	if (level >= maxLevel) return;

	const int K = K_[level];

	if (idx >= K) return;

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	ORB_SLAM3::GpuPoint *final_points = &(final_points_[points_offset*level]);

	if (final_points[idx].clust_assn != -1)
		d_datapoints[idx] = final_points[idx];

}

__global__ void last_cluster_assign(ORB_SLAM3::GpuPoint *d_datapoints_, ORB_SLAM3::GpuPoint *final_points_, int *k_scores_, /*int *d_clust_assn_,*/ int2 *d_centroids_, uint *N_, int *K_, int maxLevel, int points_offset, int cluster_offset)
{
	//get idx for this datapoint
	const int level = blockIdx.y;
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const uint N = N_[level];
	const int K = K_[level];

	ORB_SLAM3::GpuPoint *final_points = &(final_points_[points_offset*level]);

	//bounds check
	if (idx >= K || final_points[idx].clust_assn != -1) return;

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	int2 *d_centroids = &(d_centroids_[cluster_offset*level]);

	float min_dist = INFINITY;
	int closest_point = 0;

	for(int i=0; i<N; i++)
	{
		if (d_datapoints[i].clust_assn != -1) continue;

		float dist = distance(d_datapoints[i],d_centroids[idx]);
		
		if(dist < min_dist)
		{
			min_dist = dist;
			closest_point = i;
		}
	}

	final_points[idx] = d_datapoints[closest_point];
}


void filter_points(ORB_SLAM3::GpuPoint *d_datapoints, ORB_SLAM3::GpuPoint *final_points_buffer, int2 *d_centroids, int *d_clust_sizes, uint *N, int *K, int2 *initial_centroids, int *mnFeatrues, int maxLevel, int points_offset, int cluster_offset, cudaStream_t stream) {
	dim3 dg( ceil( (float)points_offset/TDB ), maxLevel );
	dim3 dg2( ceil( (float)cluster_offset/TDB ), maxLevel );
    dim3 db( TDB );

	cudaMemcpyAsync(d_centroids, initial_centroids, sizeof(int2)*maxLevel*cluster_offset, cudaMemcpyDeviceToDevice, stream);
	cudaMemsetAsync(d_clust_sizes,0,cluster_offset*maxLevel*sizeof(int), stream);

	// for (int i=0; i<N_ITER; i++){
	kMeansClusterAssignment<<<dg, db, 0, stream>>>(d_datapoints, final_points_buffer, d_centroids, N, K, maxLevel, points_offset, cluster_offset);

		//reset centroids and cluster sizes (will be updated in the next kernel)
		// cudaMemsetAsync(d_centroids,0.0,cluster_offset*maxLevel*sizeof(int2), stream);
		// cudaMemsetAsync(d_clust_sizes,0,cluster_offset*maxLevel*sizeof(int), stream);

		// kMeansCentroidUpdate<TDB><<<dg, db, cluster_offset*sizeof(int)*3, stream>>>(d_datapoints, /*d_clust_assn,*/ d_centroids, d_clust_sizes, N, K, maxLevel, points_offset, cluster_offset);
	// }

	get_max_score<<<dg, db, 0, stream>>>(d_datapoints, N, d_clust_sizes, maxLevel, points_offset, cluster_offset);
	get_max_points<<<dg, db, 0, stream>>>(d_datapoints, N, d_clust_sizes, final_points_buffer, maxLevel, points_offset, cluster_offset);
	last_cluster_assign<<<dg2, db, 0, stream>>>(d_datapoints, final_points_buffer, d_clust_sizes, d_centroids, N, K, maxLevel, points_offset, cluster_offset);
	copy_points<<<dg2, db, 0, stream>>>(d_datapoints, K, N, final_points_buffer, maxLevel, points_offset);
}