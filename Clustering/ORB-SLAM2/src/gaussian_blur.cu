/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#include <cuda.h>
#include <opencv2/core/hal/interface.h>

#include "gaussian_blur.h"
#include "ORBextractor.h"

#define TILE_H 15
#define TILE_W 39

__constant__ float d_kernel[KW * KH];

__global__ void gaussian_blur_kernel_tiled(
    uint old_h, uint old_w, const float *_scaleFactor,
    const uchar *original_img, const uchar *images,
    uchar *original_img_blurred, uchar *images_blurred,
    uint maxLevel, uint inputImageStep)
{
    // Thread coordinates in the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global pixel coords
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int level = blockIdx.z * blockDim.z + threadIdx.z;

    if (level >= maxLevel) return;

    float scaleFactor = _scaleFactor[level];
    uint new_rows = round(old_h / scaleFactor);
    uint new_cols = round(old_w / scaleFactor);
    if (x >= new_cols || y >= new_rows) return;

    int imageStep = (level == 0) ? inputImageStep : new_cols;
    int image_index = x + y * imageStep;

    const uchar* im[2] = { original_img, &images[level * old_w * old_h] };
    const uchar *image = im[(level != 0)];

    uchar* imBlured[2] = { original_img_blurred, &images_blurred[level * old_w * old_h] };
    uchar *imageBlured = imBlured[(level != 0)];

    // ------------------------
    // Shared memory tile with halo
    // ------------------------
    const int haloX = KW / 2;
    const int haloY = KH / 2;
    // const int tileW = blockDim.x + 2 * haloX;
    // const int tileH = blockDim.y + 2 * haloY;

    __shared__ uchar tile[TILE_H][TILE_W];

    // Global coords of top-left element in shared mem
    int tileStartX = blockIdx.x * blockDim.x - haloX;
    int tileStartY = blockIdx.y * blockDim.y - haloY;

    // Load the tile (each thread loads multiple elements if needed)
    for (int j = ty; j < TILE_H; j += blockDim.y) {
        for (int i = tx; i < TILE_W; i += blockDim.x) {
            int gx = min(max(tileStartX + i, 0), new_cols - 1);
            int gy = min(max(tileStartY + j, 0), new_rows - 1);
            tile[j][i] = image[gy * imageStep + gx];
        }
    }

    __syncthreads();

    // ------------------------
    // Apply Gaussian filter
    // ------------------------
    float acc = 0;
    for (int ky = 0; ky < KH; ky++) {
        for (int kx = 0; kx < KW; kx++) {
            acc += tile[ty + ky][tx + kx] * d_kernel[ky * KW + kx];
        }
    }

    imageBlured[image_index] = round(acc);
}

__global__ void gaussian_blur_kernel(uint old_h, uint old_w, float *_scaleFactor, const uchar *original_img, const uchar *images, uchar *original_img_blurred, uchar *images_blurred, float *kernel, uint maxLevel, uint inputImageStep) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int level = blockIdx.z * blockDim.z + threadIdx.z;

    if (level >= maxLevel){
        return;
    }
        

    const float scaleFactor = _scaleFactor[level];
    const uint new_rows = round(old_h * 1/scaleFactor);
    const uint new_cols = round(old_w * 1/scaleFactor);
    if (x >= new_cols || y >= new_rows){
        return;
    }

    const int imageStep = (level == 0) * inputImageStep + (level != 0) * new_cols;
    const int image_index = x + y * imageStep;

    const uchar* im[2] = {original_img, &(images[(level*old_w*old_h)])};
    const int imIndex = (level != 0);

    const uchar *image = im[imIndex];

    uchar* imBlured[2] = {original_img_blurred, &(images_blurred[(level*old_w*old_h)])};
    uchar *imageBlured = imBlured[imIndex];


    float acc = 0;
    for (int w = -KW/2; w<=KW/2; w++)
        for (int h = -KH/2; h<=KH/2; h++) {
            const int index = min(max(image_index+(h*imageStep)+w, 0), new_cols*new_rows);
            acc += image[index] * kernel[(h + KH/2) * KW + (w + KW/2)];
        }
    
    imageBlured[image_index] = round(acc);
}

void gaussian_blur( uchar *images, uchar *inputImage, uchar *imagesBlured, uchar *inputImageBlured, float *kernel, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, cudaStream_t cudaStream) {
    dim3 dg( ceil( (float)cols/64 ), ceil( (float)rows/8 ), maxLevel );
    dim3 db( 32, 8, 1 );

    cudaMemcpyToSymbol(d_kernel, kernel, sizeof(float)*KW*KH);
    // gaussian_blur_kernel<<<dg, db, 0, cudaStream>>>(rows, cols, mvScaleFactor, inputImage, images, inputImageBlured, imagesBlured, kernel, maxLevel, inputImageStep);
    gaussian_blur_kernel_tiled<<<dg, db>>>(rows, cols, mvScaleFactor, inputImage, images, inputImageBlured, imagesBlured, maxLevel, inputImageStep);
}