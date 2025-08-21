/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#define INIT_IMAGE_W 752
#define INIT_IMAGE_H 480

#define PATCH_SIZE 31
#define HALF_PATCH_SIZE 15
#define EDGE_THRESHOLD 19

#define KW 7
#define KH 7
#define SIGMA 2


namespace ORB_SLAM2
{

    struct GpuPoint {
        float x;
        float y;
        uint score;
        int octave;
        float angle;
        float size;
        int clust_assn;
        uchar descriptor[32];
    };

    struct copyPyrimid_t
    {
        int nlevels;
        uchar *outputImages;
        int cols;
        int rows;
        float *mvScaleFactor;
        cv::Mat *mvImagePyramid;
    };

    struct OrbKeyPoint {
        cv::KeyPoint point;
        uchar *descriptor;
    };

    class ExtractorNode
    {
    public:
        ExtractorNode():bNoMore(false){}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        #ifdef CPUONLY
        std::vector<cv::KeyPoint> vKeys;
        #else
        std::vector<OrbKeyPoint> vKeys;
        #endif
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;
    };

    class ORBextractor
    {
    public:

        enum {HARRIS_SCORE=0, FAST_SCORE=1 };

        ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                     int iniThFAST, int minThFAST);

        ~ORBextractor();

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        void operator()( cv::InputArray _image, cv::InputArray _mask,
                         std::vector<cv::KeyPoint>& _keypoints,
                         cv::OutputArray _descriptors);

        #ifdef VACCEL
        int vaccel_orb_operator(const cv::Mat& image, const cv::Mat& mask,
                    std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
                    // std::vector<cv::Mat>& pyr);

        void BuildImagePyramid(const cv::Mat& image) {
            ComputePyramid(image);
        }
        #endif
        int inline GetLevels(){
            return nlevels;}

        float inline GetScaleFactor(){
            return scaleFactor;}

        std::vector<float> inline GetScaleFactors(){
            return mvScaleFactor;
        }

        std::vector<float> inline GetInverseScaleFactors(){
            return mvInvScaleFactor;
        }

        std::vector<float> inline GetScaleSigmaSquares(){
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares(){
            return mvInvLevelSigma2;
        }

        std::vector<cv::Mat> mvImagePyramid;

        GpuPoint *d_corner_buffer2;
        uchar *d_images;
        uchar *d_inputImage;
        int imageStep;
        int rows;
        int cols;
        int nlevels;
        std::vector<int> mnFeaturesPerLevel;
        float *d_scaleFactor;


    protected:

        void ComputePyramid(cv::Mat image);
        void CopyKeyAndDescriptor(std::vector<cv::KeyPoint>& _keypoints, cv::OutputArray _descriptors);
        void ComputeMonoIndex(int vLappingArea0, int vLappingArea1);
        #ifdef CPUONLY
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

        #else
        void ComputeKeyPointsOctTree(std::vector<std::vector<OrbKeyPoint> >& allKeypoints);
        std::vector<OrbKeyPoint> DistributeOctTree(const std::vector<OrbKeyPoint>& vToDistributeKeys, const int &minX,
                                                   const int &maxX, const int &minY, const int &maxY, const int &N, const int &level);
        #endif

        void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
        std::vector<cv::Point> pattern;

        int nfeatures;
        double scaleFactor;
        int iniThFAST;
        int minThFAST;

        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;


        uint8_t n_ = 12;

        float maxScaleFactor;
        int allocatedSize;
        int allocatedInputSize;

        cudaStream_t cudaStream;
        cudaStream_t cudaStreamCpy;
        cudaStream_t cudaStreamBlur;
        cudaEvent_t resizeComplete;
        cudaEvent_t blurComplete;
        cudaEvent_t interComplete;
        cudaEvent_t filterKernelComplete;

        copyPyrimid_t copyPyrimidData;

        int *umax_gpu;

        uint8_t *d_R;
        uint8_t *d_R_low;

        GpuPoint *corner_buffer;
        uint *corner_size;

        GpuPoint *d_corner_buffer;
        uint *d_corner_size;

        int *d_score;

        int *features;

        int2 *d_centroids;
        int *d_clust_sizes;

        int2 *initial_centroids;
        int2 *centroids;

        int *d_points;
        cv::Point *d_pattern;

        float *kernel;

        //piramidi
        uchar *d_imagesBlured;
        uchar *d_inputImageBlured;
        uchar *outputImages;

        uint *d_mono_index;
        uint *d_stereo_index;

    private:
        void freeMemory();
        void freeInputMemory();
        void checkAndReallocMemory(cv::Mat);
        void allocMemory(int, int, int);
        void allocInputMemory(int, int, int);


    };

} //namespace ORB_SLAM

#endif

