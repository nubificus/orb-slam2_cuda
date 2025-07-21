/**
* This file is part of Cuda accelerated ORB-SLAM project by Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli and Benjamin Rouxel.
 * Implemented by Filippo Muzzini.
 *
 * Based on ORB-SLAM2 (Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós) and ORB-SLAM3 (Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós)
 *
 * Project under GPLv3 Licence
*
*/



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "fast.h"
#include "orientation.h"
#include "resize.h"
#include "gaussian_blur.h"
#include "descriptor.h"

#include "ORBextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

    void copyPyramid(void *data_) {
        copyPyrimid_t *data = (copyPyrimid_t *)data_;
        for (int level = 1; level < (data->nlevels); ++level) {
            uchar *imageLevel = &(data->outputImages[level*(data->cols)*(data->rows)]);
            float scale = data->mvScaleFactor[level];
            int new_rows = round(data->rows * 1/scale);
            int new_cols = round(data->cols * 1/scale);
            cv::Mat cvImageLevel(new_rows, new_cols, CV_8UC1, imageLevel, sizeof(uchar)*new_cols);
            data->mvImagePyramid[level] = cvImageLevel;
        }
    }

    void generateGaussian(float K[]) {
        const double stdev = SIGMA;
        const double pi = CV_PI;
        const double constant = 1.0 / (2.0 * pi * stdev);

        for (int h = -KH/2; h<=KH/2; h++)
            for (int w = -KW/2; w<=KW/2; w++)
                K[(h + KH/2) * KW + (w + KW/2)] = constant * (1 / exp((pow(h, 2) + pow(w, 2)) / (2 * stdev)));
    }

    void computeCentroids(int &f, int rows, int cols, int2 *centroids){
        const int new_f = ceil(f / 4.0) * 4;
        const float l = sqrt(new_f);
        const int l_o = ceil(l);
        const int l_v  = floor(l);

        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = cols-EDGE_THRESHOLD+3;
        const int maxBorderY = rows-EDGE_THRESHOLD+3;

        const int h = maxBorderY - minBorderY;
        const int w = maxBorderX - minBorderX;

        const float offset_x = (float)w / l_o;
        const float offset_y = (float)h / l_v;


        int ic = 0;
        for (int i=0; i<l_o; i++){
            for (int j=0; j<l_v; j++){
                const float x1 = minBorderX+(i+1)*offset_x;
                const float y1 = minBorderY+(j+1)*offset_y;
                const int cx = round(x1 - offset_x/2);
                const int cy = round(y1 - offset_y/2);
                const int2 c = make_int2(cx, cy);
                centroids[ic] = c;
                ic++;
            }
        }
        f = ic;
    }


    static int bit_pattern_31_[256*4] =
            {
                    8,-3, 9,5/*mean (0), correlation (0)*/,
                    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
                    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
                    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
                    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
                    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
                    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
                    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
                    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
                    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
                    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
                    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
                    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
                    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
                    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
                    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
                    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
                    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
                    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
                    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
                    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
                    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
                    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
                    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
                    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
                    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
                    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
                    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
                    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
                    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
                    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
                    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
                    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
                    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
                    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
                    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
                    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
                    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
                    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
                    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
                    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
                    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
                    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
                    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
                    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
                    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
                    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
                    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
                    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
                    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
                    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
                    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
                    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
                    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
                    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
                    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
                    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
                    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
                    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
                    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
                    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
                    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
                    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
                    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
                    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
                    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
                    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
                    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
                    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
                    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
                    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
                    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
                    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
                    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
                    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
                    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
                    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
                    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
                    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
                    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
                    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
                    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
                    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
                    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
                    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
                    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
                    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
                    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
                    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
                    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
                    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
                    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
                    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
                    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
                    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
                    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
                    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
                    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
                    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
                    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
                    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
                    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
                    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
                    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
                    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
                    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
                    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
                    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
                    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
                    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
                    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
                    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
                    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
                    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
                    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
                    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
                    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
                    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
                    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
                    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
                    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
                    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
                    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
                    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
                    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
                    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
                    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
                    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
                    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
                    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
                    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
                    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
                    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
                    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
                    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
                    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
                    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
                    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
                    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
                    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
                    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
                    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
                    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
                    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
                    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
                    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
                    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
                    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
                    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
                    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
                    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
                    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
                    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
                    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
                    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
                    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
                    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
                    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
                    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
                    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
                    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
                    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
                    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
                    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
                    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
                    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
                    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
                    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
                    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
                    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
                    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
                    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
                    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
                    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
                    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
                    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
                    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
                    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
                    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
                    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
                    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
                    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
                    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
                    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
                    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
                    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
                    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
                    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
                    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
                    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
                    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
                    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
                    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
                    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
                    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
                    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
                    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
                    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
                    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
                    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
                    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
                    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
                    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
                    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
                    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
                    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
                    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
                    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
                    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
                    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
                    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
                    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
                    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
                    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
                    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
                    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
                    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
                    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
                    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
                    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
                    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
                    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
                    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
                    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
                    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
                    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
                    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
                    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
                    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
                    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
                    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
                    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
                    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
                    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
                    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
                    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
                    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
                    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
                    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
                    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
                    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
                    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
                    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
                    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
                    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
                    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
                    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
                    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
                    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
                    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
                    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
                    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
                    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
                    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
                    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
                    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
            };

    ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST):
            nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
            iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0]=1.0f;
        mvLevelSigma2[0]=1.0f;
        maxScaleFactor = 0;
        for(int i=1; i<nlevels; i++)
        {
            float _scaleFactor = mvScaleFactor[i-1]*scaleFactor;
            if (maxScaleFactor < _scaleFactor) {
                maxScaleFactor = _scaleFactor;
            }
            mvScaleFactor[i]= _scaleFactor;
            mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
        }
        if (scaleFactor >= 1){
            maxScaleFactor = 1;
        } else {
            maxScaleFactor = 1/maxScaleFactor;
        }

        int points[32] = {0,  3,  1,  3, 2,  2, 3,  1, 3, 0, 3, -1, 2, -2, 1, -3,
                          0, -3, -1, -3, -2, -2, -3, -1, -3, 0, -3,  1, -2,  2, -1,  3};

        int cuda_device = 0;
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, cuda_device);

        cudaStreamCreateWithPriority(&cudaStream, cudaStreamNonBlocking, 0);
        cudaStreamCreateWithPriority(&cudaStreamCpy, cudaStreamNonBlocking, 2);
        cudaStreamCreateWithPriority(&cudaStreamBlur, cudaStreamNonBlocking, 1);
        cudaEventCreateWithFlags(&resizeComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&blurComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&interComplete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&filterKernelComplete, cudaEventDisableTiming);

        // set_half_patch_size(HALF_PATCH_SIZE, cudaStream);

        cudaMalloc(&d_scaleFactor, sizeof(float)*mvScaleFactor.size());
        cudaMemcpy(d_scaleFactor, mvScaleFactor.data(), sizeof(float)*mvScaleFactor.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&d_points, 32*sizeof(int));
        cudaMemcpyAsync(d_points, points, 32*sizeof(int), cudaMemcpyHostToDevice, cudaStream);
        cudaMalloc(&d_corner_size, sizeof(uint)*nlevels);
        cudaMalloc(&d_mono_index, sizeof(uint));
        cudaMalloc(&d_stereo_index, sizeof(uint));

        float k[KW*KH];
        generateGaussian(k);
        cudaMalloc(&(kernel), sizeof(float)*KW*KH);
        cudaMemcpy(kernel, k, sizeof(float)*KW*KH, cudaMemcpyHostToDevice);

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

        allocMemory(INIT_IMAGE_W, INIT_IMAGE_H, INIT_IMAGE_W);
        allocInputMemory(INIT_IMAGE_W, INIT_IMAGE_H, INIT_IMAGE_W);

        cudaMemcpy(this->features, mnFeaturesPerLevel.data(), sizeof(uint)*mnFeaturesPerLevel.size(), cudaMemcpyHostToDevice);
        const int npoints = 512;
        const Point* pattern0 = (const Point*)bit_pattern_31_;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
        cudaMalloc(&(d_pattern), sizeof(cv::Point)*pattern.size());
        cudaMemcpy(d_pattern, pattern.data(), sizeof(cv::Point)*pattern.size(), cudaMemcpyHostToDevice);

        //This is for orientation
        // pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }

        cudaMalloc(&umax_gpu, sizeof(int)*umax.size());
        cudaMemcpyAsync(umax_gpu, umax.data(), sizeof(int)*umax.size(), cudaMemcpyHostToDevice, cudaStream);
    }

    void ORBextractor::ComputeKeyPointsOctTree(vector<vector<OrbKeyPoint> >& allKeypoints)
    {
        allKeypoints.resize(nlevels);

        fast_extract(d_images, d_inputImage, iniThFAST, minThFAST, d_R, d_R_low, d_points, n_, d_corner_buffer, d_corner_size, cols, rows, imageStep, d_scaleFactor, nlevels, cudaStream, interComplete, this->mvImagePyramid[0]);
//        filter_points(this->d_corner_buffer, this->d_corner_buffer2, this->d_centroids, this->d_clust_sizes, this->d_corner_size, this->features, this->initial_centroids, this->mnFeaturesPerLevel.data(), this->nlevels, cols*rows, this->nfeatures, this->cudaStream);
//        int h_sizes[nlevels];
//        cudaMemcpyAsync(h_sizes, d_corner_size, sizeof(uint)*nlevels, cudaMemcpyDeviceToHost, cudaStream);
//        cudaStreamSynchronize(cudaStream);
//        for (int l=0; l<nlevels; l++) {
//            int size = h_sizes[l];
//            ORB_SLAM2::GpuPoint keypoints[size];
//            cudaMemcpyAsync(keypoints, &d_corner_buffer[rows*cols*l], sizeof(ORB_SLAM2::GpuPoint)*size, cudaMemcpyDeviceToHost, cudaStream);
//            cudaStreamSynchronize(cudaStream);
//            std::vector<cv::KeyPoint> toDraw;
//            for (int i=0; i<size; i++){
//                ORB_SLAM2::GpuPoint p = keypoints[i];
//                cv::KeyPoint kp(p.x, p.y, 7.0, -1, p.score);
//                toDraw.push_back(kp);
//            }
//            cv::Mat imageOut;
//            cv::drawKeypoints(mvImagePyramid[l], toDraw, imageOut);
//            std::ostringstream name;
//            name << "keys" << l << ".png";
//            cv::imwrite(name.str(), imageOut);
//            std::cout << "level " << l << ": " << size << std::endl;
//        }
//        exit(0);
        compute_orientation(d_images, d_inputImage, d_corner_buffer, this->d_corner_size, rows*cols, this->umax_gpu, imageStep, nlevels, cols, rows, d_scaleFactor, cudaStream);
        cudaStreamWaitEvent(cudaStream, blurComplete, 0);
        compute_descriptor(d_imagesBlured, d_inputImageBlured, d_corner_buffer, d_corner_size, rows*cols, d_pattern, imageStep, nlevels, cols, rows, d_scaleFactor, cudaStream);
        uint corner_size[nlevels];
        cudaMemcpyAsync(corner_size, this->d_corner_size, sizeof(uint)*nlevels, cudaMemcpyDeviceToHost, cudaStream);
        cudaStreamSynchronize(cudaStream);

        for (int level = 0; level < nlevels; ++level){
            uint size = corner_size[level];
            ORB_SLAM2::GpuPoint *corner_buffer = &(this->corner_buffer[level*rows*cols]);
            ORB_SLAM2::GpuPoint *d_corner_buffer = &(this->d_corner_buffer[level*rows*cols]);
            cudaMemcpyAsync(corner_buffer, d_corner_buffer, sizeof(GpuPoint)*size, cudaMemcpyDeviceToHost, cudaStream);
        }
        cudaStreamSynchronize(cudaStream);

        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;

        for (int level = 0; level < nlevels; ++level)
        {
            vector<OrbKeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures*10);

            float scale = mvScaleFactor[level];
            int new_rows = round(rows * 1/scale);
            int new_cols = round(cols * 1/scale);
            const int maxBorderX = new_cols-EDGE_THRESHOLD+3;
            const int maxBorderY = new_rows-EDGE_THRESHOLD+3;

            uint size = corner_size[level];
            ORB_SLAM2::GpuPoint *corner_buffer = &(this->corner_buffer[level*rows*cols]);

            for(uint i=0; i<size; i++)
            {
                // cout << size << " " << i << endl;
                const float x = corner_buffer[i].x;
                const float y = corner_buffer[i].y;
                const float score = corner_buffer[i].score;
                const float size = corner_buffer[i].size;
                const float angle = corner_buffer[i].angle;
                const int octave = corner_buffer[i].octave;
                OrbKeyPoint orbKeypoint;
                orbKeypoint.point = cv::KeyPoint(x, y, size, angle, score, octave);
                orbKeypoint.descriptor = corner_buffer[i].descriptor;
                vToDistributeKeys.push_back(orbKeypoint);
            }

            // cv::Mat out;
            // cv::drawKeypoints(mvImagePyramid[level], vToDistributeKeys, out);
            // std::ostringstream name;
            // name << "parrallel_" << level << ".png";
            // cv::imwrite(name.str(), out);

            vector<OrbKeyPoint> & keypoints = allKeypoints[level];
            keypoints.reserve(nfeatures);

            keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX+minBorderX,
                                          minBorderY, maxBorderY+minBorderY,mnFeaturesPerLevel[level], level);

//            struct {
//                bool operator()(OrbKeyPoint &a, OrbKeyPoint &b) const {
//                    if (a.point.pt.x == b.point.pt.x) {
//                        return a.point.pt.y < b.point.pt.y;
//                    }
//                    return a.point.pt.x < b.point.pt.x;
//                }
//            } customLess;
//            std::sort(vToDistributeKeys.begin(), vToDistributeKeys.end(), customLess);
//            for (int i=0; i<vToDistributeKeys.size(); i++){
//                std::cout << vToDistributeKeys[i].point.pt.x << " " << vToDistributeKeys[i].point.pt.y << " " << vToDistributeKeys[i].point.response << std::endl;
//            }
//            exit(0);

        }
//        //        exit(0);
//        int sizes[this->nlevels];
//        cudaMemcpyAsync(&sizes, this->features, sizeof(uint)* this->nlevels, cudaMemcpyDeviceToHost, cudaStream);
//        cudaStreamSynchronize(cudaStream);
//        for(int l=0; l<this->nlevels; l++) {
//            int size = sizes[l];
//            cout << "level: " << l <<  " size: " << size << endl;
//            ORB_SLAM2::GpuPoint keypoints[size];
//            cudaMemcpyAsync(keypoints, &d_corner_buffer[l*rows*cols], sizeof(ORB_SLAM2::GpuPoint) * size, cudaMemcpyDeviceToHost,
//                            cudaStream);
//            cudaStreamSynchronize(cudaStream);
//            struct {
//                bool operator()(ORB_SLAM2::GpuPoint &a, ORB_SLAM2::GpuPoint &b) const {
//                    if (a.x == b.x) {
//                        return a.y < b.y;
//                    }
//                    return a.x < b.x;
//                }
//            } customLess;
//            std::sort(keypoints, keypoints + size, customLess);
//            std::vector<cv::KeyPoint> toDraw;
//            std::vector<uchar *> descriptors;
//            for (int i = 0; i < size; i++) {
//                ORB_SLAM2::GpuPoint &p = keypoints[i];
//                cv::KeyPoint kp(p.x, p.y, p.size, p.angle, p.score);
//                toDraw.push_back(kp);
//                descriptors.push_back(p.descriptor);
//            }
//            cv::Mat imageOut;
//            cv::drawKeypoints(this->mvImagePyramid[0], toDraw, imageOut);
//            std::ostringstream name;
//            name << "keys" << 0 << ".png";
//            cv::imwrite(name.str(), imageOut);
//            for (int i = 0; i < toDraw.size(); i++) {
//                cout << toDraw[i].pt.x << " " << toDraw[i].pt.y << " " << toDraw[i].size << " " << toDraw[i].response
//                     << " " << toDraw[i].angle << " desc: ";
//                for (int j = 0; j < 32; j++) {
//                    cout << +descriptors[i][j] << " ";
//                }
//                cout << endl;
//            }
//        }
    }

    void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
        const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

        //Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x+halfX,UL.y);
        n1.BL = cv::Point2i(UL.x,UL.y+halfY);
        n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x,UL.y+halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x,BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        //Associate points to childs
        for(size_t i=0;i<vKeys.size();i++)
        {
            const OrbKeyPoint &kp = vKeys[i];
            if(kp.point.pt.x<n1.UR.x)
            {
                if(kp.point.pt.y<n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if(kp.point.pt.y<n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        if(n1.vKeys.size()==1)
            n1.bNoMore = true;
        if(n2.vKeys.size()==1)
            n2.bNoMore = true;
        if(n3.vKeys.size()==1)
            n3.bNoMore = true;
        if(n4.vKeys.size()==1)
            n4.bNoMore = true;

    }

    static bool compareNodes(pair<int,ExtractorNode*>& e1, pair<int,ExtractorNode*>& e2){
        if(e1.first < e2.first){
            return true;
        }
        else if(e1.first > e2.first){
            return false;
        }
        else{
            if(e1.second->UL.x < e2.second->UL.x){
                return true;
            }
            else{
                return false;
            }
        }
    }

    vector<OrbKeyPoint> ORBextractor::DistributeOctTree(const std::vector<OrbKeyPoint>& vToDistributeKeys, const int &minX,
                                                        const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
    {
        // Compute how many initial nodes
        const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));
        const float hX = static_cast<float>(maxX-minX)/nIni;

        list<ExtractorNode> lNodes;

        vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);

        for(int i=0; i<nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
            ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
            ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
            ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        //Associate points to childs
        for(size_t i=0;i<vToDistributeKeys.size();i++)
        {
            const OrbKeyPoint &kp = vToDistributeKeys[i];
            vpIniNodes[kp.point.pt.x/hX]->vKeys.push_back(kp);
        }

        list<ExtractorNode>::iterator lit = lNodes.begin();

        while(lit!=lNodes.end())
        {
            if(lit->vKeys.size()==1)
            {
                lit->bNoMore=true;
                lit++;
            }
            else if(lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        bool bFinish = false;

        int iteration = 0;

        vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size()*4);

        while(!bFinish)
        {
            iteration++;

            int prevSize = lNodes.size();

            lit = lNodes.begin();

            int nToExpand = 0;

            vSizeAndPointerToNode.clear();

            while(lit!=lNodes.end())
            {
                if(lit->bNoMore)
                {
                    // If node only contains one point do not subdivide and continue
                    lit++;
                    continue;
                }
                else
                {
                    // If more than one point, subdivide
                    ExtractorNode n1,n2,n3,n4;
                    lit->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit=lNodes.erase(lit);
                    continue;
                }
            }

            // Finish if there are more nodes than required features
            // or all nodes contain just one point
            if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
            {
                bFinish = true;
            }
            else if(((int)lNodes.size()+nToExpand*3)>N)
            {

                while(!bFinish)
                {
                    prevSize = lNodes.size();

                    vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end(),compareNodes);
                    for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                    {
                        ExtractorNode n1,n2,n3,n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                        // Add childs if they contain points
                        if(n1.vKeys.size()>0)
                        {
                            lNodes.push_front(n1);
                            if(n1.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n2.vKeys.size()>0)
                        {
                            lNodes.push_front(n2);
                            if(n2.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n3.vKeys.size()>0)
                        {
                            lNodes.push_front(n3);
                            if(n3.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n4.vKeys.size()>0)
                        {
                            lNodes.push_front(n4);
                            if(n4.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                        if((int)lNodes.size()>=N)
                            break;
                    }

                    if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize){
                        bFinish = true;
                    }


                }
            }
        }

        // Retain the best point in each node
        vector<OrbKeyPoint> vResultKeys;
        vResultKeys.reserve(nfeatures);
        for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
        {
            vector<OrbKeyPoint> &vNodeKeys = lit->vKeys;
            OrbKeyPoint* pKP = &vNodeKeys[0];
            float maxResponse = pKP->point.response;

            for(size_t k=1;k<vNodeKeys.size();k++)
            {
                if(vNodeKeys[k].point.response>maxResponse)
                {
                    pKP = &vNodeKeys[k];
                    maxResponse = vNodeKeys[k].point.response;
                }
            }

            vResultKeys.push_back(*pKP);
        }

        return vResultKeys;
    }

    void ORBextractor::freeMemory() {
        cudaFree(this->d_R);
        cudaFree(this->d_R_low);

        cudaFree(this->d_corner_buffer);
        cudaFree(this->d_corner_buffer2);
        cudaFreeHost(this->corner_buffer);

        cudaFree(this->d_centroids);
        cudaFree(this->d_clust_sizes);
        cudaFree(this->initial_centroids);

        cudaFree(d_images);
        cudaFree(d_imagesBlured);
        cudaFreeHost(outputImages);

        cudaFree(this->features);

        this->allocatedSize = 0;
    }

    void ORBextractor::freeInputMemory() {
        cudaFree(d_inputImage);
        cudaFree(d_inputImageBlured);
        this->allocatedInputSize = 0;
    }

    void ORBextractor::allocMemory(int w, int h, int imageStep) {
        cudaMalloc(&(this->d_R), sizeof(uint8_t)*w*h*this->nlevels);
        cudaMalloc(&(this->d_R_low), sizeof(uint8_t)*w*h*this->nlevels);

        cudaMalloc(&(this->d_corner_buffer), sizeof(GpuPoint)*w*h*this->nlevels);
        cudaMalloc(&(this->d_corner_buffer2), sizeof(GpuPoint)*w*h*this->nlevels);
        cudaMallocHost(&(this->corner_buffer), sizeof(GpuPoint)*w*h*this->nlevels);

        cudaMalloc(&(this->features), sizeof(uint)*this->nlevels);

        cudaMalloc(&d_images, sizeof(uchar)*w*h*nlevels);
        cudaMalloc(&d_imagesBlured, sizeof(uchar)*w*h*nlevels);
        cudaMallocHost(&outputImages, sizeof(uchar)*w*h*nlevels);

        cudaMalloc(&(this->d_centroids), sizeof(int2)*nlevels*nfeatures);
        cudaMalloc(&(this->d_clust_sizes), sizeof(int)*nlevels*nfeatures);

        // int2 centroids[nlevels*nfeatures];
        cudaMallocHost(&(this->centroids), sizeof(int2)*nlevels*nfeatures);
        for (int i=0; i<nlevels; i++){
            int &f = this->mnFeaturesPerLevel[i];
            float scale = mvScaleFactor[i];
            int new_rows = round(h * 1/scale);
            int new_cols = round(w * 1/scale);
            computeCentroids(f, new_rows, new_cols, &(centroids[i*nfeatures]));
        }
        cudaMalloc(&(this->initial_centroids), sizeof(int2)*nlevels*nfeatures);
        cudaMemcpy(this->initial_centroids, centroids, sizeof(int2)*nlevels*nfeatures, cudaMemcpyHostToDevice);

        cudaMemcpy(this->features, this->mnFeaturesPerLevel.data(), this->mnFeaturesPerLevel.size()*sizeof(uint), cudaMemcpyHostToDevice);

        this->allocatedSize = w*h;
    }

    void ORBextractor::allocInputMemory(int w, int h, int imageStep) {
        cudaMalloc(&d_inputImage, sizeof(uchar)*h*imageStep);
        cudaMalloc(&d_inputImageBlured, sizeof(uchar)*h*imageStep);

        this->allocatedInputSize = imageStep*h;
    }

    inline void ORBextractor::checkAndReallocMemory(cv::Mat image) {
        //compute the max scaled image
        int new_cols = cvRound((float)image.cols*maxScaleFactor);
        int new_rows = cvRound((float)image.rows*maxScaleFactor);
        if (this->allocatedSize < new_cols*new_rows) {
            this->freeMemory();
            this->allocMemory(new_cols, new_rows, image.step[0]);
        }
        if (this->allocatedInputSize < new_rows*image.step[0]) {
            this->freeInputMemory();
            this->allocInputMemory(new_cols, new_rows, image.step[0]);
        }
    }

    void ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors)
    {
        //cout << "[ORBextractor]: Max Features: " << nfeatures << endl;
        if(_image.empty())
            return;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );

        this-> cols = image.cols;
        this->rows = image.rows;
        this->imageStep = image.step[0];

        this->checkAndReallocMemory(image);

        // Pre-compute the scale pyramid
        ComputePyramid(image);

        vector < vector<OrbKeyPoint> > allKeypoints;
        ComputeKeyPointsOctTree(allKeypoints);

        Mat descriptors;

        int nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints += (int)allKeypoints[level].size();
        if( nkeypoints == 0 )
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 32, CV_8U);
            descriptors = _descriptors.getMat();
        }

        _keypoints.clear();
        _keypoints.reserve(nkeypoints);
//        _keypoints = vector<cv::KeyPoint>(nkeypoints);

        int offset = 0;
        //Modified for speeding up stereo fisheye matching
        for (int level = 0; level < nlevels; ++level)
        {
            vector<OrbKeyPoint>& keypoints = allKeypoints[level];
            int nkeypointsLevel = (int)keypoints.size();

            if(nkeypointsLevel==0)
                continue;

            // preprocess the resized image
            // Mat workingMat = mvImagePyramid[level].clone();
            // GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

            // Compute the descriptors
            //Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
            // Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
            // computeDescriptors(workingMat, keypoints, desc, pattern);


            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<OrbKeyPoint>::iterator keypoint = keypoints.begin(),
                         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){
                cv::Mat desc(1, 32, CV_8U, (*keypoint).descriptor);

                cv::KeyPoint kp = (*keypoint).point;
                kp.pt.x *= scale;
                kp.pt.y *= scale;
                _keypoints.push_back(kp);

                desc.row(0).copyTo(descriptors.row(offset));

                offset++;
            }
        }

    }

    void ORBextractor::CopyKeyAndDescriptor(vector<KeyPoint>& _keypoints, OutputArray _descriptors) {
        uint nkeypoints = 0;

        for (int level = 0; level < nlevels; ++level){
            uint size = mnFeaturesPerLevel[level];
            ORB_SLAM2::GpuPoint *corner_buffer = &(this->corner_buffer[nkeypoints]);
            ORB_SLAM2::GpuPoint *d_corner_buffer = &(this->d_corner_buffer[level*rows*cols]);
            cudaMemcpyAsync(corner_buffer, d_corner_buffer, sizeof(GpuPoint)*size, cudaMemcpyDeviceToHost, cudaStream);
            nkeypoints += size;
        }
        cudaStreamSynchronize(cudaStream);

        Mat descriptors;

        if( nkeypoints == 0 )
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 32, CV_8U);
            descriptors = _descriptors.getMat();
        }

        _keypoints.reserve(nkeypoints);
        cudaStreamSynchronize(cudaStream);
        int i = 0;
        for (ORB_SLAM2::GpuPoint *point = corner_buffer,
                     *pointEnd = corner_buffer + nkeypoints; point != pointEnd; ++point){
            cv::Mat desc(1, 32, CV_8U, (*point).descriptor);
//            cout << desc << endl;

            cv:KeyPoint keypoint(point->x, point->y, point->size, point->angle, point->score, point->octave);
            _keypoints.push_back(keypoint);
            desc.row(0).copyTo(descriptors.row(i));
            i++;
        }


    }

    void ORBextractor::ComputePyramid(cv::Mat image)
    {
        cudaMemcpyAsync(d_inputImage, image.data, sizeof(uchar)*image.rows*image.step[0], cudaMemcpyHostToDevice, cudaStream);
        resize(image.rows, image.cols, d_scaleFactor, d_inputImage, d_images, nlevels, image.step[0], cudaStream);
        cudaEventRecord(resizeComplete, cudaStream);

        //BLUR
        cudaStreamWaitEvent(cudaStreamBlur, resizeComplete, 0);
        gaussian_blur(d_images, d_inputImage, d_imagesBlured, d_inputImageBlured, kernel, cols, rows, imageStep, d_scaleFactor, nlevels, cudaStreamBlur);
        cudaEventRecord(blurComplete, cudaStreamBlur);

        cudaStreamWaitEvent(cudaStreamCpy, resizeComplete, 0);
        cudaMemcpyAsync(outputImages, d_images, sizeof(uchar)*image.cols*image.rows*nlevels, cudaMemcpyDeviceToHost, cudaStreamCpy);
        copyPyrimidData.cols = image.cols;
        copyPyrimidData.rows = image.rows;
        copyPyrimidData.mvImagePyramid = mvImagePyramid.data();
        copyPyrimidData.mvScaleFactor = mvScaleFactor.data();
        copyPyrimidData.nlevels = nlevels;
        copyPyrimidData.outputImages = outputImages;
        cudaLaunchHostFunc(cudaStreamCpy, copyPyramid, &(this->copyPyrimidData));

        mvImagePyramid[0] = image;
    }

    ORBextractor::~ORBextractor() {
        this->freeMemory();
        this->freeInputMemory();
        cudaFree(d_scaleFactor);
        cudaFree(this->d_points);
        cudaFree(this->d_corner_size);
        cudaFree(this->umax_gpu);
        cudaFree(kernel);
        cudaFree(d_pattern);
        cudaFree(d_mono_index);
        cudaFree(d_stereo_index);
        cudaStreamDestroy(cudaStream);
        cudaStreamDestroy(cudaStreamCpy);
        cudaStreamDestroy(cudaStreamBlur);
        cudaEventDestroy(resizeComplete);
        cudaEventDestroy(blurComplete);
        cudaEventDestroy(interComplete);
        cudaEventDestroy(filterKernelComplete);
    }

} //namespace ORB_SLAM
