#include "wrapper.hpp"


extern "C" {
    ORB_SLAM2::ORBextractor* mpORBextractor = NULL;

    int my_wrapped_orb_operator(struct vaccel_arg *read, size_t nr_read,
            struct vaccel_arg *write, size_t nr_write)
    {

        cv::Mat image;
        deserialize_mat(read[0].buf,read[0].size,image);

        cv::Mat mask;
        deserialize_mat(read[1].buf,read[1].size,mask);

        int f_id = *reinterpret_cast<int*>(read[2].buf);

        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;

        if (mpORBextractor == NULL) {
            int nFeatures=2000;
            float fScaleFactor= 1.2;
            int nLevels =8;
            int fIniThFAST=12;
            int fMinThFAST=7;
            mpORBextractor = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST,0);
            std::cout << "[WRAPPER] ORBextractor [" << f_id << "] initialized" << std::endl;
        }
        // mpORBextractorLeft->operator()(im, cv::Mat(), mvKeys, mDescriptors);
        (*mpORBextractor)(image,mask,keypoints,descriptors);
        
        vector<cv::Mat> pyr = (mpORBextractor)->mvImagePyramid;

        // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        size_t keypoints_size = keypoints.size();
        write[0].buf = serialize_vec_of_keypoints_new(keypoints, write[0].buf, keypoints_size);
        write[0].size = keypoints_size;
        
        size_t descriptors_size = descriptors.total()*descriptors.elemSize();
        write[1].buf = serialize_mat_new(descriptors, write[1].buf, descriptors_size);
        write[1].size = descriptors_size;

        size_t pyr_size = pyr.size();
        write[2].buf = serialize_vec_of_mat_new(pyr, write[2].buf, pyr_size);
        write[2].size = pyr_size;
       
        // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        // double tser= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        // std::cout << "serialization: " << tser << std::endl;
        // mpORBextractor->~ORBextractor();

        return 0;

    }


}