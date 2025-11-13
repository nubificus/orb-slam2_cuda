#include "wrapper.hpp"


extern "C" {
    ORB_SLAM2::ORBextractor* mpORBextractor = NULL;

    int my_wrapped_orb_operator(struct vaccel_arg *read, size_t nr_read,
            struct vaccel_arg *write, size_t nr_write)
    {

        cv::Mat image;
        // deserialize_mat(read[0].buf,read[0].size,image);
        image.create(*reinterpret_cast<int*>(read[0].buf), *reinterpret_cast<int*>(read[1].buf), *reinterpret_cast<int*>(read[2].buf));
        image.data = (uchar*)read[4].buf;

        cv::Mat mask;
        // deserialize_mat(read[1].buf,read[1].size,mask);
        mask.create(*reinterpret_cast<int*>(read[5].buf), *reinterpret_cast<int*>(read[6].buf), *reinterpret_cast<int*>(read[7].buf));
        mask.data = (uchar*)read[9].buf;

        int f_id = *reinterpret_cast<int*>(read[10].buf);

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
        // write[0].buf = serialize_vec_of_keypoints_new(keypoints, write[0].buf, keypoints_size);
        // write[0].size = keypoints_size;
        int args_cnt = -1;
        
        // write[++args_cnt].size = sizeof(int);
        *(int*)write[++args_cnt].buf = keypoints_size;
        write[++args_cnt].size = keypoints_size*sizeof(cv::KeyPoint);
        memcpy(write[args_cnt].buf, keypoints.data(), keypoints_size*sizeof(cv::KeyPoint));

        size_t descriptors_size = descriptors.total()*descriptors.elemSize();
        // write[1].buf = serialize_mat_new(descriptors, write[1].buf, descriptors_size);
        // write[1].size = descriptors_size;

        // Mat
        // write[++args_cnt].size = sizeof(int);
        *(int*)write[++args_cnt].buf = descriptors.rows;
        // write[++args_cnt].size = sizeof(int);
        *(int*)write[++args_cnt].buf = descriptors.cols;
        // write[++args_cnt].size = sizeof(int);
        uint t = descriptors.type();
        *(int*)write[++args_cnt].buf = t;
        // write[++args_cnt].size = sizeof(int);
        *(int*)write[++args_cnt].buf = descriptors_size;
        write[++args_cnt].size = descriptors_size;
        memcpy(write[args_cnt].buf, descriptors.data, descriptors_size);

        // std::cout << "[WRAPPER]: " << *reinterpret_cast<uint*>(write[2].buf) << "\t"
        //     << *reinterpret_cast<uint*>(write[3].buf) << "\t"
        //     << *reinterpret_cast<uint*>(write[4].buf) << std::endl;
            
        size_t pyr_size = pyr.size();
        // write[2].buf = serialize_vec_of_mat_new(pyr, write[2].buf, pyr_size);
        // write[2].size = pyr_size;
        // Vec of Mat

        // write[++args_cnt].size = sizeof(int);
        *(int*)write[++args_cnt].buf = pyr_size;
        for (int i = 0; i < pyr.size(); i ++) {
            // write[++args_cnt].size = sizeof(int);
            *(int*)write[++args_cnt].buf = pyr[i].rows;
            // write[++args_cnt].size = sizeof(int);
            *(int*)write[++args_cnt].buf = pyr[i].cols;
            // write[++args_cnt].size = sizeof(int);

            uint t = pyr[i].type();
            *(int*)write[++args_cnt].buf = t;
            // write[++args_cnt].size = sizeof(int);
            size_t sz = pyr[i].total()*pyr[i].elemSize();

            *(int*)write[++args_cnt].buf = sz;
            write[++args_cnt].size = sz;
            memcpy(write[args_cnt].buf, pyr[0].data, sz);
        }
        
        
        // std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        // double tser= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        // std::cout << "serialization: " << tser << std::endl;
        // mpORBextractor->~ORBextractor();

        return 0;

    }


}