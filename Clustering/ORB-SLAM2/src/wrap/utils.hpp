#include <iostream>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <cuda_runtime.h>  // Needed for cudaMemcpy

using namespace std;
using namespace cv;

size_t
get_mat_size(const Mat & mat) {
    size_t data_size = mat.isContinuous() ? mat.total() * mat.elemSize() :
	    mat.step[0] * mat.rows;

    return sizeof(int) * 4 + data_size;
}

void
serialize_mat(const Mat & mat, void * buffer) {
    int * header = (int * ) buffer;

    header[0] = mat.rows;
    header[1] = mat.cols;
    header[2] = mat.type();
    header[3] = mat.isContinuous() ? mat.total() * mat.elemSize() : 
	    mat.step[0] * mat.rows; 
    char * data = (char * )((int * ) buffer + 4);

    memcpy(data, mat.data, header[3]);
}

void *serialize_mat_new(Mat mat, void*& buf, size_t& size){
		size_t mat_size = get_mat_size(mat);

        if (!buf){
            buf = malloc(mat_size);
        }

		size = mat_size;
		serialize_mat(mat, buf);
		return buf;
}

void
deserialize_mat(void * buffer, size_t buffer_size, Mat & mat) {
    int * header = (int * ) buffer;
    // Mat myMat;

    mat.create(header[0], header[1], header[2]);
    size_t data_size = header[3]; // * header[0] * header[1];

    memcpy(mat.data, (char*)buffer + sizeof(int) * 4, data_size);
}


size_t
get_vec_of_keypoints_size(const std::vector<KeyPoint> &vec) {
    return sizeof(size_t) + vec.size() * sizeof(cv::KeyPoint);
}

void
serialize_vec_of_keypoints(const std::vector<KeyPoint> &vec, void * buffer) {
    size_t * header = (size_t * ) buffer;

    header[0] = vec.size();
    memcpy((char * ) buffer + sizeof(size_t), vec.data(), vec.size() * sizeof(cv::KeyPoint));
}

void *serialize_vec_of_keypoints_new(const std::vector<KeyPoint> &vec, void*& buf, size_t& size){
		size_t vec_size = get_vec_of_keypoints_size(vec);

        if (!buf){
            buf = malloc(vec_size);
        }

		size = vec_size;
		serialize_vec_of_keypoints(vec, buf);
		return buf;
}

void
deserialize_vec_of_keypoints(void * buffer, size_t buffer_size, std::vector<KeyPoint> &vec) {
    size_t * header = (size_t * ) buffer;

    vec.resize(header[0]);
    memcpy(vec.data(), (char*)buffer + sizeof(size_t), header[0] * sizeof(cv::KeyPoint));
}


size_t get_vec_of_mat_size(const std::vector<cv::Mat>& vec) {
    size_t totalSize = sizeof(int) * (vec.size() + 1); // one int for count, one per mat offset
    for (const auto& mat : vec) {
        totalSize += get_mat_size(mat);
    }
    return totalSize;
}

void serialize_vec_of_mat(const std::vector<cv::Mat>& vec, void* buffer) {
    int* header = reinterpret_cast<int*>(buffer);
    header[0] = static_cast<int>(vec.size());

    // Offsets start right after the count (first int)
    size_t offset = sizeof(int) * (vec.size() + 1);
    char* bufferPtr = reinterpret_cast<char*>(buffer);

    for (size_t i = 0; i < vec.size(); ++i) {
        header[i + 1] = static_cast<int>(offset);
        serialize_mat(vec[i], bufferPtr + offset);
        offset += get_mat_size(vec[i]);
    }
}

void* serialize_vec_of_mat_new(const std::vector<cv::Mat>& vec, void*& buf, size_t& size) {
    size = get_vec_of_mat_size(vec);
    if (!buf) {
        buf = malloc(size);
    }
    serialize_vec_of_mat(vec, buf);
    return buf;
}

void deserialize_vec_of_mat(void* buffer, size_t buffer_size, std::vector<cv::Mat>& vec) {
    int* header = reinterpret_cast<int*>(buffer);
    int numMats = header[0];

    vec.resize(numMats);

    for (int i = 0; i < numMats; ++i) {
        char* matPtr = reinterpret_cast<char*>(buffer) + header[i + 1];
        deserialize_mat(matPtr, buffer_size - header[i + 1], vec[i]);
    }
}
