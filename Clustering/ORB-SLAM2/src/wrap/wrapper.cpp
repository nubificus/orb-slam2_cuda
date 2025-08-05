#include "wrapper.hpp"
#include "../include/System.h"


extern "C" {

int my_wrapped_track_stereo(struct vaccel_arg *read, size_t nr_read,
                            struct vaccel_arg *write, size_t nr_write)
{
    cv::Mat imLeft, imRight;
    double timestamp;

    deserialize_mat(read[0].buf, read[0].size, imLeft);
    deserialize_mat(read[1].buf, read[1].size, imRight);
    memcpy(&timestamp, read[2].buf, sizeof(double));

    static ORB_SLAM2::System* mpSLAM = nullptr;
    static std::mutex slam_mutex;
    std::lock_guard<std::mutex> lock(slam_mutex);

    if (!ORB_SLAM2::gSLAM) {
        fprintf(stderr, "[WRAPPER] gSLAM is null!\n");
        return 1;
    }

    cv::Mat pose = ORB_SLAM2::gSLAM->TrackStereo(imLeft, imRight, timestamp);

    if (pose.empty()) {
        fprintf(stderr, "[VACCEL WRAPPER] Pose is empty (tracking might have failed)\n");
        return 1;
    }

    size_t pose_size;
    write[0].buf = serialize_mat_new(pose, write[0].buf, pose_size);
    write[0].size = pose_size;

    return 0;
}

}