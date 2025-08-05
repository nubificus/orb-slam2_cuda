#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <ORBextractor.h>

#include <vaccel.h>

#include "utils.hpp"
#include "../include/ORBextractor.h"
#include "../include/Frame.h"



extern "C" {

int my_wrapped_track_stereo(struct vaccel_arg *read, size_t nr_read,
			struct vaccel_arg *write, size_t nr_write);

}