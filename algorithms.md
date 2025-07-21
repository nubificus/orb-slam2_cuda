# CUDA accelerated ORB-SLAM

**Authors:** Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli, Benjamin Rouxel.

If you use the **NoClustering** version of this software in an academic work, please cite:

Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli, and Benjamin Rouxel. 2023. Brief Announcement: Optimized GPU-accelerated Feature Extraction
for ORB-SLAM Systems. In Proceedings of the 35th ACM Symposium on
Parallelism in Algorithms and Architectures (SPAA ’23). https://doi.org/10.1145/3558481.3591310

```
    @inproceedings{Cuda-ORB-SLAM,
      title={Brief Announcement: Optimized GPU-accelerated Feature Extraction for ORB-SLAM Systems},
      author={Muzzini, Filippo AND Capodieci, Nicola AND Cavicchioli, Roberto AND Rouxel, Benjamin},
      year = {2023},
      isbn = {978-1-4503-9545-8/23/06},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3558481.3591310},
      doi = {10.1145/3558481.3591310},
      location = {Orlando, FL, USA},
      series = {SPAA '23}
      booktitle = {Proceedings of the 35th ACM Symposium on Parallelism in Algorithms and Architectures},
     }
```

If you use the **Clustering** version of this software in an academic work, please cite:

Filippo Muzzini, Nicola Capodieci, Roberto Cavicchioli, and Benjamin Rouxel. 2024. High-Performance Feature Extraction
for GPU-accelerated ORB-SLAMx. In Proceedings of Design, Automation & Test in Europe Conference & Exhibition (DATE).
To appear.

Cuda accelerated ORB-SLAM is an ORB-SLAM2/3 implementation that exploits CUDA to accelerate the execution time.
In the **NoClustering** version CUDA is exploited to parallelize the ORB-SLAM tracking part using also streams and events to perform concurrent tasks. Moreover, a novel more efficient image Pyramid construction is implemented. It is integrated into both ORB-SLAM2 and ORB-SLAM3.

The **Clustering** version introduce a novel algorithm for the point filter phase. This algorithm is accelerated on GPU. In the classic version (and also in **NoClustering** verison) point filter is performed using an Octree algorithm that are not suitable for GPU parallelization.

We provide examples to run the integration on ORB-SLAM3 in the [EuRoC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) 
and the integration on ORB-SLAM2 in the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

This software is based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) (developed by Raul Mur-Artal, Juan D. Tardos, J. M. M. Montiel and Dorian Galvez-Lopez), [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (developed by Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, Juan D. Tardos) and http://yunchih.github.io/ORB-SLAM2-GPU2016-final/ for GPU implementation of FAST algorithm.

### Related Publications:

[ORB-SLAM3] Campos, Carlos, et al. "Orb-slam3: An accurate open-source library for visual, visual–inertial, and multimap slam." IEEE Transactions on Robotics 37.6 (2021): 1874-1890.

[ORB-SLAM2] Mur-Artal, Raul, and Juan D. Tardós. "Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras." IEEE transactions on robotics 33.5 (2017): 1255-1262.


# 1. License

This software is released under [GPLv3 license](https://github.com/UZ-SLAMLab/ORB_SLAM3/LICENSE).
For a list of all code/library dependencies (and associated licenses), please see the README of [ORB-SLAM2](./ORB-SLAM2/README.md) and [ORB-SLAM3](./ORB-SLAM3/README.md). 

# 2. Prerequisites

The prerequisites are the same of ORB-SLAM2 and ORB-SLAM3 projects. Moreover, it is reuqired a CUDA capable device and the CUDA Toolkit.

## ORB-SLAM2/3 Prerequisites

[Pangolin](https://github.com/stevenlovegrove/Pangolin)

[OpenCV](http://opencv.org)

[Eigen3](http://eigen.tuxfamily.org)


## CUDA Toolkit

We tested our implementation using CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit) version 11.
But it should work also with version 10.

# 3. Building ORB-SLAM3 library and examples

Clone the repository:
```
git clone https://git.hipert.unimore.it/fmuzzini/cuda-accelerated-orb-slam
```

Go to the directory you want to build (ORB-SLAM2 or ORB-SLAM3)
```
cd cuda-accelerated-orb-slam
cd ./{Clustering|NoClustering}/ORB_SLAM{2|3}
```

run the build script
```
./build.sh
```

this will compile the ORB-SLAM using our CUDA accelerated implementation


# 5. ORB-SLAM3 with EuRoC Examples
[EuRoC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) was recorded with two pinhole cameras and an inertial sensor.
In the ORB-SLAM3 Example folder there is an example script to launch EuRoC sequences in all the sensor configurations.

1. Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

2. Run the example as follows
```
./Examples/<Monocular|Stereo>/<mono|stereo>_euroc ./Vocabulary/ORBvoc.txt ./Examples/<Monocular|Stereo>/EuRoC.yaml <pathDatasetEuroc>/MH01 ./Examples/Monocular/EuRoC_TimeStamps/MH01.txt
```
to evaluate the result run the following script
```
python2 evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt f_dataset-MH01_<stereo|mono>.txt --plot MH01_stereo.pdf
```

# 6. ORB-SLAM2 with KITTI Dataset

1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php

2. Execute the following command. Change `KITTIX.yaml`by KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11.
```
./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```
to evaluate the result you can use the standard KITTI evaluation tool or other tools like https://github.com/Huangying-Zhan/DF-VO

# 7. Running time analysis
## ORB-SLAM3
A flag in `include\Settings.h` activates time measurements. It is necessary to uncomment the line `#define REGISTER_TIMES` to obtain the time stats of one execution which is shown at the terminal and stored in a text file(`ExecTimeMean.txt`).

## ORB-SLAM2
The ORB-SLAM2 prints the times on screen by default

