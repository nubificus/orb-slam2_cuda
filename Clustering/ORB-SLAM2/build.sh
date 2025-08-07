echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo "Download vocabulary ..."

mkdir -p Vocabulary
cd Vocabulary
if [ -f "ORBvoc.txt" ]; then
  echo "Vocabulary file exists."
else
    wget https://s3.nbfc.io/orb/ORBvoc.txt
fi
cd ..

echo "---Configuring and building ORB_SLAM2---"

mkdir build
cd build
echo "Build GPU executables ..."
echo "cmake .. -DVACCEL=OFF -DCMAKE_BUILD_TYPE=Release"
cmake .. -DVACCEL=OFF -DCMAKE_BUILD_TYPE=Release
make -j
rm -r CMakeFiles CMakeCache.txt cmake_install.cmake Makefile

echo "Build CPU executable ..."
echo "cmake .. -DVACCEL=OFF -DCPUONLY=ON -DCMAKE_BUILD_TYPE=Release"
cmake .. -DVACCEL=OFF -DCPUONLY=ON -DCMAKE_BUILD_TYPE=Release
make -j
rm -r CMakeFiles CMakeCache.txt cmake_install.cmake Makefile

echo "Build CPU executable with VACCEL ..."
echo "cmake .. -DVACCEL=ON -DCPUONLY=ON -DCMAKE_BUILD_TYPE=Release"
cmake .. -DVACCEL=ON -DCPUONLY=ON -DCMAKE_BUILD_TYPE=Release
make -j
rm -r CMakeFiles CMakeCache.txt cmake_install.cmake Makefile

echo "Build GPU executable with VACCEL ..."
echo "cmake .. -DVACCEL=ON -DCPUONLY=OFF -DCMAKE_BUILD_TYPE=Release"
cmake .. -DVACCEL=ON -DCPUONLY=OFF -DCMAKE_BUILD_TYPE=Release
make -j
rm -r CMakeFiles CMakeCache.txt cmake_install.cmake Makefile
