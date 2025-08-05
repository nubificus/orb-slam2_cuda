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

echo "Configuring and building ORB_SLAM2 ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
