#!/bin/bash
# 
# 2018.09.01
# Hong-Yu Lee

build_pkg=$HOME/build
pkg_name="tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl"
workspace=$HOME/elastic-dev/tensorflow
folder_name=$(date +%y%m%d)

cd workspace

echo "[INFO] Build pip packages.."
bazel-bin/tensorflow/tools/pip_package/build_pip_package $build_pkg/$folder_name

echo "[INFO] Uninstall old tensorflow.."
pip uninstall -y tensorflow

echo "[INFO] Install tensorflow from $folder_name"
pip install $build_pkg/$folder_name/$pkg_name
