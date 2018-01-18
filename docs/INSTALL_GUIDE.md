### Install Keras with GPU support

1. Install [CUDA](https://developer.nvidia.com/cuda-toolkit) v8.0 - Keras only support v8.0, so make sure you pick the right version
2. Install [cuDNN](https://developer.nvidia.com/cudnn) v6.0 - Same remark as above :
 * Copy the cudnn64_5.dll, cudnn.h and cudnn.lib to the following locations (Default %ROOT% : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0):
  * %ROOT%\bin\cudnn64_5.dll
  * %ROOT%\include\cudnn.h
  * %ROOT%\lib\x64\cudnn.lib
3. The simplest way to have a clean install of tensorflow-gpu is to use Anaconda :
  ```bash
  conda create -n tfgpu python=3.5
  conda activate tfgpu
  pip install --ignore-installed --upgrade tensorflow-gpu
  ```
4. Install Keras :
 ```bash
 pip install keras
 ```

If it doesn't work, this [dependencies check script](https://gist.github.com/mrry/ee5dbcfdd045fa48a27d56664411d41c) may help
