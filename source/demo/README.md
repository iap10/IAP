현재 directory 내부의 파일들은 demo에 사용된 case에 대한 python code들을 담고 있습니다.

code를 다운받으신 후, demo python code 내부에서 engine 및 library 주소를 다음과 같이 설정해주시면 됩니다.

demo python code의 경우, case code와 달리 빠른 동작을 위해 FPS 측정 부분이 빠져있습니다.

#### DEMO의 ENGINE 및 LIBRARY 환경

| 항목                  | 값                       |
|-----------------------|--------------------------|
| PLUGIN_LIBRARY        | 416100.so                |
| ENGINE_FILE_PATH      | yolov7_416100.engine     |
| HEADPOSE_ENGINE_PATH  | mobilenetv3_small.onnx   |

위의 code를 실행하기 위해서는 onnxruntime을 설치하여야합니다.
Jetson Nano 
#### Jetson Nano 시스템 버전

| 항목                | 값                          |
|---------------------|-----------------------------|
| Model               | NVIDIA Jetson Nano Dev Kit  |
| JetPack Version     | 4.6.1                        |
| L4T Version         | 32.7.1                       |
| CUDA Version        | 10.2                         |
| Python              | 3.6.9                        |

저희는 Jetson Nano의 setting에 맞게 아래의 방식으로 onnxruntime을 설치하였습니다.

```
wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl

```
