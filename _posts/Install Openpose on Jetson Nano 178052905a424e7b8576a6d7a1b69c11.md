# Install Openpose on Jetson Nano

---

## layout: post
title: Install Openpose on Jetson Nano
tags: [Jetson Nano, OpenPose]

## Giới thiệu

Xin chào mọi người, buổi hôm nay mình sẽ hướng dẫn các cài đặt thư viện OpenPose trên máy tính nhúng Jetson Nano.

OpenPose([https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)) là thư viện hay dùng để phát hiện các keypoint của cơ thể người qua camera 2D. Trong bài viết này mình xin giới thiệu lý thuyết cơ bản của OpenPose cũng như cách cài đặt thư viện này trên máy tính nhúng Jetson Nano.

Ngoài ra các bạn cũng hoàn toàn có thể thử chạy thư viện này ở trên Colab, tại đây:

[https://colab.research.google.com/github/tugstugi/dl-colab-notebooks/blob/master/notebooks/OpenPose.ipynb](https://colab.research.google.com/github/tugstugi/dl-colab-notebooks/blob/master/notebooks/OpenPose.ipynb)

Ở các bài viết sắp tới, mình sẽ chia sẻ thêm các phương pháp khác để detect human pose và sử dụng human pose để nhận biết các hành động.

## Lý thuyết đằng sau OpenPose

Trước đó để detect human keypoints, chúng ta thường cần dùng các camera có sensor hồng ngoại để thu được depth map, từ đó mới có thể nhận diện được. Với OpenPose, chỉ với camera 2D bình thường, thông tin human keypoints có thể nhận diện rất chính xác.

Để tìm hiểu kỹ hơn (nhiều công thức toán hơn) thì các bạn có thể xem chi tiết bài báo tại đây ([https://arxiv.org/abs/1812.08008](https://arxiv.org/abs/1812.08008))

Mình chỉ dám xin tóm tắt các bước của OpenPose như dưới đây:

- Đầu tiên, ảnh đầu vào sẽ được đưa vào mạng neuron gồm 10 layers tương ứng với 10 layers đầu của mô hình VGG-19 (mô hình hay được sử dụng gồm 19 layers). Mục đích là để tạo ra tập hợp feature maps F. (Chính là đầu vào của hình bên dưới)
- Từ đó feature maps F sẽ thành đầu vào của 2 stages, stage 1 có mục đích để dự đoán `Part Affinity Fields (PAF)` gồm tập hợp các toạ độ bộ phận trên cơ thể, và tập hợp các vector nối các phần đó và ở stage 1 này dự đoán sẽ được lặp T_p lần để liên tục hiệu chỉnh lại các thông số của các vector sao cho chính xác hơn. Ở stage 2, có T_c vòng lặp, output sẽ là confidence maps (độ tin cậy của vector sinh ra ở stage 1)
- Ở cuối mỗi vòng lặp stage sẽ có tính toán Loss để tính toán đến khi feature maps và PAF đạt tối ưu.
- Kiến trúc mạng neuron chứa các layer Convolution có kernel 1x1 hoặc 3x3

![openposemulti-CNN.png](Install%20Openpose%20on%20Jetson%20Nano%20178052905a424e7b8576a6d7a1b69c11/openposemulti-CNN.png)

## Cài đặt OpenPose trên Jetson Nano 2

- Đảm bảo version của Cmake ≥ 3.12.2. Check version của Cmake bằng câu lệnh dưới đây:

    ```bash
    cmake --version
    ```

- Cài đặt các thư viện cần thiết (bao gồm cả Cmake nếu như version thấp hơn 3.12.2)

    ```bash
    apt-get update
    apt-get install -y libssl-dev libcurl4-openssl-dev qt5-default
    apt-get install -y build-essential libboost-all-dev libboost-dev libhdf5-dev libatlas-base-dev
    apt-get install -y python3-dev python3-pip
    apt-get remove -y cmake
    wget https://github.com/Kitware/CMake/releases/download/v3.19.4/cmake-3.19.4.tar.gz
    tar -xvzf cmake-3.19.4.tar.gz
    cd cmake-3.19.4
    ./bootstrap --qt-gui
    make -j4
    make install
    ```

- Cài đặt Protocol Buffer

    ```bash
    apt-get install -y libprotobuf-dev protobuf-compiler libgflags-dev libgoogle-glog-dev
    ```

- Tải sourcecode OpenPose về để build

    ```bash
    wget https://github.com/CMU-Perceptual-Computing-Lab/openpose/archive/v1.7.0.tar.gz
    tar -xvzf v1.7.0.tar.gz
    cd openpose-1.7.0/
    cd 3rdparty

    git clone https://github.com/CMU-Perceptual-Computing-Lab/caffe.git
    git clone https://github.com/pybind/pybind11.git
    ```

- Tạo folder build ở sourcecode với `mkdir build` và bật Cmake để tiến hành build.

    Chọn source là folder sourcecode, build là folder build mới tạo rồi configure với tuỳ chọn `Linux Native Compiler`

    Sau khi Configure xong, thì sẽ tick chọn vào `BUILD PYTHON` và bỏ tick ở `USE_CUDNN` như 2 hình dưới đây. Sau đó nhấn Generate để tạo Makefile

    ![cmake1.png](Install%20Openpose%20on%20Jetson%20Nano%20178052905a424e7b8576a6d7a1b69c11/cmake1.png)

    ![cmake2.png](Install%20Openpose%20on%20Jetson%20Nano%20178052905a424e7b8576a6d7a1b69c11/cmake2.png)

- cd vào folder build và chạy các lệnh sau để cài openpose

    ```bash
    make -j4
    make install
    cd python
    make -j4
    ```

Như vậy phần cài đặt đã thành công rồi. Bạn có thể chạy chương trình mẫu tại đây để check OpenPose hoạt động. Trước đó ta cần copy file mới build vào thư mục chứa các modules của python, để có thể import được OpenPose

```bash
cp -r ./build/python/openpose/ /usr/lib/python3.6/dist-packages
```

- Chạy sample bằng lệnh dưới đây:

    ```bash
    ./build/examples/openpose/openpose.bin --video ./examples/media/video.avi  --net_resolution "256x128"
    ```

## Test OpenPose với 2D camera

Các sample của OpenPose không cung cấp code để run với Camera, vì vậy bạn cần viết script để test, bạn có thể dùng script dưới đây:

```python
import logging
import sys
import time
import math
import cv2
import numpy as np
from openpose import pyopenpose as op

if __name__ == '__main__':
    fps_time = 0

    params = dict()
    # Change folder according to your openpose installation folder
    params["model_folder"] = "usr/local/src/openpose/models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    print("OpenPose start")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret_val, img = cap.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))

    count = 0

    if cap is None:
        print("Camera Open Error")
        sys.exit(0)
    while cap.isOpened() and count < 30:
        ret_val, dst = cap.read()
        if ret_val == False:
            print("Camera read Error")
            break
        #dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        #cv2.imshow("OpenPose - Tutorial Python API", dst)
        #continue

        datum = op.Datum()
        datum.cvInputData = dst
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        newImage = datum.cvOutputData[:, :, :]
        cv2.putText(newImage , "FPS: %f" % (fps), (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out_video.write(newImage)

        print("captured fps %f"%(fps))
        cv2.imshow("OpenPose - Tutorial Python API", newImage)
        count += 1

    cv2.destroyAllWindows()
    out_video.release()
    cap.release()
```

## Kết quả

Chạy openpose trên con Nano của mình cho FPS khá chậm ~0.6 fps. 

Mặc dù OpenPose cho kết quả detect khá ngon kể cả khi các bộ phận cơ thể ko hiển thị hết trên camera, nhưng trên các thiết bị nhỏ như điện thoại hay các thiết bị edge khác thì sẽ cần các mô hình nhỏ gọn hơn. Ở các bài viết sắp tới mình sẽ tiến hành tìm hiểu và chạy thử các mô hình khác như PoseNet hay TensorRT trên con Nano này.

## Các tài liệu tham khảo:

- Sách Mastering Computer Vision with TensorFlow2.x
- Setup install OpenPose on Jetson Nano

    [https://spyjetson.blogspot.com/2019/10/jetsonnano-human-pose-estimation-using.html](https://spyjetson.blogspot.com/2019/10/jetsonnano-human-pose-estimation-using.html)