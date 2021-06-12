# Real-time-object-detection

In this project, we focus on the problem of computation time as well as the elimination of false positives and false negatives which remarkably deteriorates the accuracy of the object detection process.

## Yolo

YOLOv3 (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. Since it prooved its efficiency Yolo architecture and algorithm is used in this project.

## Research Paper
- Link : [YOLOv3: An Incremental Improvement -Joseph Redmon, Ali Farhadi](https://pjreddie.com/media/files/papers/YOLOv3.pdf "Yolov3")
- Website : [pjreddie](https://pjreddie.com/ "pjreddie")




## Architecture
![alt text](https://miro.medium.com/max/3802/1*d4Eg17IVJ0L41e7CTWLLSg.png "Yolov3")

The idea of using tuplets, lists, strings is inspired by [Persson](https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ "Persson")'s implementation under [MIT license](https://en.wikipedia.org/wiki/MIT_License "MIT license").

## Requirements
- Pytorch
- Albumentations
- Numpy
- Pandas
- Pillow
- Tqdm
- Matplotlib
- Opencv
- Scikit-Learn

## Results
| Dataset  | Pascal VOC  |  COCO |
| ------------ | :------------: | :------------: |
| Trained Model  | [Download](https://drive.google.com/file/d/1vYFwiRIN7qhQgPcV5A5fNcShExUS7oFz/view?usp=sharing)  |  [Download](https://drive.google.com/file/d/1bLBbahxRvmw8HcbMJ2OaBjtULSbQm9gW/view?usp=sharing) |
| Demo Video / 0.87 Conf | [Watch](https://drive.google.com/file/d/1E6HDOAlPVNL1hCpzuNA1U-CkrlYQYc7B/view?usp=sharing)  |  [Watch](https://drive.google.com/file/d/1zkoUaeKSgLFcryWNxGxJCwvsYQdESRDs/view?usp=sharing) |
| Class Accuracy  | 88.07 %  |  75.13 % |
| Object Accuracy  |  67.51 % |  57.06 % |
| No Object Accuracy  | 99.66 %  |  98.99% |
| mAP / 50 IoU  | 60.37  |  35.61 |
| FPS  | 26.16 (Collab GPU)  |  25.86 (Collab GPU) |

