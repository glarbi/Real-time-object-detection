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
| Dataset  | Pascal VOC  |  COCO |  ImageNet |
| ------------ | :------------: | :------------: | :------------: |
| Trained Model  | [Download](https://drive.google.com/file/d/1vYFwiRIN7qhQgPcV5A5fNcShExUS7oFz/view?usp=sharing)  |  Soon.. |  N/A |
| Demo Video / 0.87 Conf | [Watch](https://drive.google.com/file/d/1E6HDOAlPVNL1hCpzuNA1U-CkrlYQYc7B/view?usp=sharing)  |  Soon.. |  N/A |
| Class Accuracy  | 88.07 %  |  57.05 % |  N/A |
| Object Accuracy  |  67.51 % |  22.02 % |  N/A |
| No Object Accuracy  | 99.66 %  |  99.59 % |  N/A |
| mAP / 50 IoU  | 60.37  |  Soon.. |  N/A |
| FPS  | 26.16 (Collab GPU)  |  Soon.. |  N/A |

