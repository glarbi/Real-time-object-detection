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
- SciPy

## Results
| Dataset  | Pascal VOC  |  COCO |  ImageNet |
| ------------ | :------------: | :------------: | :------------: |
| Trained Model  | [Download](https://drive.google.com/file/d/1-15dU1YoPJlet5d9TRi88bhfgj_iCZQN/view?usp=sharing "Pascal")  | N/A  |  N/A |
| Class Accuracy  | 89.45 %  | N/A  |  N/A |
| Object Accuracy  |  76.12 % | N/A  | N/A  |
| No Object Accuracy  | 99.02 %  | N/A  | N/A  |
| mAP / 50 IoU  | 60.66  | N/A  | N/A  |

