# cp-VTryon-plus-Flask-App

Flask App version of https://github.com/minar09/cp-vton-plus

**Master branch is for cuda based version,
cpu branch is for cpu based version.**

if you find any problem feel free to raise issue.

## what's new [Oct 2023]
- Python 3.11 support
- Supporting new version of torch and torchvision.
  - torch = 2.0.1
  - torchvision = 0.15
  - opencv = 4.8.1.78
- fix all the deprecated warning of torch and resolve all isuses regarding dependency.
- now you can upload .JPG, .png/.PNG and .jpeg/.JPEG files. (resolve the .jpg dependency).
- showing both images on result page. original and result.

### Installation

**For cuda version (GPU), install torch with cuda [Here.](https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP)**
**Remove torch and torchvision from requirements.txt and install it manually.**

`pip install -r requirements.txt`

`python application.py`


### Pretrained Models

Download the these models and paste in folder "tryon_utils/checkpoints/"

- Tryon [Models](https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP) 
  - paste in checkpoints/GMM and checkpoints/TOM
- Graphonomy [inference.pth](https://drive.google.com/uc?id=1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9)
  - paste in checkpoints/ 
- OpenPose [model](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel)
  - paste in checkpoints/