# cp-VTryon-plus-Flask-App

Flask App version of https://github.com/minar09/cp-vton-plus (**CPU** based) for **custom images**

New version of https://github.com/vinodbukya6/cp-VTryon-plus-Flask-App **@vinodbukya6**

it is CPU based. on intel 4 core 8 thread 10th gen i7 it take 30 sec to produce output.

if you find any problem feel free to raise issue.
# what's new 
- code is tested on new version of torch=1.13.1, torchvision=0.14.1 and opencv=4.5.5.64 using with python 3.9 pip installation.
- fix all the deprecated warning of torch and resolve all isuses regarding dependency.
- now you can upload .JPG, .png/.PNG and .jpeg/.JPEG files. (resolve the .jpg dependency).
- showing both images on result page. original and result.

**Installation**

run pip install -r requirements.txt

**Pretrained Models**

Download pretrained models and paste in folder "tryon_utils/checkpoints/"

Tryon(GMM & TOM)Models (checkpoints/GMM/) (checkpoints/TOM/) - https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP

Graphonomy(inference.pth) (checkpoints/) - https://drive.google.com/uc?id=1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9

OpenPose (checkpoints/) - http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel

**Testing**

Run "python app.py"
