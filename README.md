# cp-VTryon-plus-Flask-App

Falsk App version of https://github.com/minar09/cp-vton-plus (**CPU** based) for **custom images**. Thank you so much **@minar09** for your Great Work!!!

**Installation**

conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch. For all packages, run pip install -r requirements.txt

**Pretrained Models**

Download pretrained models and paste in folder "checkpoints/"

Tryon(GMM & TOM)Models - https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP

Graphonomy - https://drive.google.com/uc?id=1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9

OpenPose - !wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel and openpose_pose_coco.prototxt

**Testing with custom images**

to run the model with custom internet images, make sure you have the following:

Create "data/test/cloth, cloth-mask, image, image-mask, image-parse, image-parse-new, pose, warp-cloth, warp-mask" **(total 9 folders)**

1. image (image of a person, crop/resize to 192 x 256 (width x height) pixels) - https://github.com/vinodbukya6/cp-VTryon-plus-Flask-App/blob/36860ed0ebaaabf4bea2c02c49d2f0dff346b9b9/app.py#L44
        
2. image-parse (you can generate with CIHP_PGN or Graphonomy pretrained networks from the person image. See this comment) - https://github.com/vinodbukya6/cp-VTryon-plus-Flask-App/blob/36860ed0ebaaabf4bea2c02c49d2f0dff346b9b9/app.py#L61

3. cloth (in-shop cloth image, crop/resize to 192 x 256 (width x height) pixels)
4. cloth-mask (binary mask of cloth image, you can generate it with simple pillow/opencv function) - https://github.com/vinodbukya6/cp-VTryon-plus-Flask-App/blob/36860ed0ebaaabf4bea2c02c49d2f0dff346b9b9/app.py#L57

5. pose (pose keypoints of the person, generate with openpose COCO-18 model (OpenPose from the official repository is preferred)) - https://github.com/vinodbukya6/cp-VTryon-plus-Flask-App/blob/36860ed0ebaaabf4bea2c02c49d2f0dff346b9b9/app.py#L72

6. Also, make a test_pairs.txt file for your custom images. Follow the VITON dataset format to keep same arrangements, otherwise you can modify the code.- 
https://github.com/vinodbukya6/cp-VTryon-plus-Flask-App/blob/36860ed0ebaaabf4bea2c02c49d2f0dff346b9b9/app.py#L76

7. Run Geometric Matching Module(GMM) Model - https://github.com/vinodbukya6/cp-VTryon-plus-Flask-App/blob/36860ed0ebaaabf4bea2c02c49d2f0dff346b9b9/app.py#L81
8. Copy "result/GMM/test/warp-cloth & warp-mask folders to "data/test/" 

10. Run Try-on Module(TOM) Model - https://github.com/vinodbukya6/cp-VTryon-plus-Flask-App/blob/36860ed0ebaaabf4bea2c02c49d2f0dff346b9b9/app.py#L93
Results in "result/TOM/test/try-on/"

**Testing**

Run "python app.py"

Upload target person image and cloth image. Get results!!!
