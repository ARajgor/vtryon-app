## import
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import shutil
import subprocess
import time
import cv2

## from custom py files
from tryon_utils.openpose_json import generate_pose_keypoints
from tryon_utils.cloth_mask import cloth_masking
from tryon_utils.image_mask import make_body_mask

app = Flask(__name__)
app.config['DATABASE'] = 'database/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

# check extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
    
@app.route('/')
def index():
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Uploaded Person and Cloth Images
    file_person = request.files['personImage']
    file_cloth = request.files['clothImage']
    print("file_person, file_cloth:", file_person, file_cloth)
    
    if file_person and allowed_file(file_person.filename) and file_cloth and allowed_file(file_cloth.filename):
        filename_person = secure_filename(file_person.filename)
        filename_cloth = secure_filename(file_cloth.filename)
        # save images
        file_person.save(os.path.join("data/test/image/", filename_person))
        file_cloth.save(os.path.join("data/test/cloth/", filename_cloth))
        print("filename1, filename2:", filename_person, filename_cloth)
        
        # ..... Resize/Crop Images 192 x 256 (width x height) ..... # 
        img_p = cv2.imread("data/test/image/"+filename_person)
        person_resize = cv2.resize(img_p, (192, 256))
        # save resized person image
        cv2.imwrite("data/test/image/"+filename_person, person_resize) 
        
        img_c = cv2.imread("data/test/cloth/"+filename_cloth)
        cloth_resize = cv2.resize(img_c, (192, 256)) 
        # save resized cloth image
        cv2.imwrite("data/test/cloth/"+ filename_cloth, cloth_resize)

        shutil.copyfile("data/test/image/"+filename_person, "static/data/test/image/"+filename_person)
        
        # ..... Cloth Masking ..... #
        image_path = "data/test/cloth/" + filename_cloth
        res_path = "data/test/cloth-mask/" + filename_cloth
        clothmask = cloth_masking(image_path, res_path)
        
        
        # ..... Image parser ..... # 
        cmd_parse = "python tryon_utils/inference.py --loadmodel tryon_utils/checkpoints/inference.pth --img_path " + filename_person + " --output_path img/ --output_name "+ filename_person
        subprocess.call(cmd_parse, shell=True)
        
        
        # ..... Person Image Masking ..... #
        #img_file = "000010_0.jpg", seg_file = "000010_0.png" 
        seg_file = filename_person.replace(".jpg", ".png")
        img_mask = make_body_mask(filename_person, seg_file)
        
        
        # ..... Generate Pose Keypoints .....# 
        pose_keypoints = generate_pose_keypoints(filename_person)
        
        
        # ..... Write test sample pair txt file ..... #
        with open("data/test_samples_pair.txt", "w") as text_file:
            text_file.write(str(filename_person) +" "+ str(filename_cloth))


        # ..... Run Geometric Matching Module(GMM) Model ..... #
        cmd_gmm = "python tryon_utils/test.py --name GMM --stage GMM --workers 4 --datamode test --data_list test_samples_pair.txt --checkpoint tryon_utils/checkpoints/GMM/gmm_final.pth"
        subprocess.call(cmd_gmm, shell=True)
        #time.sleep(10)
        # move generated files to data/test/
        # result/GMM/test/warp-cloth/004325_1.jpg
        warp_cloth = "static/result/GMM/test/warp-cloth/" + filename_person
        warp_mask = "static/result/GMM/test/warp-mask/" + filename_person
        shutil.copyfile(warp_cloth, "data/test/warp-cloth/"+ filename_person)
        shutil.copyfile(warp_mask, "data/test/warp-mask/"+ filename_person)


        # ..... Run Try-on Module(TOM) Model ..... #
        cmd_tom = "python tryon_utils/test.py --name TOM --stage TOM --workers 4 --datamode test --data_list test_samples_pair.txt --checkpoint tryon_utils/checkpoints/TOM/tom_final.pth"
        subprocess.call(cmd_tom, shell=True)
        return render_template('result.html', user_image=filename_person)


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5001)
    
