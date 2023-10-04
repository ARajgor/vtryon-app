from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import shutil
import subprocess
import time
import cv2

# from custom py files
from tryon_utils.openpose_json import generate_pose_keypoints
from tryon_utils.cloth_mask import cloth_masking
from tryon_utils.image_mask import make_body_mask
from tryon_utils.inference import inference

application = Flask(__name__)
application.config['DATABASE'] = 'database/'
application.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}


# check extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in application.config['ALLOWED_EXTENSIONS']


@application.route('/')
def index():
    return render_template('home.html')


@application.route('/upload', methods=['POST'])
def upload():
    t = time.time()
    file_person = request.files['personImage']
    file_cloth = request.files['clothImage']

    main_dir = "static/data/input/"

    image_dir = main_dir + "image/"

    cloth_dir = main_dir + "cloth/"
    cloth_mask_dir = main_dir + "cloth-mask/"

    warp_cloth_dir = main_dir + "warp-cloth/"
    warp_mask_dir = main_dir + "warp-mask/"

    if file_person and allowed_file(file_person.filename) and file_cloth and allowed_file(file_cloth.filename):
        filename_person = secure_filename(file_person.filename).lower()
        filename_cloth = secure_filename(file_cloth.filename).lower()

        # save images
        file_person.save(os.path.join(image_dir, filename_person))
        file_cloth.save(os.path.join(cloth_dir, filename_cloth))

        print("Images saved", "person: ", filename_person, "cloth: ", filename_cloth)

        # ..... Resize/Crop Images 192 x 256 (width x height) ..... #
        img_p = cv2.imread(image_dir + filename_person)
        person_resize = cv2.resize(img_p, (192, 256))

        # save resized person image
        cv2.imwrite(image_dir + filename_person, person_resize)

        img_c = cv2.imread(cloth_dir + filename_cloth)
        cloth_resize = cv2.resize(img_c, (192, 256))
        # save resized cloth image
        cv2.imwrite(cloth_dir + filename_cloth, cloth_resize)

        # ..... Cloth Masking ..... #
        cloth_masking(cloth_dir + filename_cloth, cloth_mask_dir + filename_cloth)

        # ..... Image parser ..... #
        inference('tryon_utils/checkpoints/inference.pth', main_dir, filename_person)

        # ..... Person Image Masking ..... #
        make_body_mask(main_dir, filename_person)

        # ..... Generate Pose Keypoint's .....#
        generate_pose_keypoints(main_dir, filename_person)

        # ..... Write input sample pair txt file ..... #
        with open("static/data/test_samples_pair.txt", "w") as text_file:
            text_file.write(str(filename_person) + " " + str(filename_cloth))

        # ..... Run Geometric Matching Module(GMM) Model ..... #
        cmd_gmm = "python tryon_utils/test.py --name GMM --stage GMM --workers 1 --datamode input --data_list " \
                  "test_samples_pair.txt --checkpoint tryon_utils/checkpoints/GMM/gmm_final.pth"
        subprocess.call(cmd_gmm, shell=True)

        # move generated files to data/input/
        warp_cloth = "static/result/GMM/input/warp-cloth/" + filename_person
        warp_mask = "static/result/GMM/input/warp-mask/" + filename_person

        shutil.copyfile(warp_cloth, warp_cloth_dir + filename_person)
        shutil.copyfile(warp_mask, warp_mask_dir + filename_person)

        # ..... Run Try-on Module(TOM) Model ..... #
        cmd_tom = "python tryon_utils/test.py --name TOM --stage TOM --workers 1 --datamode input --data_list " \
                  "test_samples_pair.txt --checkpoint tryon_utils/checkpoints/TOM/tom_final.pth"
        subprocess.call(cmd_tom, shell=True)

        print("Total time: ", time.time() - t)

        return render_template('result.html', user_image=filename_person)


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000)
