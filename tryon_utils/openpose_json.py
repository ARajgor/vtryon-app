import cv2
import time
import numpy as np
import json

protoFile = "tryon_utils/checkpoints/openpose_pose_coco.prototxt"
weightsFile = "tryon_utils/checkpoints/pose_iter_440000.caffemodel"
nPoints = 18

# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']


def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    # find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# generate target image pose key points    
def generate_pose_keypoints(base_dir, img_file):
    '''
    Generates pose keypoints
    Input: Person Image
    Output: Writes json file with keypoints(shape: 18*3 = 54)
    '''

    image1 = cv2.imread(base_dir + '/image/' + img_file)
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]

    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # print("Using CPU device")
    # elif args.device == "gpu":
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    #     print("Using GPU device")

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight / frameHeight) * frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    frameClone = image1.copy()
    pose_keypoints = []
    for i in range(nPoints):
        if detected_keypoints[i] == []:
            pose_keypoints.append(0)
            pose_keypoints.append(0)
            pose_keypoints.append(0)

        for j in range(len(detected_keypoints[i])):
            pose_keypoints.append(detected_keypoints[i][j][0])
            pose_keypoints.append(detected_keypoints[i][j][1])
            pose_keypoints.append(detected_keypoints[i][j][2].astype(float))

    json_data = {"version": 1.0, "people": [{
        "face_keypoints": [],
        "pose_keypoints": pose_keypoints,
        "hand_right_keypoints": [],
        "hand_left_keypoints": []
    }]}

    # write json file
    json_file = img_file.split('.')[0] + "_keypoints.json"
    with open(base_dir + "/pose/" + json_file, 'w') as outfile:
        json.dump(json_data, outfile)
