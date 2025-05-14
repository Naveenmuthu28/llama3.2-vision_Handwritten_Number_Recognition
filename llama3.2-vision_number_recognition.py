
import ollama
import re
import cv2
import numpy as np
import os
import imutils
from imutils.contours import sort_contours
import matplotlib.pyplot as plt



# Function to get absolute paths for standardized folders
def get_standard_paths():
    # Get the current script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths relative to the script directory
    ref_image_path = os.path.join(base_dir, "reference_image.png")
    input_images_folder = os.path.join(base_dir, "input_images")

    # Check if paths exist
    if not os.path.exists(ref_image_path):
        raise FileNotFoundError(f"Reference image not found at {ref_image_path}")
    if not os.path.exists(input_images_folder):
        raise FileNotFoundError(f"Input images folder not found at {input_images_folder}")

    return ref_image_path, input_images_folder

# Automatically get the paths
try:
    refimg, inpimg = get_standard_paths()
    print(f"Reference Image Path: {refimg}")
    print(f"Input Images Folder Path: {inpimg}")
except FileNotFoundError as e:
    print(e)
    exit()



roi_values = [
    [(466, 1220), (764, 1302), 'text', 'a1'],
    [(776, 1226), (1088, 1306), 'text', 'a2'],
    [(1100, 1230), (1424, 1310), 'text', 'a3'],
    [(470, 1310), (764, 1392), 'text', 'a4'],
    [(778, 1318), (1084, 1396), 'text', 'a5'],
    [(1100, 1322), (1422, 1400), 'text', 'a6'],
    [(472, 1396), (766, 1476), 'text', 'b6a1'],
    [(776, 1404), (1086, 1482), 'text', 'b8a2'],
    [(1100, 1408), (1424, 1488), 'text', 'b6a3'],
    [(472, 1486), (764, 1564), 'text', 'b6b1'],
    [(776, 1494), (1084, 1570), 'text', 'b6b2'],
    [(1098, 1498), (1424, 1576), 'text', 'b6b3'],
    [(472, 1570), (766, 1650), 'text', 'b7a1'],
    [(776, 1576), (1088, 1656), 'text', 'b7a2'],
    [(1098, 1582), (1424, 1668), 'text', 'b7a3'],
    [(474, 1660), (766, 1738), 'text', 'b7b1'],
    [(776, 1666), (1088, 1746), 'text', 'b7b2'],
    [(1100, 1674), (1426, 1756), 'text', 'b7b3'],
    [(474, 1746), (766, 1828), 'text', 'c8a1'],
    [(778, 1752), (1086, 1836), 'text', 'c8a2'],
    [(1100, 1758), (1426, 1846), 'text', 'c8a3'],
    [(474, 1836), (764, 1918), 'text', 'c8b1'],
    [(778, 1842), (1088, 1928), 'text', 'c8b2'],
    [(1100, 1850), (1428, 1944), 'text', 'c8b3']
]

def llama_text_recognition(imgcrop):
    # Convert the image (NumPy array) to a byte string
    _, img_bytes = cv2.imencode('.jpg', imgcrop)  # Encoding image as a JPEG byte string
    img_bytes = img_bytes.tobytes()  # Convert the encoded image to bytes

    # Send the image to the LLaMA model
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': 'I want you to recognize this number and print only the number',
            'images': [img_bytes]  # Send the byte array
        }]
    )
    # Extracting the number from the response
    match = re.search(r'\d+', response['message']['content'])

    # Check if a match was found
    if match:
        return match.group()
    else:
        print("No number found in the image.")
        return "No number found"  # Return a default value if no number is found


def start_program():
    imgq = cv2.imread(refimg)
    path = inpimg
    roi = roi_values

    per = 50
    h, w, c = imgq.shape
    orb = cv2.ORB_create(10000)  # Feature detection using ORB
    kp1, des1 = orb.detectAndCompute(imgq, None)
    impkp1 = cv2.drawKeypoints(imgq, kp1, None)
    
    mypiclist = os.listdir(path)
    for j, y in enumerate(mypiclist):
        img = cv2.imread(path + "/" + y)

        # Finding descriptors and matches
        kp2, des2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2, des1)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:int(len(matches) * (per / 100))]
        imgmatch = cv2.drawMatches(img, kp2, imgq, kp1, good[:100], None, flags=2)

        srcpoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstpoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(srcpoints, dstpoints, cv2.RANSAC, 5.0)
        imgscan = cv2.warpPerspective(img, M, (w, h))

        imgshow = imgscan.copy()
        imgmask = np.zeros_like(imgshow)

        mydata = []
        for x, r in enumerate(roi):
            cv2.rectangle(imgmask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
            imgcrop = imgscan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

            # Using LLaMA for text recognition on each cropped image
            predicted_text = llama_text_recognition(imgcrop)
            print(predicted_text, end=' ', flush=True)  # print each number inline
            mydata.append(predicted_text)

        print()  # newline after all numbers for one image


            #Display the cropped image
            #cv2.imshow(f"Cropped Image - {r[3]}", imgcrop)
            #cv2.waitKey(0)  # Wait for a key press to continue to the next image
            #cv2.destroyAllWindows()  # Close the current image window

        with open('output.csv', 'a+') as f:
            for data in mydata:
                f.write(str(data) + ',')
            f.write('\n')
            print("output is saved in .csv file")

start_program()
