from tkinter import Image
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("key1")
endpoint = os.getenv("endpoint")


from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection03
from azure.core.credentials import AzureKeyCredential

# Create an instance of the ImageAnalysisClient
face_client = FaceClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
    )

# Read the image file
img_path = 'python/vision/data/pics/person.jpg'
with open(img_path, "rb") as image:
        img = image.read()


features = [
    FaceAttributeTypeDetection03.HEAD_POSE,
    FaceAttributeTypeDetection03.BLUR,
    FaceAttributeTypeDetection03.MASK
    ]

df = face_client.detect(
    image_content=img,
    detection_model=FaceDetectionModel.DETECTION03,
    recognition_model=FaceRecognitionModel.RECOGNITION04,
    return_face_id=False,
    return_face_attributes=features,
    )


import matplotlib as plt


if len(df) > 0:
    print(f"Detected {len(df)} faces.")

    fig = plt.figure(figsize=(8, 6))
    plt.axis("off")
    image = Image.open(img)
    draw = Image.Draw(image)
    color = 'lightgreen'
    face_count = 0


for face in df:
    face_count += 1
    print('\nFace number {}'.format(face_count))

    print(' - Head Pose (Yaw): {}'.format(face.face_attributes.head_pose.yaw))
    print(' - Head Pose (Pitch): {}'.format(face.face_attributes.head_pose.pitch))
    print(' - Head Pose (Roll): {}'.format(face.face_attributes.head_pose.roll))
    print(' - Blur: {}'.format(face.face_attributes.blur.blur_level))
    print(' - Mask: {}'.format(face.face_attributes.mask.type))

    r = face.face_rectangle
    bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
    draw = Image.Draw(image)
    draw.rectangle(bounding_box, outline=color, width=5)
    annotation = 'Face number {}'.format(face_count)
    plt.annotate(annotation,(r.left, r.top), backgroundcolor=color)

    plt.imshow(image)
    outputfile = 'detected_faces.jpg'
    fig.savefig(outputfile)

    print('\nResults saved in', outputfile)