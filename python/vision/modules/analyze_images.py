from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("key1")
endpoint = os.getenv("endpoint")


from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


# Create an instance of the ImageAnalysisClient
imageanalysisclient = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
    )

# Read the image file
img_path = 'vision\data\pics\orange.jpeg'
with open(img_path, "rb") as image:
        img = image.read()


# Call the analyze method to analyze the image and get the result
result = imageanalysisclient.analyze(
    image_data=img,
    visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.DENSE_CAPTIONS,
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.PEOPLE,
        ]
    )

#Need to structurize the output // maybe openai will help with that

