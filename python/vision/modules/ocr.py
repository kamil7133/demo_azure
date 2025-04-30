from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("key1")
endpoint = os.getenv("endpoint")


from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import ReadResult, VisualFeatures
from azure.core.credentials import AzureKeyCredential

img_client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
    )

img_path = 'python/vision/data/pics/Note.jpg'
with open(img_path, "rb") as image:
     img = image.read()


result = img_client.analyze(
    image_data=img,
    visual_features=[VisualFeatures.READ]
    )


xd = result.get("readResult").get("blocks")

xd = xd[0].get("lines")


counter = 0
for i in xd:
    counter += 1
    print(counter, i.get("text"))
