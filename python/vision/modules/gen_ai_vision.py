import pathlib
from urllib.request import Request, urlopen
from dotenv import load_dotenv
import base64
from pathlib import Path
import os

load_dotenv()

key = os.getenv("key1")
endpoint = os.getenv("endpoint")
model_name = os.getenv("model_name")


from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.inference.models import (
	SystemMessage,
	UserMessage,
	TextContentItem,
	ImageContentItem,
	ImageUrl
	)

project_client = AIProjectClient.from_connection_string(
	conn_str=endpoint,
	credential=DefaultAzureCredential(),
	)

chat_client = project_client.inference.get_chat_completions_client(model=model_name)


image_url = "https://github.com/MicrosoftLearning/mslearn-ai-vision/raw/refs/heads/main/Labfiles/08-gen-ai-vision/orange.jpeg"
image_format = "jpeg"
request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
image_data = base64.b64encode(urlopen(request).read().decode("utf-8"))
data_url = f"data:image/{image_format};base64,{image_data}"


system_message = "You are an AI assistant in a grocery store that sells fruit."
prompt = input("Prompt: ")


response = chat_client.complete(
	messages=[
		SystemMessage(system_message),
		UserMessage(content=[
			TextContentItem(text=prompt),
			ImageContentItem(image_url=ImageUrl(url=data_url))
		]),
	]
)
print(response.choices[0].message.content)
