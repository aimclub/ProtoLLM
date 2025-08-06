import base64
from io import BytesIO

from langchain_core.messages import HumanMessage
from PIL import Image


def convert_to_base64(file_path):
    """
    Convert PIL images to Base64 encoded strings

    :param file_path: path to image
    :return: Re-sized Base64 string
    """
    pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    text = data["text"]
    content_parts = []

    if "image" in data.keys():
        imgs = data["image"]
        for img in imgs:
            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{img}",
            }
            content_parts.append(image_part)

    text_part = {"type": "text", "text": text}
    content_parts.append(text_part)

    return HumanMessage(content=content_parts)
