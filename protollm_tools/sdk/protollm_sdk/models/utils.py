import base64
import binascii
import time

import uuid


def validate_image_base64(image_content: str) -> str:
    """
    Checks if the provided string is a valid base64 encoded image.

    Args:
        image_content (str): The string to check.

    Returns:
        str: The Error message if the string is not a valid base64 encoded image, otherwise returns an empty string.
    """
    if not image_content.startswith("data:image/"):
        return "The content should start with 'data:image/', but it doesn't"

    try:
        header, encoded = image_content.split(",", 1)
    except ValueError:
        return "The content should contain a comma separating the header and the base64 encoded data, but it doesn't"

    if "base64" not in header:
        return "The content should contain 'base64' in the header, but it doesn't"

    try:
        image_data = base64.b64decode(encoded, validate=True)
        return ""
    except (binascii.Error, ValueError):
        return "Invalid base64 encoding"


def generate_job_id() -> str:
    """
    Generate a unique job ID.

    Returns:
        str: A unique job ID.
    """
    return str(uuid.uuid4())

def current_time() -> str:
    """
    Get the current time in ISO 8601 format.

    Returns:
        str: The current time in ISO 8601 format.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

