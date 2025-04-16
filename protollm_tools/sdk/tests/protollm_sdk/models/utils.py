import pytest
import uuid

from protollm_sdk.models.utils import validate_image_base64, generate_job_id


@pytest.mark.parametrize("input_str,expected", [
    # Correct Base64 image string
    (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgcBBH5C60cAAAAASUVORK5CYII=",
        ""
    ),
    # Content not starting with "data:image/"
    (
        "data:application/octet-stream;base64,SGVsbG8sIHdvcmxkIQ==",
        "The content should start with 'data:image/', but it doesn't"
    ),
    # Content without a comma
    (
        "data:image/png;base64iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgcBBH5C60cAAAAASUVORK5CYII=",
        "The content should contain a comma separating the header and the base64 encoded data, but it doesn't"
    ),
    # Header without "base64"
    (
        "data:image/png;utf8,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgcBBH5C60cAAAAASUVORK5CYII=",
        "The content should contain 'base64' in the header, but it doesn't"
    ),
    # Incorrect Base64 encoding
    (
        "data:image/png;base64,NotAValidBase64",
        "Invalid base64 encoding"
    ),
])
def test_validate_image_base64(input_str, expected):
    result = validate_image_base64(input_str)
    assert result == expected


def test_generate_job_id():
    # Generate 100 job IDs
    job_ids = [generate_job_id() for _ in range(100)]
    try:
        [uuid.UUID(job_id) for job_id in job_ids]
    except ValueError:
        pytest.fail("Function generate_job_id() did not return valid UUID strings.")
    assert len(set(job_ids)) == 100, "Generated job IDs are not unique."