import base64

SUPPORTED = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}


def load_image_as_base64(image_path: str) -> tuple:
    """Returns (base64_string, mime_type)"""
    ext = "." + image_path.split(".")[-1].lower()
    mime = SUPPORTED.get(ext, "image/png")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return b64, mime
