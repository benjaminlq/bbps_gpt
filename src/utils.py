import base64
import io
import json
import math
import os

from callbacks import TokenCounter
from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat
from PIL import Image
from typing import Optional, Literal, List, Tuple, Union, Literal

def encode_image(image_path: str, resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None):
    with open(image_path, "rb") as image_file:
        img_64_str = base64.b64encode(image_file.read()).decode('utf-8')
    
    if resize:
        img_data = base64.b64decode(img_64_str)
        img = Image.open(io.BytesIO(img_data))
        h, w = img.size
        if isinstance(resize, int):
            resize = (resize, resize)
        elif resize == "auto":
            if h < 512 and w < 512:
                resize = (h, w)
            elif h > w:
                resize = (512, int(w / (h/512)))
            else:
                resize = (int(h / (w/512)), 512)
                
        resized_img = img.resize(resize)
        
        # Save the resized image to a buffer
        buffer = io.BytesIO()
        resized_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode the resized image to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    else:
        return img_64_str
    
def generate_img_url(image_path: str, resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None):
    encoded_image = encode_image(image_path, resize=resize)
    image_url = f"data:image/jpeg;base64,{encoded_image}"
    return image_url

def extract_score(response_str: str, client: OpenAI, token_counter: Optional[TokenCounter] = None):
    system_prompt = """You are given a response containing the BBPS grading.
    Extract the BBPS score given in the response. If the score is not available, return empty string.
    Your output should be a JSON dictionary with key "Score" and value containing integer score. 
    Example Output: {"Score": 0}, {"Score": ""}
    """
    user_prompt = f"Response: {response_str}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ],
        max_tokens=128, temperature=0,
        response_format=ResponseFormat(type="json_object")
    )
    if token_counter:
        token_counter.update(response)
    
    return json.loads(response.choices[0].message.content, strict=False)

def calculate_token_img(
    w: int, h: int, quality: Literal["low", "high"] = "high"
    ) -> int:
    if quality == "low":
        return 85
    
    if w > 2048 and h > 2048:
        if w <= h:
            w = int(w / (h / 2048))
            h = 2048
        else:
            h = int(h / (w/2048))
            w = 2048
    elif w > 2048:
        h = int(h / (w/2048))
        w = 2048
    elif h > 2048:
        w = int(w / (h / 2048))
        h = 2048    
    if w > 512 and h > 512:
        if w >= h:
            w = int(w / (h/768))
            h = 768
        else:
            h = int(h / (w/768))
            w = 768
        
    no_of_tiles = math.ceil(w/512) * math.ceil(h/512)
    return 170 * no_of_tiles + 85

def save_checkpoint(
    ckpt_folder: str,
    img_paths: List[str],
    gpt_raw_answers: List[str],
    gpt_scores: List[str]
):
    content = dict(
        img_paths=img_paths,
        gpt_raw_answers=gpt_raw_answers,
        gpt_scores=gpt_scores
    )
    
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
    with open(os.path.join(ckpt_folder, "ckpt.json"), "w") as f:
        json.dump(content, f)