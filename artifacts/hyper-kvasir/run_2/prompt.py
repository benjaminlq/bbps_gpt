from utils import generate_img_url
from typing import Union, Optional, Tuple, Literal

task_system_prompt = """You are an expert endoscopist in charge of bowel preparation for colonoscopy.
If you don't know the answer, say 'I don't know' do not try to make up an answer.
=====
TASK:
You are given an image of a bowel after cleansing, your task is to assess the quality of the bowel preparation.
The grading should be performed using the standardized Boston-Bowel-Preparation-Scale (BBPS). Perform the following step:
1. Analyse the given image and identify the degree of stool and residual staining and whether mucosa of colon can be seen well.
2. Based on the EXAMPLES given, return the BBPS grade for the given image. Can be one of [0, 1, 2, 3]
=====
"""

query_prompt="Analyse this bowel image and return the BBPS score"

def get_gpt4v_messages(
    query_img_path: str,
    resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None,
):
    query_img_url = generate_img_url(query_img_path, resize=resize)
    messages=[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": task_system_prompt}
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query_prompt},
                {"type": "image_url",
                    "image_url": {"url": query_img_url, "detail": "high"}},
                ],
        }
    ]
    
    return messages