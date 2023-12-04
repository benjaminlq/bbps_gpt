from utils import generate_img_url
from typing import Optional, Union, Tuple, Literal, List

EXAMPLE_PROMPT: str = "=====\nEXAMPLE:\n====="

QUERY_PROMPT: str = "Analyse this bowel image and return the BBPS score"

def get_gpt4v_messages(
    query_img_path: str,
    system_prompt: str,
    query_prompt: str = None,
    query_resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None,
    example_images: Optional[Union[str, List[str]]] = None,
    example_prompt: str = None,
    example_resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None
):
    query_prompt = query_prompt or QUERY_PROMPT
    example_prompt = example_prompt or EXAMPLE_PROMPT
        
    messages=[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
                ],
        }
    ]
    
    if example_images:
        if isinstance(example_images, str):
            example_images = [example_images]
            
        for example_image in example_images:
            sample_img_url = generate_img_url(example_image, resize=example_resize)
            example_content = {
                "role": "system",
                "content": [
                    {"type": "text", "text": example_prompt},
                    {"type": "image_url",
                     "image_url": {"url": sample_img_url, "detail": "high"}},
                    ]
                }
            messages.append(example_content)
    
    query_img_url = generate_img_url(query_img_path, resize=query_resize)
    query_content = {
        "role": "user",
        "content": [
            {"type": "text", "text": query_prompt},
            {"type": "image_url",
             "image_url": {"url": query_img_url, "detail": "high"}},
            ]
        }
    
    messages.append(query_content)
    
    return messages