import numpy as np
import torch
import torchvision.transforms as T
#from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import sys

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    print(f"loading image {image_file}")
    if(os.path.exists(image_file)):
        image = Image.open(image_file).convert('RGB')
    else:
        print("no image file found")
        sys.exit()
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OpenGVLab/InternVL2-2B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    #load_in_8bit=True,
    #load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval().cuda()
#config = AutoConfig.from_pretrained(path)
#model = AutoModel.from_pretrained(config)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
#pixel_values = load_image('/home/schoch/internvl/examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = load_image('/mnt/c/Files/screenshots/t3.png', max_num=12).to(torch.bfloat16).cuda()
# gen_config=GenerationConfig(temperature=0.3))
generation_config = dict(max_new_tokens=1024, do_sample=True,temperature=0.1)

def rewrite_file_path(path):
    """
    Rewrite a Windows file path to a Unix/Linux format based on the provided mapping.
    
    :param windows_path: The Windows file path (e.g., "z:/Files/screenshots")
    :param mapping: A dictionary where keys are drive letters and values are mount points (e.g., {"z": "/mnt/c/Files"})
    :return: The rewritten Unix/Linux file path
    """
    # Split the path into drive letter and remaining part
    path = path.replace("\\", "/")

    if(path[0] == 'C' or path[0] == 'c'):
        mapping = {"c": "/mnt/c/"}
    if(path[0] == 'Z' or path[0] == 'z'):
        mapping = {"z": "/mnt/c/Files"}
    parts = path.split(':')
    
    if len(parts) == 2:
        drive, path = parts[0], parts[1]
        
        if drive.lower() in mapping:
            return f"{mapping[drive.lower()]}{path}"
    
    # If no match is found, return the original path
    return path



def run_inference(path):
    global model 
    # test hardcode path
    #path = '/mnt/c/Files/screenshots/t3.png'
    print(f"got path: {path}")
    
    rewritten_path = rewrite_file_path(path )
    pixel_values = load_image(rewritten_path, max_num=12).to(torch.bfloat16).cuda()
    question = f"""<image>
    The data to focus on can be found under the text "Ground Firmness:", the data 
    is in square boxes that are red (2x3 grid), green(2x3 grid) and blue (2,3 grid).  describe the data in the image. 
    some values may be prefixed with "L" or "R" 
    the data should be easily parsed into this peewee orm class 
    'class LMData(BaseModel):
    carry = FloatField(null=True)
    total = FloatField(null=True)
    roll = FloatField(null=True)
    v_launch = FloatField(null=True)
    height = FloatField(null=True)
    descent = FloatField(null=True)
    h_launch = FloatField(null=True)
    lateral = FloatField(null=True)
    spin = FloatField(null=True)
    spin_axis = FloatField(null=True)
    club_path = FloatField(null=True)'
    to denote the direction. null values are denoted with "-----".  output the 
    key value pairs as json.   
    just return the raw json that can be parsed correctly.
    VERY IMPORTANT:  the json keys should match the peewee field names and the json values should be string type.
    for example: 
    
    "'total':'102.2',... "
    
    ensure you wrap all number values as strings in quotes.  
    ensure the json is valid and can be parsed correctly.
    do not elaborate."""
    print(f"run inf: question: {question}")

    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')
    return response