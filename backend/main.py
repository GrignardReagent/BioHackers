from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles

from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import os
import cv2
from scipy.ndimage import label, binary_dilation, binary_erosion

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoTokenizer
import time

app = FastAPI()

app.mount("/data", StaticFiles(directory="data"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

def load_color_image(image_path):
    """加载彩色图像"""
    if not os.path.exists(image_path):
        print(f"错误: 文件 {image_path} 不存在")
        return None, None
    
    # 使用OpenCV读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None, None
    
    # 转换到RGB色彩空间用于显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换到HSV色彩空间以便更容易识别红色
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    return img_rgb, img_hsv

def process_lake_areas(img_hsv, black_threshold=30):
    """处理图像中的湖泊区域（原接近黑色区域）"""
    if img_hsv is None:
        return None
    
    # 提取V通道（亮度）
    v_channel = img_hsv[:,:,2]
    
    # 设置黑色阈值 - V值小于阈值被认为是接近黑色的
    lake_area_mask = v_channel > black_threshold
    
    # 创建二值化图像: 黑色区域为黑色(0)，其他区域为白色(1)
    result_lake = np.where(lake_area_mask, 0, 1)
    return result_lake

def trim_white_borders(img_array, threshold=0.95):
    """去除图像周围的白边"""
    # 对于灰度图像，白色像素接近1.0
    if len(img_array.shape) == 2:
        # 找出非白色区域的边界
        mask = img_array < threshold
        # 如果图像全白，返回原图
        if not np.any(mask):
            return img_array, False
        
        # 找出非白色区域的边界
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 裁剪图像
        trimmed = img_array[rmin:rmax+1, cmin:cmax+1]
        return trimmed, True
    else:
        # 如果不是灰度图像，直接返回
        return img_array, False
    
def make_square(img_array, pad_value=1.0):
    """将图像调整为正方形，通过添加白色填充"""
    height, width = img_array.shape[:2]
    
    # 检查图像是否已经是正方形
    if height == width:
        return img_array, False
    
    # 计算需要添加的填充大小
    size = max(height, width)
    
    if len(img_array.shape) == 2:
        # 灰度图像
        square_img = np.ones((size, size), dtype=img_array.dtype) * pad_value
        # 计算居中位置
        h_offset = (size - height) // 2
        w_offset = (size - width) // 2
        # 将原始图像复制到正方形的中心
        square_img[h_offset:h_offset+height, w_offset:w_offset+width] = img_array
    else:
        # 彩色图像
        channels = img_array.shape[2]
        square_img = np.ones((size, size, channels), dtype=img_array.dtype) * pad_value
        # 计算居中位置
        h_offset = (size - height) // 2
        w_offset = (size - width) // 2
        # 将原始图像复制到正方形的中心
        square_img[h_offset:h_offset+height, w_offset:w_offset+width] = img_array
    
    return square_img, True

def load_building_image(building_path):
    """加载并预处理建筑物图像（原LiDAR图像）"""
    if not os.path.exists(building_path):
        print(f"错误: 文件 {building_path} 不存在")
        return None
    
    # 打开图像
    img = Image.open(building_path)
    
    # 转换为灰度图像（如果不是灰度图）
    if img.mode != 'L':
        img = img.convert('L')
    
    # 转换为NumPy数组进行处理
    building_img_array = np.array(img).astype(float) / 255.0
    
    # 去除白边
    trimmed_building, was_trimmed = trim_white_borders(building_img_array)
    if was_trimmed:
        print("建筑物图像白边已去除")
        building_img_array = trimmed_building
        
        # 检查是否为正方形
        height, width = building_img_array.shape
        if height != width:
            print(f"建筑物图像不是正方形: {width}x{height}")
            square_building, was_squared = make_square(building_img_array)
            if was_squared:
                print(f"建筑物图像已调整为正方形: {square_building.shape[0]}x{square_building.shape[1]}")
                building_img_array = square_building
    
    return building_img_array

def process_green_areas(img_hsv):
    """处理图像中的绿色区域（原红色区域）"""
    if img_hsv is None:
        return None
    
    # 定义红色的HSV范围
    lower_red1 = np.array([0, 70, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩码
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    green_area_mask = mask1 + mask2

    # 创建二值化图像: 红色区域为黑色(0)，其他区域为白色(1)
    result_green = np.where(green_area_mask > 0, 0, 1)
    result_green = np.logical_not(result_green)
    return result_green

def remove_noise(mask, size_threshold):
    labeled_array, num_features = label(mask)

    # Create an output mask to keep components larger than the threshold
    output_mask = np.zeros_like(mask)

    # Iterate over labeled components
    for i in range(1, num_features + 1):
        # Count the number of pixels in the current component
        component_size = np.sum(labeled_array == i)
    
        # If the component size is greater than or equal to the threshold, keep it
        if component_size >= size_threshold:
            output_mask[labeled_array == i] = 1
    return output_mask

@app.post("/upload")
async def upload(
    image: UploadFile,
    redImage: UploadFile,
    lidarImage: UploadFile,
):
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/image.jpg', 'wb') as f:
        f.write(image.file.read())
    with open('data/redImage.jpg', 'wb') as f:
        f.write(redImage.file.read())
    with open('data/lidarImage.png', 'wb') as f:
        f.write(lidarImage.file.read())

    ori_img = Image.open('data/image.jpg')

    ori_img_size = ori_img.size

    img_rgb, img_hsv = load_color_image('data/redImage.jpg')
    lake_mask = process_lake_areas(img_hsv, 50)

    lidar_img_array = load_building_image('data/lidarImage.png')

    lidar_water_threshold = 0.5
    more_water = lidar_img_array < lidar_water_threshold
    
    # 创建二值化图像: 黑色区域为黑色(0)，其他区域为白色(1)
    more_water = np.where(more_water, 0, 1)

    resized_more_water = cv2.resize(more_water.astype(np.uint8), ori_img_size, 
                                   interpolation=cv2.INTER_NEAREST)

    merged_water = np.logical_or(lake_mask, resized_more_water)

    lake_mask_no_noise = remove_noise(merged_water, 138)

    lake_mask_dilated = binary_dilation(lake_mask_no_noise, iterations=1, structure=np.ones((2,2)))

    lake_overlay_blue = np.zeros((ori_img_size[1], ori_img_size[0], 4), dtype=np.uint8)
    lake_overlay_blue[:, :, 2] = lake_mask_dilated * 255
    lake_overlay_blue[:, :, 3] = lake_mask_dilated * 100
    plt.imsave('data/lake_overlay.png', lake_overlay_blue)

    green_mask = process_green_areas(img_hsv)
    green_mask = remove_noise(green_mask, 10)

    green_overlay = np.zeros((ori_img_size[1], ori_img_size[0], 4), dtype=np.uint8)
    green_overlay[:, :, 1] = green_mask * 255
    green_overlay[:, :, 3] = green_mask * 100
    plt.imsave('data/green_overlay.png', green_overlay)

    lidar_building_threshold = 0.21
    lidar_not_building_threshold = 0.6
    lidar_img_array_ = np.copy(lidar_img_array)
    lidar_img_array_[lidar_img_array_ > lidar_not_building_threshold] = 0
    building_mask = lidar_img_array_ > lidar_building_threshold

    lake_mask_dilated_extra = binary_dilation(lake_mask_dilated, iterations=6, structure=np.ones((3,3)))
    resized_lake_mask = cv2.resize(lake_mask_dilated_extra.astype(np.uint8), building_mask.shape, 
                                   interpolation=cv2.INTER_NEAREST)
    building_mask_no_lake = np.logical_and(building_mask, np.logical_not(resized_lake_mask))

    building_mask = binary_dilation(building_mask_no_lake, iterations=20, structure=np.ones((3,3)))
    building_mask = binary_erosion(building_mask, iterations=1, structure=np.ones((3,3)))

    resize_building_mask = cv2.resize(building_mask.astype(np.uint8), ori_img.size, 
                                   interpolation=cv2.INTER_NEAREST)
    resize_building_mask = binary_dilation(resize_building_mask, iterations=1, structure=np.ones((2,2)))
    building_mask_no_green = np.logical_and(resize_building_mask, np.logical_not(green_mask))
    building_mask_no_green = building_mask_no_green.astype(bool)

    building_overlay = np.zeros((ori_img_size[1], ori_img_size[0], 4), dtype=np.uint8)
    building_overlay[:, :, 0] = building_mask_no_green * 255
    building_overlay[:, :, 3] = building_mask_no_green * 100
    plt.imsave('data/building_overlay.png', building_overlay)

    return {
        'message': 'success',
    }

@app.get("/results")
async def results():
    if not os.path.exists('data/lake_overlay.png'):
        return {
            'message': 'no results',
        }
    if not os.path.exists('data/green_overlay.png'):
        return {
            'message': 'no results',
        }
    if not os.path.exists('data/building_overlay.png'):
        return {
            'message': 'no results',
        }
    return {
        'lake_overlay': '//localhost:8000/data/lake_overlay.png',
        'green_overlay': '//localhost:8000/data/green_overlay.png',
        'building_overlay': '//localhost:8000/data/building_overlay.png',
        'image': '//localhost:8000/data/image.jpg',
    }

# -------------------------------
# 2. Define GRAFT Model
# -------------------------------
class GRAFT(nn.Module):
    def __init__(self, CLIP_version="openai/clip-vit-base-patch16", temp=False, bias_projector=True):
        super().__init__()
        self.satellite_image_backbone = CLIPVisionModelWithProjection.from_pretrained(CLIP_version)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.satellite_image_backbone.config.hidden_size, eps=self.satellite_image_backbone.config.layer_norm_eps),
            nn.Linear(self.satellite_image_backbone.config.hidden_size, self.satellite_image_backbone.config.projection_dim, bias=bias_projector),
        )
        self.temp = temp
        if temp:
            self.register_buffer("logit_scale", torch.ones([]) * (1 / 0.07))

    def forward_features(self, image_tensor):
        embed = self.satellite_image_backbone(image_tensor).image_embeds
        return F.normalize(embed)

# -------------------------------
# 3. Load Model & Configure GPU
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize GRAFT model
model = GRAFT(temp=True, bias_projector=False).to(device)

# Load CLIP text model & tokenizer
textmodel = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").eval().to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    
# -------------------------------
# 4. Classification Function (Directly Resize to 224x224)
# -------------------------------
def zero_shot_classification(image, idx):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Generate and save expanded image
    expanded_image = image.resize((224, 224), Image.BICUBIC)
    
    with torch.no_grad():
        tr_image = transform(expanded_image).unsqueeze(0).to(device)
        image_feature = model.forward_features(tr_image)

    classes = ["residential", "industrial", "waterway", "green belt", "farm"]
    with torch.no_grad():
        textsenc = tokenizer(classes, padding=True, return_tensors="pt").to(device)
        class_embeddings = F.normalize(textmodel(**textsenc).text_embeds, dim=-1)

    classlogits = image_feature.cpu().numpy() @ class_embeddings.cpu().numpy().T
    class_scores = {cls: float(score) for cls, score in zip(classes, classlogits[0])}
    best_class = max(class_scores, key=class_scores.get)
    return best_class, class_scores, expanded_image

# -------------------------------
# 5. Directly Split into 10x10 Grid and Classify
# -------------------------------
def grid_classification(image, grid_size=10):
    width, height = image.size
    # Calculate patch width and height based on the grid size and image dimensions
    patch_w = width // grid_size  # width of each patch
    patch_h = height // grid_size  # height of each patch
    
    # If the image is not perfectly divisible by grid_size, adjust the last patches
    results = []
    idx = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            left, upper = i * patch_w, j * patch_h
            # Handle the last column and row patches to cover the entire image
            right = min(left + patch_w, width)
            lower = min(upper + patch_h, height)
            
            # Crop the image into sub-images (patches)
            sub_image = image.crop((left, upper, right, lower))
            best_class, class_scores, expanded_image = zero_shot_classification(sub_image, idx)
            results.append((sub_image, best_class, class_scores, expanded_image))
            idx += 1
    
    return results

# -------------------------------
# 7. Save Classification Results and Track Class Frequencies
# -------------------------------
results_json = {}
class_count = {cls: 0 for cls in ["residential", "industrial", "waterway", "green belt", "farm"]}

def save_final_results(results):
    global class_count
    for idx, (patch, best_class, class_scores, expanded_image) in enumerate(results):
        # Only save the scores images (with _scores.png suffix)
        # output_scores_img_path = os.path.join(output_dir, f"image_{idx + 1}_scores.png")
        
        # Update class count using the dictionary
        class_count[best_class] += 1
        
        # Plot expanded image and classification scores side by side
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display the expanded image on the left
        # axes[0].imshow(expanded_image)
        # axes[0].axis("off")
        # axes[0].set_title("Expanded Image")
        
        # # Display the classification bar chart on the right
        # axes[1].bar(class_scores.keys(), class_scores.values())
        # axes[1].set_xticks(range(len(class_scores)))
        # axes[1].set_xticklabels(class_scores.keys(), rotation=45)
        # axes[1].set_xlabel("Classes")
        # axes[1].set_ylabel("Score")
        # axes[1].set_title(f"Classification: {best_class}")

        # Save only the scores image
        # plt.savefig(output_scores_img_path)
        # plt.close()

        results_json[f"image_{idx + 1}"] = {
            "best_class": best_class,
            "scores": class_scores,
            # "scores_image": output_scores_img_path
        }
        
        # print(f"Saved: {output_scores_img_path}")

@app.post('/statistic')
async def statistic(
    image: UploadFile,
):
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # 6. Process Image and Store Results
    # -------------------------------
    # image_path = "./data/test_ROI1.png"
    # image = Image.open(image_path).convert("RGB")

    image = Image.open(image.file).convert("RGB")
    # Start the timer for classification and result saving
    start_time = time.time()

    results = grid_classification(image)

    # Save all results
    save_final_results(results)

    # Output the class distribution (printing the counts)
    print("\nClass Distribution across all patches:")

    return class_count.items()

    # for cls, count in class_count.items():
    #     print(f"{cls}: {count}")

    # # Save classification results to JSON
    # json_output_path = os.path.join(output_dir, "classification_results.json")
    # with open(json_output_path, "w") as f:
    #     json.dump(results_json, f, indent=4)
    # print(f"Classification results saved to {json_output_path}")

    # # End the timer and print elapsed time
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Total processing time: {elapsed_time:.2f} seconds")
