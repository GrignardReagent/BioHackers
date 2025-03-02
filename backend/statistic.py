import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoTokenizer
import time

# -------------------------------
# 1. Ensure output directory exists
# -------------------------------
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

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
# 6. Process Image and Store Results
# -------------------------------
image_path = "./data/test_ROI1.png"
image = Image.open(image_path).convert("RGB")

# Start the timer for classification and result saving
start_time = time.time()

results = grid_classification(image)

# -------------------------------
# 7. Save Classification Results and Track Class Frequencies
# -------------------------------
results_json = {}
class_count = {cls: 0 for cls in ["residential", "industrial", "waterway", "green belt", "farm"]}

def save_final_results(results):
    global class_count
    for idx, (patch, best_class, class_scores, expanded_image) in enumerate(results):
        # Only save the scores images (with _scores.png suffix)
        output_scores_img_path = os.path.join(output_dir, f"image_{idx + 1}_scores.png")
        
        # Update class count using the dictionary
        class_count[best_class] += 1
        
        # Plot expanded image and classification scores side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display the expanded image on the left
        axes[0].imshow(expanded_image)
        axes[0].axis("off")
        axes[0].set_title("Expanded Image")
        
        # Display the classification bar chart on the right
        axes[1].bar(class_scores.keys(), class_scores.values())
        axes[1].set_xticks(range(len(class_scores)))
        axes[1].set_xticklabels(class_scores.keys(), rotation=45)
        axes[1].set_xlabel("Classes")
        axes[1].set_ylabel("Score")
        axes[1].set_title(f"Classification: {best_class}")

        # Save only the scores image
        plt.savefig(output_scores_img_path)
        plt.close()

        results_json[f"image_{idx + 1}"] = {
            "best_class": best_class,
            "scores": class_scores,
            "scores_image": output_scores_img_path
        }
        
        print(f"Saved: {output_scores_img_path}")

# Save all results
save_final_results(results)

# Output the class distribution (printing the counts)
print("\nClass Distribution across all patches:")
for cls, count in class_count.items():
    print(f"{cls}: {count}")

# Save classification results to JSON
json_output_path = os.path.join(output_dir, "classification_results.json")
with open(json_output_path, "w") as f:
    json.dump(results_json, f, indent=4)
print(f"Classification results saved to {json_output_path}")

# End the timer and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total processing time: {elapsed_time:.2f} seconds")
