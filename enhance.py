from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.font_manager import FontProperties
import argparse
from datetime import datetime

def setup_fonts():
    """设置字体，优先使用中文字体，如果找不到则使用英文"""
    try:
        # 尝试加载系统中可能存在的中文字体
        font_paths = [
            '/System/Library/Fonts/STHeiti Light.ttc',  # macOS
            '/System/Library/Fonts/PingFang.ttc',       # macOS
            'C:/Windows/Fonts/simhei.ttf',              # Windows
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = FontProperties(fname=path)
                break
        
        if font is None:
            # 如果找不到中文字体，使用sans-serif并使用英文标题
            font = FontProperties(family='sans-serif')
            use_english = True
        else:
            use_english = False
            
    except:
        # 如果出现任何异常，使用默认字体和英文
        font = FontProperties(family='sans-serif')
        use_english = True
        
    return font, use_english

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

def process_building_image(building_img_array, threshold=0.21):
    """处理建筑物图像并应用阈值"""
    if building_img_array is None:
        return None
    
    # 应用阈值处理
    thresholded = np.where(building_img_array < threshold, 1, 0)
    return thresholded

def process_green_areas(img_hsv):
    """处理图像中的绿色区域（原红色区域）"""
    if img_hsv is None:
        return None
    
    # 定义红色的HSV范围
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩码
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    green_area_mask = mask1 + mask2
    
    # 创建二值化图像: 红色区域为黑色(0)，其他区域为白色(1)
    result_green = np.where(green_area_mask > 0, 0, 1)
    return result_green

def process_lake_areas(img_hsv, black_threshold=30):
    """处理图像中的湖泊区域（原接近黑色区域）"""
    if img_hsv is None:
        return None
    
    # 提取V通道（亮度）
    v_channel = img_hsv[:,:,2]
    
    # 设置黑色阈值 - V值小于阈值被认为是接近黑色的
    lake_area_mask = v_channel < black_threshold
    
    # 创建二值化图像: 黑色区域为黑色(0)，其他区域为白色(1)
    result_lake = np.where(lake_area_mask, 0, 1)
    return result_lake

def create_overlay_image(ori_img_array, building_mask, green_mask, lake_mask):
    """创建带有彩色掩码的叠加图像"""
    if (ori_img_array is None or building_mask is None or 
        green_mask is None or lake_mask is None):
        return None
    
    # 获取原始图像尺寸
    ori_height, ori_width = ori_img_array.shape[:2]
    
    # 创建一个原始图像的副本
    overlay_img = np.copy(ori_img_array).astype(float)
    
    # 调整所有掩码到与原始图像相同的尺寸
    resized_building_mask = cv2.resize(building_mask.astype(np.uint8), (ori_width, ori_height), 
                                     interpolation=cv2.INTER_NEAREST)
    resized_green_mask = cv2.resize(green_mask.astype(np.uint8), (ori_width, ori_height), 
                                   interpolation=cv2.INTER_NEAREST)
    resized_lake_mask = cv2.resize(lake_mask.astype(np.uint8), (ori_width, ori_height), 
                                  interpolation=cv2.INTER_NEAREST)
    
    # 将掩码转换为布尔数组（黑色区域为True）
    building_mask_bool = (resized_building_mask == 0)
    green_mask_bool = (resized_green_mask == 0)
    lake_mask_bool = (resized_lake_mask == 0)
    
    # 应用掩码并添加颜色（红色、绿色、蓝色）
    # 红色掩码 - 建筑物
    alpha_red = 0.5  # 红色透明度
    overlay_img[building_mask_bool, 0] = overlay_img[building_mask_bool, 0] * (1-alpha_red) + 255 * alpha_red
    
    # 绿色掩码 - 绿地
    alpha_green = 0.5  # 绿色透明度
    overlay_img[green_mask_bool, 1] = overlay_img[green_mask_bool, 1] * (1-alpha_green) + 255 * alpha_green
    
    # 蓝色掩码 - 湖泊
    alpha_blue = 0.5  # 蓝色透明度
    overlay_img[lake_mask_bool, 2] = overlay_img[lake_mask_bool, 2] * (1-alpha_blue) + 255 * alpha_blue
    
    return overlay_img.astype(np.uint8)

def create_labeled_matrix(building_mask, green_mask, lake_mask, size=(200, 200)):
    """
    创建一个标签矩阵，将不同区域分类为0、1、2
    0: 建筑物区域（原LiDAR处理区域）
    1: 湖泊区域（原接近黑色区域）
    2: 绿地区域（原红色区域）
    """
    if building_mask is None or green_mask is None or lake_mask is None:
        return None
    
    # 创建一个指定大小的全零矩阵（默认为200x200）
    labeled_matrix = np.zeros(size, dtype=np.uint8)
    
    # 调整所有掩码到相同的大小
    resized_building_mask = cv2.resize(building_mask.astype(np.uint8), size, 
                                     interpolation=cv2.INTER_NEAREST)
    resized_green_mask = cv2.resize(green_mask.astype(np.uint8), size, 
                                   interpolation=cv2.INTER_NEAREST)
    resized_lake_mask = cv2.resize(lake_mask.astype(np.uint8), size, 
                                  interpolation=cv2.INTER_NEAREST)
    
    # 将掩码转换为布尔数组（黑色区域为True）
    building_mask_bool = (resized_building_mask == 0)
    green_mask_bool = (resized_green_mask == 0)
    lake_mask_bool = (resized_lake_mask == 0)
    
    # 标记不同区域为不同的值
    # 首先全部设为背景（这里不使用背景类别，所有像素都会被分类）
    
    # 0: 建筑物区域
    labeled_matrix[building_mask_bool] = 0
    
    # 1: 湖泊区域
    labeled_matrix[lake_mask_bool] = 1
    
    # 2: 绿地区域
    labeled_matrix[green_mask_bool] = 2
    
    return labeled_matrix

def save_matrix(matrix, filename):
    """保存矩阵为文件"""
    if matrix is None:
        print("错误: 无矩阵可保存")
        return False
    
    try:
        # 保存为NumPy二进制文件
        np.save(filename + '.npy', matrix)
        
        # 同时保存为CSV便于查看
        np.savetxt(filename + '.csv', matrix, delimiter=',', fmt='%d')
        
        # 创建可视化图像
        plt.figure(figsize=(10, 10))
        plt.imshow(matrix, cmap='viridis')  # 使用viridis色图，不同值有不同颜色
        plt.colorbar(label='类别')
        plt.title('标签矩阵可视化')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename + '.png', dpi=300)
        plt.close()
        
        print(f"矩阵已保存: {filename}.npy, {filename}.csv, {filename}.png")
        return True
    except Exception as e:
        print(f"保存矩阵时出错: {e}")
        return False

def save_image(img_array, filename, cmap=None):
    """保存图像到文件"""
    if img_array is None:
        print("错误: 无图像可保存")
        return False
    
    if not filename:
        print("错误: 未提供文件名")
        return False
    
    try:
        if cmap == 'gray':
            # 对于灰度图像，确保值范围在0-255之间
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            plt.imsave(filename, img_array, cmap='gray')
        else:
            # 对于彩色图像
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            
            if len(img_array.shape) == 2:
                # 如果是二维数组，转换为三通道
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            # 使用OpenCV保存，需要BGR顺序
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(filename, img_array)
        
        print(f"图像已保存: {filename}")
        return True
    except Exception as e:
        print(f"保存图像 {filename} 时出错: {e}")
        return False

def create_output_directory(output_dir="output"):
    """创建输出目录，如果不存在"""
    # 添加时间戳到输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    return output_dir

def display_images_and_save(ori_img_array, building_img_array, img_rgb, building_mask, 
                           green_mask, lake_mask, overlay_img, labeled_matrix,
                           output_dir, font, use_english):
    """显示所有图像并保存处理后的图像"""
    # 创建一个3x3的图形布局
    plt.figure(figsize=(18, 15))
    
    # 第一行显示原始图像
    if ori_img_array is not None:
        plt.subplot(3, 3, 1)
        title = "Original Image" if use_english else "原始图像"
        plt.title(title, fontproperties=font)
        plt.imshow(ori_img_array)
        plt.axis('off')
    
    if building_img_array is not None:
        plt.subplot(3, 3, 2)
        title = "Building Image" if use_english else "建筑物图像"
        plt.title(title, fontproperties=font)
        plt.imshow(building_img_array, cmap='gray')
        plt.axis('off')
    
    if img_rgb is not None:
        plt.subplot(3, 3, 3)
        title = "Color Image" if use_english else "彩色图像"
        plt.title(title, fontproperties=font)
        plt.imshow(img_rgb)
        plt.axis('off')
    
    # 第二行显示处理后的图像
    if building_mask is not None:
        plt.subplot(3, 3, 4)
        title = "Building Areas" if use_english else "建筑物区域"
        plt.title(title, fontproperties=font)
        plt.imshow(building_mask, cmap='gray')
        plt.axis('off')
        
        # 保存处理后的建筑物图像
        if output_dir:
            save_image(building_mask, os.path.join(output_dir, "building_processed.png"), cmap='gray')
    
    if green_mask is not None:
        plt.subplot(3, 3, 5)
        title = "Green Areas" if use_english else "绿地区域"
        plt.title(title, fontproperties=font)
        plt.imshow(green_mask, cmap='gray')
        plt.axis('off')
        
        # 保存绿地处理图像
        if output_dir:
            save_image(green_mask, os.path.join(output_dir, "green_processed.png"), cmap='gray')
    
    if lake_mask is not None:
        plt.subplot(3, 3, 6)
        title = "Lake Areas" if use_english else "湖泊区域"
        plt.title(title, fontproperties=font)
        plt.imshow(lake_mask, cmap='gray')
        plt.axis('off')
        
        # 保存湖泊处理图像
        if output_dir:
            save_image(lake_mask, os.path.join(output_dir, "lake_processed.png"), cmap='gray')
    
    # 第三行显示图例和叠加图像
    plt.subplot(3, 3, 7)
    title = "Color Legend" if use_english else "颜色图例"
    plt.title(title, fontproperties=font)
    plt.axis('off')
    
    # 创建图例方块和文本
    legend_text = []
    if use_english:
        legend_text = [
            "Red: Building areas",
            "Green: Green areas",
            "Blue: Lake areas"
        ]
    else:
        legend_text = [
            "红色: 建筑物区域",
            "绿色: 绿地区域",
            "蓝色: 湖泊区域"
        ]
    
    # 绘制图例
    for i, text in enumerate(legend_text):
        color = ['red', 'green', 'blue'][i]
        plt.text(0.1, 0.8 - i*0.2, text, fontproperties=font, 
                 bbox=dict(facecolor=color, alpha=0.5), fontsize=12)
    
    if overlay_img is not None:
        plt.subplot(3, 3, 8)
        title = "Image with Color Masks" if use_english else "带彩色掩码的图像"
        plt.title(title, fontproperties=font)
        plt.imshow(overlay_img)
        plt.axis('off')
        
        # 保存叠加图像
        if output_dir:
            save_image(overlay_img, os.path.join(output_dir, "overlay_image.png"))
    
    if labeled_matrix is not None:
        plt.subplot(3, 3, 9)
        title = "Label Matrix (0:Building, 1:Lake, 2:Green)" if use_english else "标签矩阵 (0:建筑, 1:湖泊, 2:绿地)"
        plt.title(title, fontproperties=font)
        plt.imshow(labeled_matrix, cmap='viridis')
        plt.axis('off')
        
        # 保存标签矩阵
        if output_dir:
            save_matrix(labeled_matrix, os.path.join(output_dir, "label_matrix"))
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "all_images.png"), dpi=300)
    plt.show()

def process_images(ori_path, building_path, color_path, output_dir="output", save_images=True):
    """主处理函数，处理所有图像并显示结果"""
    # 设置字体
    font, use_english = setup_fonts()
    
    # 如果需要保存图像，创建输出目录
    if save_images:
        output_dir = create_output_directory(output_dir)
    else:
        output_dir = None
    
    # 加载原始图像
    ori_img_array = None
    if os.path.exists(ori_path):
        img_ori = Image.open(ori_path)
        if img_ori.mode != 'RGB':
            img_ori = img_ori.convert('RGB')
        ori_img_array = np.array(img_ori)
    else:
        print(f"错误: 文件 {ori_path} 不存在")
    
    # 加载并预处理建筑物图像
    building_img_array = load_building_image(building_path)
    
    # 加载彩色图像
    img_rgb, img_hsv = load_color_image(color_path)
    
    # 处理图像
    building_mask = process_building_image(building_img_array)
    green_mask = process_green_areas(img_hsv)
    lake_mask = process_lake_areas(img_hsv)
    
    # 创建叠加图像
    overlay_img = None
    if ori_img_array is not None and building_mask is not None and green_mask is not None and lake_mask is not None:
        overlay_img = create_overlay_image(ori_img_array, building_mask, green_mask, lake_mask)
    
    # 创建200x200的标签矩阵 (0:Building, 1:Lake, 2:Green)
    labeled_matrix = create_labeled_matrix(building_mask, green_mask, lake_mask, size=(200, 200))
    
    # 显示图像并保存
    display_images_and_save(ori_img_array, building_img_array, img_rgb, building_mask, 
                           green_mask, lake_mask, overlay_img, labeled_matrix,
                           output_dir, font, use_english)

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='处理建筑物、绿地和湖泊图像')
    
    parser.add_argument('--ori', type=str, default="ori.jpg", help='原始图像路径')
    parser.add_argument('--building', type=str, default="lidar.png", help='建筑物图像路径（原LiDAR）')
    parser.add_argument('--color', type=str, default="red.jpg", help='彩色图像路径（原红色图像）')
    parser.add_argument('--output', type=str, default="output", help='输出目录')
    parser.add_argument('--no-save', action='store_false', dest='save', help='不保存图像')
    
    args = parser.parse_args()
    
    process_images(args.ori, args.building, args.color, args.output, args.save)

if __name__ == "__main__":
    main()
