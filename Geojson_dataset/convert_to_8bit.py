import numpy as np  # 用于数值计算和数组操作
from tqdm import tqdm  # 用于显示进度条
import os  # 用于文件和目录操作
import rasterio  # 用于读取和写入地理空间栅格数据
import cv2  # 用于图像处理

if __name__ == "__main__":
    # 定义输入数据目录列表和输出目录
    dir_path_list = [
        r"F:\NET\KDGraph\data\jiaozhouwan",
    ]
    other_save_path = r"F:\NET\KDGraph-main\spacenet_transform\data\spacenet\image"

    # 遍历每个输入目录
    for dir_path in dir_path_list:
        # 定义输入图像路径和输出目录
        image_path = os.path.join(dir_path, "1")
        output_dir = os.path.join(dir_path, "8bit")

        # 检查输入图像路径是否存在，若不存在则抛出异常
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The path {image_path} does not exist. Please check the directory structure.")

        # 如果输出目录不存在则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建CLAHE对象，用于直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))

        # 获取输入图像路径下的所有文件名，并创建进度条
        with tqdm(total=len(os.listdir(image_path))) as pbar:
            # 遍历每个图像文件
            for image_name in os.listdir(image_path):
                # 使用rasterio打开图像文件
                with rasterio.open(os.path.join(image_path, image_name)) as f:
                    # 复制输入图像的元数据
                    input_profile = f.profile.copy()

                    # 读取图像的三个波段（R、G、B），并转换为float32类型
                    R = f.read(1).astype(np.float32)
                    G = f.read(2).astype(np.float32)
                    B = f.read(3).astype(np.float32)

                    # 对每个波段进行归一化处理，将像素值缩放到0-255范围
                    R = 255.0 * ((R - np.min(R)) / (np.max(R) - np.min(R)))
                    G = 255.0 * ((G - np.min(G)) / (np.max(G) - np.min(G)))
                    B = 255.0 * ((B - np.min(B)) / (np.max(B) - np.min(B)))

                    # 对每个波段应用CLAHE进行直方图均衡化
                    R = clahe.apply(np.asarray(R, dtype=np.uint8))
                    G = clahe.apply(np.asarray(G, dtype=np.uint8))
                    B = clahe.apply(np.asarray(B, dtype=np.uint8))

                    # 将处理后的波段转换为uint8类型
                    R = R.astype(np.uint8)
                    G = G.astype(np.uint8)
                    B = B.astype(np.uint8)

                    # 复制输入图像的元数据，并修改数据类型为uint8
                    output_profile = input_profile.copy()
                    output_profile["dtype"] = "uint8"

                    # 定义输出文件路径
                    output_fn = os.path.join(output_dir, image_name)
                    output_fn2 = os.path.join(other_save_path, image_name)

                    # 使用rasterio将处理后的图像写入输出文件
                    with rasterio.open(output_fn, "w", **output_profile) as output:
                        output.write(R, 1)  # 写入红色波段
                        output.write(G, 2)  # 写入绿色波段
                        output.write(B, 3)  # 写入蓝色波段

                    # 更新进度条
                    pbar.update()