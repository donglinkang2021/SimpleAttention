import os
import imageio

# 指定要读取文件的目录
image_folder = './fin'

# 指定帧率
fps = 16

# 创建GIF
gif_name = f'{image_folder}_{fps}.gif'

# for png_name in os.listdir(image_folder):
#     if png_name.endswith(".png"):
#         number = png_name.split("_")[-3]
#         # print(number)
#         os.rename(f"{image_folder}/{png_name}", f"{image_folder}/{number:0>3}.png")

with imageio.get_writer(gif_name, mode='I', fps=fps) as writer:
    for image in os.listdir(image_folder):
        img = imageio.imread(os.path.join(image_folder, image))
        writer.append_data(img)

print(f"GIF已保存为 {gif_name}")