import os
# fashion_train_path = './data/fashion-png/train'
fashion_train_path = './train_video2d'
fashion_train_video = os.listdir(fashion_train_path)
total_img_num = 0
for video_path in fashion_train_video:
    image_path = os.path.join(fashion_train_path, video_path)
    image = os.listdir(image_path)
    total_img_num += len(image)
print(total_img_num)
