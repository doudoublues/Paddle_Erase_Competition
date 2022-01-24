import os
import random
image_dir = "datasets/dehw_train_dataset/images"
image_paths = os.listdir(image_dir)

random.shuffle(image_paths)
print(image_paths)

length = len(image_paths)

train_path = open('train_9.txt', 'w')
val_path = open('val_9.txt', 'w')


for i in range(length):
    if i < int(0.9*length):
        train_path.write(os.path.join(image_dir, image_paths[i]))
        train_path.write('\n')
    else:
        val_path.write(os.path.join(image_dir, image_paths[i]))
        val_path.write('\n')
