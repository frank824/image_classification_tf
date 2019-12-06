import cv2 as cv
import os
import numpy as np

import random
import pickle

import time

start_time = time.time()

data_dir = './data'
batch_save_path = './batch_files'

# 创建batch文件存储的文件夹
os.makedirs(batch_save_path, exist_ok=True)

# 图片统一大小：100 * 100
# 训练集 20000：100个batch文件，每个文件200张图片
# 验证集 5000： 一个测试文件，测试时 50张 x 100 批次

# 进入图片数据的目录，读取图片信息
all_data_files = os.listdir(os.path.join(data_dir, 'train_company/'))

# print(all_data_files)
IMAGE_SIZE = 100
# 打算数据的顺序
random.shuffle(all_data_files)

all_train_files = all_data_files[:120]
all_test_files = all_data_files[120:]

train_data = []
train_label = []
train_filenames = []

test_data = []
test_label = []
test_filenames = []

count = 0
# 训练集
# 图片转为100x100尺寸
# 根据文件所属关键词归类图片标签
for each in all_train_files:
    img = cv.imread(os.path.join(data_dir, 'train_company/', each), 1)

    resized_img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_data = np.array(resized_img)
    train_data.append(img_data)
    if 'business_certificate' in each:
        train_label.append(0)
    elif 'certificate_copy' in each:
        train_label.append(1)
    elif 'doctor' in each:
        train_label.append(2)
    elif 'id' in each:
        train_label.append(3)
    elif 'sign' in each:
        train_label.append(4)
    else:
        raise Exception('%s is wrong train file' % each)
    count += 1
    print("resized %s, it's the %dth object" % (each, count))
    train_filenames.append(each)

# 测试集
count_test = 0
for each in all_test_files:
    img = cv.imread(os.path.join(data_dir, 'train_company/', each), 1)
    resized_img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_data = np.array(resized_img)
    test_data.append(img_data)
    if 'business_certificate' in each:
        test_label.append(0)
    elif 'certificate_copy' in each:
        test_label.append(1)
    elif 'doctor' in each:
        test_label.append(2)
    elif 'id' in each:
        test_label.append(3)
    elif 'sign' in each:
        test_label.append(4)
    else:
        raise Exception('%s is wrong train file' % each)
    count_test += 1

    print("resized %s, it's the %dth object" % (each, count_test))
    test_filenames.append(each)

print(str(len(train_data)) + " train images resized", str(len(test_data)) + " test images resized")

# 制作12个batch文件
# 每个batch文件中有10张图片，共120张训练图片
start = 0
end = 10
for num in range(1, 13):
    batch_data = train_data[start: end]
    batch_label = train_label[start: end]
    batch_filenames = train_filenames[start: end]
    batch_name = 'training batch {} of 12'.format(num)

    all_data = {
        'data': batch_data,
        'label': batch_label,
        'filenames': batch_filenames,
        'name': batch_name
    }

    with open(os.path.join(batch_save_path, 'train_batch_{}'.format(num)), 'wb') as f:
        pickle.dump(all_data, f)

    start += 10
    end += 10

# 制作测试文件
all_test_data = {
    'data': test_data,
    'label': test_label,
    'filenames': test_filenames,
    'name': 'test batch 1 of 1'
}

with open(os.path.join(batch_save_path, 'test_batch'), 'wb') as f:
    pickle.dump(all_test_data, f)


end_time = time.time()
print('制作结束, 用时{}秒'.format(end_time - start_time))
