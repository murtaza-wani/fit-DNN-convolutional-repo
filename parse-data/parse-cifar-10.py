import numpy as np
import pickle


def unpickle(file):    
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def make_files(data_batch, img_filename, labels_filename):
    img_array = data_batch[b"data"]
    with open(img_filename, "w") as img_file:
        for line_index in range(10000):
            img_ints = img_array[line_index, :]
            img_ints_reshaped = np.zeros(3072, dtype=np.int64)
            for row in range(32):
                for col in range(32):
                    r = img_ints[32*row + col]
                    g = img_ints[1024 + 32*row + col]
                    b = img_ints[2048 + 32*row + col]
                    img_ints_reshaped[32*row + col] = r
                    img_ints_reshaped[1024 + 32*row + col] = g
                    img_ints_reshaped[2048 + 32*row + col] = b
            line = ""
            for value in img_ints_reshaped:
                hex_code = hex(value)[2:]
                if len(hex_code) == 1:
                    hex_code = "0" + hex_code
                line += hex_code
            line += "\n"
            img_file.write(line)    
    label_list = data_batch[b"labels"]
    with open(labels_filename, "w") as labels_file:
        line = ""
        for number in label_list:
            line += str(number)
        line += "\n"
        labels_file.write(line)
    return




for i in range(1,6):
    fname = "data_batch_" + str(i)
    img_filename = "train_images_" + str(i) + ".txt"
    labels_filename = "train_labels_" + str(i) + ".txt"
    data_batch = unpickle(fname)
    make_files(data_batch, img_filename, labels_filename)

fname = "test_batch"
img_filename = "test_images.txt"
labels_filename = "test_labels.txt"
data_batch = unpickle(fname)
make_files(data_batch, img_filename, labels_filename)



