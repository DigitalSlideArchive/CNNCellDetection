

import os

png_files = os.listdir('./pascalvoc_format_data/JPEGImages/')
anot_files = os.listdir('./pascalvoc_format_data/Annotations/')
filename = './pascalvoc_format_data/ImageSets/Main/train.txt'
classfilename = './pascalvoc_format_data/ImageSets/Main/objectness_train.txt'


print png_files

with open(filename, "w") as file:
    for png_file in png_files:
        file.write(png_file[:-4] + '\n')


with open(classfilename, "w") as file:
    for png_file in png_files:
        file.write(png_file[:-4] + ' 1 \n')
