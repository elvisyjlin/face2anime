import argparse
import os
from glob import glob
from skimage import io
from skimage.transform import resize
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_folders', type=str, nargs = '*', required=True)
    parser.add_argument('-d', '--destination_folder', type=str, required=True)
    parser.add_argument('-f', '--image_format', type=str, default='jpg', choices=['jpg', 'png'])
    parser.add_argument('-r', '--resize_image', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    print(args)
    source_folders = args.source_folders
    destination_folder = args.destination_folder
    image_format = args.image_format
    
    file_list = []
    for src_path in source_folders:
        file_list += sorted(glob(src_path + '/*'))
    print('# of total images:', len(file_list))
    
    os.makedirs(destination_folder, exist_ok=True)
    
    for idx, file in tqdm(enumerate(file_list)):
        im = io.imread(file)
        if args.resize_image is not None:
            im = resize(im, (args.resize_image, args.resize_image))
        io.imsave(os.path.join(destination_folder, str(idx)+'.'+image_format), im)