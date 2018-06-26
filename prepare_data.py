import argparse
import os
from glob import glob
from skimage import io
from tqdm

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_folders', type=str, nargs = '*', required=True)
    parser.add_argument('-d', '--destination_folder', type=str, required=True)
    parser.add_argument('-f', '--image_format', type=str, default='jpg', choices=['jpg', 'png'])
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
        io.imsave(os.path.join(destination_folder, str(idx)+'.'+image_format), im)