import argparse
import cv2
import sys
import os
import mimetypes
from glob import glob

PAD_RATIO = 6

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True)
    parser.add_argument('-d', '--destination', type=str, required=True)
    parser.add_argument('-c', '--cascade_file', type=str, required=True)
    parser.add_argument('-r', '--resize_image', type=int)
    return parser.parse_args()

def detect_and_crop(in_path, out_path, cascade, resize=None):
    if not os.path.isfile(cascade):
        raise RuntimeError('{} not found'.format(cascade))
        
    os.makedirs(out_path, exist_ok=True)

    classifier = cv2.CascadeClassifier(cascade)
    count = []

    for in_file in glob(os.path.join(in_path, '*')):
        print(in_file)
        
        if mimetypes.guess_type(in_file)[0].split('/')[0] != 'image':
            continue
            
        images = []
        filename = os.path.basename(in_file)
        
        if filename.rsplit('.', 1)[1] == 'gif':
            continue
            
            gif = cv2.VideoCapture(filename)
            # Loop until there are frames left
            while True:
                try:
                    # Try to read a frame. Okay is a BOOL if there are frames or not
                    okay, frame = gif.read()
                    # Append to empty frame list
                    images.append(frame)
                    # Break if there are no other frames to read
                    if not okay:
                        break
                    # # Increment value of the frame number by 1
                    # frame_num += 1
                except KeyboardInterrupt:  # press ^C to quit
                    break
            print(len(images))
        else:
            image = cv2.imread(in_file, cv2.IMREAD_COLOR)
            images.append(image)
            
        idx = 0
        for image in images:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            except Exception as e:
                print(in_file)
                print(image.shape)
                raise e
            gray = cv2.equalizeHist(gray)
            faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

            for x, y, w, h in faces:
                pad = h / PAD_RATIO
                # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                out_image = image[y:y+h, x:x+w]
                if resize is not None:
                    out_image = cv2.resize(out_image, (resize, resize), interpolation = cv2.INTER_CUBIC)
                out_file = os.path.join(out_path, rreplace(filename, '.', '_{}.'.format(idx), 1))
                cv2.imwrite(out_file, out_image)
                idx += 1

            # cv2.imshow('Detect', image)
            # cv2.waitKey(0)
            # cv2.imwrite(out_file, image)

            print(filename, image.shape, len(faces))
            count.append(len(faces))
        
    print('Detected and cropped', len(count), 'images!')
    print('Detected and cropped', sum(count), 'faces!')

if __name__ == '__main__':
    args = parse()
    print(args)
    detect_and_crop(args.source, args.destination, args.cascade_file, args.resize_image)
    # detect('data/orig','data/2', 'haarcascade_frontalface_default.xml')
    # detect('data/orig','data/3', 'lbpcascade_frontalface.xml')
    # detect('data/orig','data/4', 'lbpcascade_profileface.xml')