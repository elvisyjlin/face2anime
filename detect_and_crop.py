import argparse
import cv2
import mimetypes
import os
from glob import glob
from smart_scale import smart_scale

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True)
    parser.add_argument('-d', '--destination', type=str, required=True)
    parser.add_argument('-c', '--cascade_file', type=str, required=True)
    parser.add_argument('-r', '--resize_image', type=int)
    parser.add_argument('-p', '--padding_ratio', type=float, default=1.0)
    parser.add_argument('-f', '--filter', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def detect_and_crop(in_path, out_path, cascade, 
                    resize=None, padding_ratio=None, 
                    debug=False):
    if not os.path.isfile(cascade):
        raise RuntimeError('{} not found'.format(cascade))
        
    os.makedirs(out_path, exist_ok=True)

    classifier = cv2.CascadeClassifier(cascade)
    count = []

    for in_file in glob(os.path.join(in_path, '*')):
        if debug: print(in_file)
        
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
            img_h, img_w, img_c = image.shape
            if img_h > 4096:
                if debug: print('Too large!!!')
                ratio = 4096 / img_h
                img_h = int(img_h * ratio)
                img_w = int(img_w * ratio)
                if debug: print(image.shape)
                image = cv2.resize(image, (img_w, img_h), interpolation = cv2.INTER_CUBIC)
                if debug: print(image.shape)
                import time
                time.sleep(10)
            
            if debug: print('Converting...')
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            except Exception as e:
                print(in_file)
                print(image.shape)
                raise e
            if debug: print('Equal hist...')
            gray = cv2.equalizeHist(gray)
            if debug: print('Detect...')
            faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

            for x, y, w, h in faces:
                if debug: print(image.shape, (x, y, w, h))
                pad = 0 if padding_ratio is None else h * padding_ratio
                # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                if resize is not None and filter and (w < resize or h < resize):
                    continue
                (x, y, w, h), _ = smart_scale((x, y, w, h), (0, 0, img_w, img_h), padding_ratio)
                out_image = image[y:y+h, x:x+w]
                if resize is not None:
                    out_image = cv2.resize(out_image, (resize, resize), interpolation = cv2.INTER_CUBIC)
                out_file = os.path.join(out_path, rreplace(filename, '.', '_{}.'.format(idx), 1))
                if not debug:
                    cv2.imwrite(out_file, out_image)
                if w != h:
                    print('Warning:', out_file, '{}x{}'.format(h, w))
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
    detect_and_crop(args.source, args.destination, args.cascade_file, args.resize_image, args.padding_ratio, args.debug)