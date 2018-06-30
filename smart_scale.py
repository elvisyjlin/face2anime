def smart_scale(rect1, rect2, scale=1.0):
    '''
    rect1: (x, y, w, h)
    rect2: (x, y, w, h)
    scale: float

    rect1 is in rect2
    '''
    assert scale > 0.0
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    if w1 * scale > w2:
        scale = w2 / w1
    if h1 * scale > h2:
        scale = h2 / h1
    
    x1 -= (w1 * (scale - 1)) // 2
    y1 -= (h1 * (scale - 1)) // 2
    y1 -= h1 // 5
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    
    if x1 + w1*scale > w2:
        x1 -= int(x1 + w1*scale - w2)
    if y1 + h1*scale > h2:
        y1 -= int(y1 + h1*scale - h2)

    return (int(x1), int(y1), int(w1*scale), int(h1*scale)), (x2, y2, w2, h2)

def check(rect1, rect2):
    '''
    rect1: (x, y, w, h)
    rect2: (x, y, w, h)
    scale: float

    rect1 is in rect2
    '''
    import numpy as np
    print(rect1, rect2)
    x1, y1, w1, h1 = rect1
    im = np.full((rect2[2], rect2[3], 3), 255, dtype=np.uint8)
    print(im[x1:x1+w1, y1:y1+h1].shape)

def draw(rect1, rect2):
    '''
    rect1: (x, y, w, h)
    rect2: (x, y, w, h)
    scale: float

    rect1 is in rect2
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    fig, ax = plt.subplots(1)
    im = np.full((rect2[2], rect2[3], 3), 255, dtype=np.uint8)
    ax.imshow(im)

    rect = patches.Rectangle(
        (rect1[0], rect1[1]), rect1[2], rect1[3], 
        linewidth=1, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    import random
    min_size = 100
    max_size = 800
    rect2 = (0, 0, max_size, max_size)
    num_tests = 10
    for i in range(num_tests):
        x = random.randint(0, max_size-1)
        y = random.randint(0, max_size-1)
        size = random.randint(min_size, max_size)
        rect1 = (x, y, size, size)
        rect1, rect2 = smart_scale(rect1, rect2, 1.0)
        check(rect1, rect2)
        draw(rect1, rect2)