import mtutils as mt
import random
from lib import add_rainbow


def main(file_path):
    img = mt.cv_rgb_imread(file_path)
    res, bbox = add_rainbow(img)
    res_bbox = mt.draw_boxes(res, bbox, thickness=2)
    mt.PIS(res_bbox)
    pass

if __name__ == '__main__':
    img_list = mt.glob_recursively('assets/OK/', 'jpg')
    random.shuffle(img_list)
    # img_list = ['assets/1.jpg'] + img_list

    for file_path in mt.tqdm(img_list):
        main(file_path)
