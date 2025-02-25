import imageio
from PIL import Image, ImageDraw
import json
import numpy as np
import os
import cv2
import sys
sys.path.append(os.path.abspath('.'))
from arguments import get_parser

parser = get_parser()
args = parser.parse_args()
def visualize():
    image_list = os.listdir('./records/test/skeleton')
    for ii,image_name in enumerate(image_list):

        image_name1 = image_name.strip("[]")  # 去掉两侧的方括号
        image_name1 = image_name1.strip("'")  # 去掉两侧的单引号
        # print(image_name[:-6])
        file_path = os.path.join('E:/point/Topo-boundary-master/coastline/image', image_name1[:-6] + '.tif')
        # print(file_path)
        # im1 = Image.open('E:/point/Topo-boundary-master/dataset/cropped_tiff/', image_name1[:-6] + '.png')
        image = imageio.imread(file_path)[:, :, [2, 1, 0]]
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

        # image = Image.fromarray(np.array(Image.open(file_path))[:,:,:3])
        p = image.load()
        im = cv2.imread(os.path.join('./records/test/skeleton',image_name))
        kernel = np.ones((1,1), np.uint8)
        dilate = cv2.dilate(im, kernel)

        foreground_pixels = np.where(dilate[:,:,0]!=0)
        foreground_pixels = [[foreground_pixels[1][x],foreground_pixels[0][x]] for x in range(len(foreground_pixels[0]))]
        for point in foreground_pixels:
            p[int(point[0]),int(point[1])] = (255,165,0)

        draw = ImageDraw.Draw(image)
        with open('./records/test/vertices_record/{}'.format(image_name[:-3]+'json'),'r') as jf:
            points = json.load(jf)
        for point in points:
            draw.ellipse((point[1]-1,point[0]-1,point[1]+1,point[0]+1),fill='yellow',outline='yellow')
        image.save(os.path.join('./records/test/final_vis',image_name))
        print('Visualizing',ii,'/',len(image_list))

def main():
    visualize()

if __name__=='__main__':
    main()