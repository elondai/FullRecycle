import os
from PIL import Image
UNIT_SIZE = 256  # the size of image
TARGET_WIDTH = 4 * UNIT_SIZE
save_path = './merged/'
pathA = "/home/jupyter/Recycle-GAN/datasets/ObamaTrump/testA"
pathB = "/home/jupyter/Recycle-GAN/scripts/results/ObamaTrump_FullRecycle/test_latest/images"
pathC = "/home/jupyter/Recycle-GAN/scripts/results/ObamaTrump_cycle/test_latest/images"
pathD = "/home/jupyter/Recycle-GAN/scripts/results/ObamaTrump_Recycle/test_latest/images"

images = []  # all pic name


def pinjie():
    for img in sorted(os.listdir(pathA)):
        if os.path.splitext(img)[1] == '.png':
            images.append(img)
            print(img)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(images)):  # eyery 3 a group
        imagefile = []
        imagefile.append(Image.open(pathA+'/'+images[i]))
        if os.path.exists(pathB+'/'+images[i][0:5]+'_fake_B.png'):
            imagefile.append(Image.open(pathB+'/'+images[i][0:5]+'_fake_B.png'))
            imagefile.append(Image.open(pathC+'/'+images[i][0:5]+'_fake_B.png'))
            imagefile.append(Image.open(pathD+'/'+images[i][0:5]+'_fake_B.png'))
            
            target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE))  # width , height
            left = 0
            right = UNIT_SIZE
            for image in imagefile:
                target.paste(image, (left, 0, right, UNIT_SIZE))
                left += UNIT_SIZE
                right += UNIT_SIZE
                quality_value = 1000
            target.save(save_path+'out_{}.jpg'.format(i), quality=quality_value)


if __name__ == '__main__':
    print("starting merge...")
    pinjie()
