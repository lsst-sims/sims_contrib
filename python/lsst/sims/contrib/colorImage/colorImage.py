import numpy as np
import pyfits
import argparse
import Image

def scaleLinear(imageList, zmin=None, zmax=None,):
    '''Scale list of images using a linear ramp'''
    scaledImages = []

    for image in imageList:
        if (zmin==None):
            zmin = image.min()
        if (zmax==None):
            zmax = image.max()

    for image in imageList:
        image = np.where(image > zmin, image, zmin)
        image = np.where(image < zmax, image, zmax)
        scaledImage = (image - zmin) * (255.0 / (zmax - zmin)) + 0.5

        # convert to 8 bit unsigned int ("b" in numpy)
        scaledImages.append(scaledImage.astype('uint8'))

    return scaledImages


def scaleFunc(img, m, alpha=0.01, Q=10.):
    return np.arcsinh(alpha*Q*(img-m))/Q

def scaleAsinh(imageList, zmin = None, zmax = None, alpha = 0.01, Q = 3, scales = [4.9, 5.7, 7.8]):
    """Scale list of images using asinh"""
    for i, (image, scale) in enumerate(zip(imageList, scales)):
        image *= scale
        image = np.where(image > 0.0, image, 0.0)
        imageList[i] = image

    total = np.array(imageList).sum(axis=0) / 3.0
    if (zmin == None):
        zmin = total.min()
    if (zmax == None):
        zmax = total.max()

    R, G, B = imageList * scaleFunc(total, zmin, alpha, Q) / total
#    G = imageList[1] * scaleFunc(total, zmin, alpha, Q) / total
#    B = imageList[2] * scaleFunc(total, zmin, alpha, Q) / total
    
    maxRGB = np.maximum(np.maximum(R, G), B)
    R = np.where(maxRGB > 1, R / maxRGB, R)
    G = np.where(maxRGB > 1, G / maxRGB, G)
    B = np.where(maxRGB > 1, B / maxRGB, B)
    
    R = np.where(maxRGB < 0, 0, R)
    G = np.where(maxRGB < 0, 0, G)
    B = np.where(maxRGB < 0, 0, B)

    scaledImages = []
    for image in (R, G, B):
        scaledImage = image * 255.0 + 0.5
        scaledImages.append(scaledImage.astype('uint8'))

    return scaledImages


def createRGB(imageList):
    '''Create RGB image from scale numpy arrays assuming order R,G,B'''
    mode='RGB'

    im = []
    for image in imageList:
        im.append(Image.fromarray(image, mode='L'))
    image = Image.merge('RGB', tuple(im))

    return image

def addNoise (imageList, noise):
    '''Add Gaussian noise to each image in a list given an array of sigma values in noise'''

    for image,sigma in zip(imageList,noise):
        image += np.random.normal(loc=0.0, scale=sigma, size=image.shape)

def main():
    '''Input a set of R G B images and return a color image'''

    # Setup command line options
    parser = argparse.ArgumentParser(description='Coadd R,G,B fits images and return color png')
    parser.add_argument("rFile", help='Input R image', default='r.fits')
    parser.add_argument("gFile", help='Input G image', default='g.fits')
    parser.add_argument("bFile", help='Input B image', default='b.fits')
    parser.add_argument('--addNoise', dest="noise", nargs='+', type=float, help='Gaussian noise to add to each image', default=[0,0,0])
    parser.add_argument('--outputFile', dest="outputFile", help='Output PNG file name', default='rgb.png')
    args = parser.parse_args()

    images = []
    images.append(pyfits.getdata(args.rFile,0))
    images.append(pyfits.getdata(args.gFile,0))
    images.append(pyfits.getdata(args.bFile,0))

    # add noise
    if (args.noise != [0,0,0]):
        addNoise(images,args.noise)

    # scale image
    scaledImages = scaleAsinh(images)

    # create RGB
    mode='RGB'
    pngImage = Image.new(mode, scaledImages[0].shape)
    pngImage.paste(createRGB(scaledImages),(0,0))

    pngImage.save(args.outputFile)

if __name__ == "__main__":
    main()
