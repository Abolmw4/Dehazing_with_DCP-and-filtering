import numpy as np
import cv2
from itertools import combinations_with_replacement
from collections import defaultdict
from numpy.linalg import inv


class node:
    def __init__(self,x,y,value):
        self.x=x
        self.y=y
        self.value=value
class main:
    def __init__(self,image,blockSize=3,meanMode=False,percent=0.001):
        self.image=image
        self.blockSize=blockSize
        self.meanMode=meanMode
        self.percent=percent

    @property
    def image(self):
        return self._image
    @property
    def blockSize(self):
        return self._blockSize
    @image.setter
    def image(self,value):
        if value.ndim !=3:
            raise ValueError('Image must be RGB')
        self._image=np.min(value, axis=2)
    @blockSize.setter
    def blockSize(self,size):
        if size % 2 == 0 or size < 3:
            raise ValueError('blockSize is not odd or too small')
        self._blockSize=size
    
    def getDarkChannel(self):
        #AddSize
        A = int((self.blockSize-1)/2) 

        #New height and new width
        H = self.image.shape[0] + self.blockSize - 1
        W = self.image.shape[1] + self.blockSize - 1
    
        # 中间结果
        imgMiddle = 255 * np.ones((H,W))    

        imgMiddle[A:H-A, A:W-A] = self.image
    
        imgDark = np.zeros_like(self.image, np.uint8)    
    
        localMin = 255
        for i in range(A, H-A):
            for j in range(A, W-A):
                x = range(i-A, i+A+1)
                y = range(j-A, j+A+1)
                imgDark[i-A,j-A] = np.min(imgMiddle[x,y])                            
            
        return imgDark
    
    def getAtomsphericLight(self,picture):
        img=self.getDarkChannel()
        size = img.shape[0] * img.shape[1]
        height = img.shape[0]
        width = img.shape[1]
        nodes = []

        for i in range(0,height):
            for j in range(0,width):
                oneNode = node(i,j,img[i,j])
                nodes.append(oneNode)
        nodes = sorted(nodes , key = lambda node: node.value , reverse = True)
        atomsphericLight = 0

        # 原图像像素过少时，只考虑第一个像素点
        if int(self.percent*size) == 0:
            for i in range(0,3):
                if picture[nodes[0].x,nodes[0].y,i] > atomsphericLight:
                    atomsphericLight = picture[nodes[0].x,nodes[0].y,i]
            return atomsphericLight

        # 开启均值模式
        if self.meanMode:
            sum = 0
            for i in range(0,int(self.percent*size)):
                for j in range(0,3):
                    sum += picture[nodes[i].x,nodes[i].y,j]
            atomsphericLight = int(sum/(int(self.percent * size)*3))
            return atomsphericLight
        
        for i in range(0,int(self.percent*size)):
            for j in range(0,3):
                if picture[nodes[i].x,nodes[i].y,j] > atomsphericLight:
                    atomsphericLight = picture[nodes[i].x,nodes[i].y,j]
        return atomsphericLight
    
    def boxfilter(self,I, r):
        """Fast box filter implementation.
        Parameters
        ----------
        I:  a single channel/gray image data normalized to [0.0, 1.0]
        r:  window radius
        Return
        -----------
        The filtered image data.
        """        
        M, N = I.shape
        dest = np.zeros((M, N))

        # cumulative sum over Y axis
        sumY = np.cumsum(I, axis=0)
        # difference over Y axis
        dest[:r + 1] = sumY[r: 2 * r + 1]
        dest[r + 1:M - r] = sumY[2 * r + 1:] - sumY[:M - 2 * r - 1]
        dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2 * r - 1:M - r - 1]

        # cumulative sum over X axis
        sumX = np.cumsum(dest, axis=1)
        # difference over Y axis
        dest[:, :r + 1] = sumX[:, r:2 * r + 1]
        dest[:, r + 1:N - r] = sumX[:, 2 * r + 1:] - sumX[:, :N - 2 * r - 1]
        dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - \
            sumX[:, N - 2 * r - 1:N - r - 1]

        return dest

    def guided_filter(self,I, p, r=40, eps=1e-3):
        M, N = p.shape
        base = self.boxfilter(np.ones((M, N)), r)

        # each channel of I filtered with the mean filter
        means = [self.boxfilter(I[:, :, i], r) / base for i in range(3)]
        # p filtered with the mean filter
        mean_p = self.boxfilter(p, r) / base
        # filter I with p then filter it with the mean filter
        means_IP = [self.boxfilter(I[:, :, i] * p, r) / base for i in range(3)]
        # covariance of (I, p) in each local patch
        covIP = [means_IP[i] - means[i] * mean_p for i in range(3)]
    
        # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
        var = defaultdict(dict)
        for i, j in combinations_with_replacement(range(3), 2):
            var[i][j] = self.boxfilter(
                I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

        a = np.zeros((M, N, 3))
        for y, x in np.ndindex(M, N):
            #         rr, rg, rb
            # Sigma = rg, gg, gb
            #         rb, gb, bb
            Sigma = np.array([[var[0][0][y, x], var[0][1][y, x], var[0][2][y, x]],
                              [var[0][1][y, x], var[1][1][y, x], var[1][2][y, x]],
                              [var[0][2][y, x], var[1][2][y, x], var[2][2][y, x]]])
            cov = np.array([c[y, x] for c in covIP])
            a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))  # eq 14

        # ECCV10 eq.15
        b = mean_p - a[:, :, 0] * means[0] - \
            a[:, :, 1] * means[1] - a[:, :, 2] * means[2]

        # ECCV10 eq.16
        q = (self.boxfilter(a[:, :, 0], r) * I[:, :, 0] + self.boxfilter(a[:, :, 1], r) *
            I[:, :, 1] + self.boxfilter(a[:, :, 2], r) * I[:, :, 2] + self.boxfilter(b, r)) / base
    
        return q
class Haze_remove(main):
    def __init__(self, image, blockSize=3, meanMode=False, percent=0.001,omega=0.95,t0=0.1,refine=True):
        super().__init__(image, blockSize, meanMode, percent)
        self.omega=omega
        self.t0=t0
        self.refine=refine
    def hazeFree(self,pic):
        #imgGray = self.image
        imgDark = self.getDarkChannel()
        atomsphericLight = self.getAtomsphericLight(pic)

        imgDark = np.float64(imgDark)
        transmission = 1 - self.omega * imgDark / atomsphericLight

        # 防止出现t小于0的情况
        # 对t限制最小值为0.1
        transmission[transmission<0.1] = 0.1 

        if self.refine:        
            normI = (pic - pic.min()) / (pic.max() - pic.min())  # normalize I
            transmission = self.guided_filter(normI, transmission, r=40, eps=1e-3)
            
        sceneRadiance = np.zeros(pic.shape)
        pic = np.float64(pic)

        for i in range(3):        
            SR = (pic[:,:,i] - atomsphericLight)/transmission + atomsphericLight

        # 限制透射率 在0～255                  
            SR[SR>255] = 255
            SR[SR<0] = 0                    
            sceneRadiance[:,:,i] = SR 
        sceneRadiance = np.uint8(sceneRadiance)

        return sceneRadiance
    