import cv2
import numpy as np
import _pickle as cPickle
import math
import os
import MyNet
import  torch
import torchvision
from PIL import Image
from scipy.spatial import distance

net_parameters_path="net_parameters/checkpoint1_resnet18.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_transform = torchvision.transforms.Compose(transforms=[
        torchvision.transforms.Resize([400, 200]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


def extractFeatures(path_img, vector_size=32):
    #Image-processing
    img = cv2.imread(path_img)
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=300, sigmaSpace=300)

    net = MyNet.resnet18(pretrained=True).to(device)
    net.load_state_dict(torch.load(net_parameters_path, map_location=device))
    x = Image.open(path_img)
    x = test_transform(x)
    x = x.to(device)

    try:
        #ORB
        orb = cv2.ORB_create()
        kpsIMG = orb.detect(img)


        kpsIMG = sorted(kpsIMG, key=lambda x: -x.response)[:vector_size]

        kpsIMG, dscIMG = orb.compute(img, kpsIMG)

        dscIMG = dscIMG.flatten()

        needed_size = (vector_size * 64)
        if dscIMG.size < needed_size:
            dscIMG = np.concatenate([dscIMG, np.zeros(needed_size - dscIMG.size)])

        #RESNET
        out, features = net(x.unsqueeze(0))


    except cv2.error as e:
        print(e)
        return None
    return dscIMG, features

def batch_extr(images_path, pick_db_path="features_orb_retrieval.pck", features_path ="features_net_retrieval.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    result_net = {}
    for f in files:
        try:
            cv2.imread(f)
            print("Features extraction from images", f)
            name = f.split('/')[-1].lower()
            result[name], result_net[name] = extractFeatures(f)
        except Exception as e:
            print("Non sono riuscito ad elaborare l'immagine ", f)
    #Il try and catch dentro il for mi serve per evitare errori bloccanti, semplicemente se non riesco a estrarre features da quell'immagine la ignoro


    try:
        fp = open(pick_db_path,"wb")
        cPickle.dump(result, fp)
        fnet = open(features_path, "wb")
        cPickle.dump(result_net, fnet)

    except Exception as e:
        print(e)
    #return result
    #with open(pick_db_path, 'wb') as fp:
     #   cPickle.dump(result, fp)


class ImagesMatcher(object):
    def __init__(self, images_path="bck/custom_data/", pick_db_path="features_orb_retrieval.pck", features_net_path ="features_net_retrieval.pck"):
        #with open(pick_db_path, encoding='utf8') as fp:
        fp = open(pick_db_path, "rb")
        fnet =open(features_net_path, "rb")
        self.data = cPickle.load(fp)
        self.data_from_net = cPickle.load(fnet)
        self.names = []
        self.namesNet = []
        self.featuresMatrix = [] #Features estratte da ORB
        self.featuresMatrixFromNet = [] #Features estratte dalla rete Resnet

        for k in self.data:#self.data.iteritems()
            self.names.append(k)
            self.featuresMatrix.append(self.data[k])
        for k in self.data_from_net:
            self.namesNet.append(k)
            self.featuresMatrixFromNet.append(self.data_from_net[k])

        #self.featuresMatrixFromNet = np.array(self.featuresMatrixFromNet)
        #for obj in self.featuresMatrixFromNet:
        self.featuresMatrix = np.array(self.featuresMatrix)
        self.names = np.array(self.names)

    def Rmse(self, query_desc, image_desc):
        diff = np.subtract(query_desc, image_desc)
        square = np.square(diff)
        mse = square.mean()
        return math.sqrt(mse)
    def RmseFromNet(self, query_features, image_features):
        diff = torch.sub(query_features, image_features)
        square = torch.square(diff)
        mse = torch.mean(square)
        return torch.sqrt(mse)

    def match(self, image_path, ntop=3):
        net_features = torch.zeros((4, 512), device=device)
        orb_features = np.zeros((4,2048))
        #avg_net_features.to(device)
        orb_features[0], net_features[0] = extractFeatures(image_path)
        orb_features[1], net_features[1] = extractFeatures("images_from_query_exp/trans1.jpg")
        orb_features[2], net_features[2] = extractFeatures("images_from_query_exp/trans2.jpg")
        orb_features[3], net_features[3] = extractFeatures("images_from_query_exp/trans3.jpg")

        avg_net_features = (torch.sum(net_features, dim=0).to(device))/net_features.shape[0]
        avg_orb_features = (np.sum(orb_features, axis=0))/orb_features.shape[0]

        result = []
        result_fromNet = []
        for i in range(len(self.featuresMatrix)):
            result.append({'name': self.names[i], 'rmse':self.Rmse(avg_orb_features, self.featuresMatrix[i])})
        for i in range(len(self.featuresMatrixFromNet)):
            result_fromNet.append({'name': self.names[i], 'rmseNet': self.RmseFromNet(avg_net_features, self.featuresMatrixFromNet[i])})


        result = sorted(result, key=lambda x: x['rmse'])
        result_netFeatures = sorted(result_fromNet, key=lambda x: x['rmseNet'])
        values_to_return = []
        values_to_returnNet = []
        for i in range(ntop):
            values_to_return.append(result[i])
            values_to_returnNet.append(result_netFeatures[i])
        return values_to_return, values_to_returnNet







