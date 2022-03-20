import numpy as np
import torchvision
import torch
import MyNet
import Retrieval
import cv2
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import cm

net_parameters_path="net_parameters/checkpoint7_resnet34.pt"
cfg_yolo_path = 'net_parameters/yolov3_custom.cfg'
weights_yolo_path = 'net_parameters/yolov3_custom_final.weights'
orb_features_path = 'features_orb_images'
images_retrieval_path = "retrieval_images/"
LabelOfDataset={
    0: "casa",
    1: "chiesa",
    2: "condominio",
    3: "edifici_storici_monumenti",
    4: "negozi"

}

def GeometricTransform(query):
    #Funzione che esegue la query expansion

    srcTri = np.array([[0, 0], [query.shape[1] - 1, 0], [0, query.shape[0] - 1]]).astype(np.float32)
    dstTri = np.array([[0, query.shape[1] * 0.33], [query.shape[1] * 0.85, query.shape[0] * 0.25],
                       [query.shape[1] * 0.15, query.shape[0] * 0.7]]).astype(np.float32)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    warp_dst = cv2.warpAffine(query, warp_mat, (query.shape[1], query.shape[0]))
    # Rotating the image after Warp
    center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)
    angle = -50
    scale = 0.6
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

    #Trasformazione prospettica
    query = cv2.resize(query, (400, 400), interpolation=cv2.INTER_NEAREST)

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    pers_tran = cv2.warpPerspective(query, M, (300, 300))


    cv2.imwrite("images_from_query_exp/trans1.jpg", warp_dst)
    cv2.imwrite("images_from_query_exp/trans2.jpg", warp_rotate_dst)
    cv2.imwrite("images_from_query_exp/trans3.jpg",pers_tran )

    #return warp_dst, warp_rotate_dst

def findObjects(outputs, img, classes):
    confTh = 0.1
    nms_threshold = 0.3

    hT, wT, cT = img.shape

    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence =scores[classId]

            if confidence > confTh:
                w,h = int(det[2]*wT), int(det[3]*hT) #Pixels values
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT)-h/2) #center point

                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))
    indeces = cv2.dnn.NMSBoxes(bbox, confs, confTh, nms_threshold=nms_threshold)
    for i in indeces:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #(b,g,r) = (0, 255, 0)
        cv2.putText(img, f'{classes[classIds[i]].upper()} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    #Retrieval.batch_extr(images_retrieval_path)

    #ResNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #net = MyNet.resnet18(pretrained=True).to(device)
    net = MyNet.resnet34(pretrained=True).to(device)
    net.load_state_dict(torch.load(net_parameters_path, map_location=device))

    test_transform = transforms.Compose(transforms=[
        transforms.Resize([400, 200]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    #Yolo
    yolo_net = cv2.dnn.readNetFromDarknet(cfg_yolo_path, weights_yolo_path)
    yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    #Retrieval with ORB
    ma = Retrieval.ImagesMatcher()

    #Application-Run
    cap = cv2.VideoCapture(2)
    classes = ["door"]

    while True:
        success, img = cap.read()
        cv2.imwrite("frame.jpg",img)
        frame = Image.open("./frame.jpg")
        #frame = Image.fromarray(img)


        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
        resnet_input = test_transform(frame)

        #Yolo_run
        yolo_net.setInput(blob)
        layer_names = yolo_net.getLayerNames()
        outputNames = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
        outputs = yolo_net.forward(outputNames)
        findObjects(outputs, img, classes)

        #Resnet_run
        x = resnet_input.to(device)
        out, features_resnet = net(x.unsqueeze(0))
        cv2.putText(img, f'Resnet: {LabelOfDataset[int(torch.argmax(out.detach(), dim=1))]}', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



        #Retrieval with Orb

        frame = cv2.imread("./frame.jpg")
        GeometricTransform(frame)
        matches_ORB, matches_NET = ma.match("./frame.jpg", ntop=1)
        orb_label = matches_ORB[0]["name"]
        net_label = matches_NET[0]["name"]

        for i in range(len(orb_label)):

            if orb_label[i] == ' ':
                orb_label = orb_label[:i]
                break

        for i in range(len(net_label)):

            if net_label[i] == ' ':
                net_label = net_label[:i]
                break



        '''
        min_rmse = 0
        min_match_name = None

        min_rmse_net = 0
        min_match_name_net = None
        # Risultato della query originale
        for match in matches_ORB:

            min_rmse = match["rmse"]
            min_match_name = match["name"]
        for match in matches_NET:
            min_rmse_net = match["rmseNet"]
            min_match_name_net = match["name"]

        # Risultato della query con Warp transform
        matches_ORB, matches_NET = ma.match("images_from_query_exp/trans1.jpg", ntop=1)
        for match in matches_ORB:

            if match["rmse"] < min_rmse:
                min_rmse = match["rmse"]
                min_match_name = match["name"]
        for match in matches_NET:

            if match["rmseNet"] < min_rmse_net:
                min_rmse_net = match["rmseNet"]
                min_match_name_net = match["name"]

        # Risultato della query con Warp e Rotate
        matches_ORB, matches_NET = ma.match("images_from_query_exp/trans2.jpg", ntop=1)
        for match in matches_ORB:

            if match["rmse"] < min_rmse:
                min_rmse = match["rmse"]
                min_match_name = match["name"]

        for match in matches_NET:

            if match["rmseNet"] < min_rmse_net:
                min_rmse_net = match["rmseNet"]
                min_match_name_net = match["name"]

        # Risultato della query con leggero cambiamento di prospettiva
        matches_ORB, matches_NET = ma.match("images_from_query_exp/trans3.jpg", ntop=1)
        for match in matches_ORB:

            if match["rmse"] < min_rmse:
                min_rmse = match["rmse"]
                min_match_name = match["name"]
        for match in matches_NET:

            if match["rmseNet"] < min_rmse_net:
                min_rmse_net = match["rmseNet"]
                min_match_name_net = match["name"]


        for i in range(len(min_match_name)):

            if min_match_name[i] == ' ':
                orb_label = min_match_name[:i]

        for i in range(len(min_match_name_net)):

            if min_match_name_net[i] == ' ':
                net_label = min_match_name_net[:i]
        '''
        cv2.putText(img, f'ORB_Retrieval: {orb_label}', (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f'NET_Retrieval: {net_label}', (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Image', img)
        cv2.waitKey(1)






    print("ok")

if __name__ == '__main__':
    main()