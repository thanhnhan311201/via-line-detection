import os
import cv2
import torch 
import time
import argparse
import numpy as np

from src import util
from net import Net
from src.parameters import Parameters
from src.processing_image import warp_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_weight', type=str, default='32_tensor(1.1001)_lane_detection_network.pkl')
    parser.add_argument('-o','--option', type=str, default='image', help="demo line detection on single 'image', 'folder_images' or 'video', default 'image' ")
    parser.add_argument('-d','--direction', type=str, default="./images_test/wraped_image.png", help='direction of demo video')
    args = vars(parser.parse_args())
    
    model = args['model_weight']
    epoch_model, loss_model = int(model.split('_')[0]), float(model.split('_')[1][7:-1])
    
    net = Net()
    p = Parameters()
    net.load_model(epoch_model, loss_model)

    # read image from folder images test
    if args['option'] == 'image':
        image = cv2.imread(args['direction'])
        image_resized = cv2.resize(image,(512,256))
        # cv2.imshow("image",image_resized)
        #x , y are position of points in lines 
        #because previous image is warped -> warp = False
        x , y = net.predict(image_resized, warp = True)
        print(x, y)
        image_points_result = net.get_image_points()
        # cv2.imshow("points", image_points_result)
        
        if not os.path.exists('./image_outputs/'):
          os.mkdir('./image_outputs/')
        try:
          name = args['direction'].split('/')[-1]
        except:
          name = args['direction']
        cv2.imwrite(f"./image_outputs/res_" + name, image_points_result)
        # cv2.waitKey()
    if args['option'] == 'folder_images':
        dir = args['direction']
        img_lst = os.listdir(dir)
        if not os.path.exists('./res_' + dir):
          os.mkdir('./res_' + dir)

        for img in img_lst:
          image = cv2.imread(dir + img)
          image_resized = cv2.resize(image,(512,256))
          #x , y are position of points in lines 
          #because previous image is warped -> warp = False
          x , y = net.predict(image_resized, warp = True)
          image_points_result = net.get_image_points()
          # cv2.imshow("points", image_points_result)
          
          cv2.imwrite(f"./res_" + dir + "/res_" + img, image_points_result)
          # cv2.waitKey()
    if args['option'] == 'video':
        cap = cv2.VideoCapture(args['direction'])
        if not os.path.exists('./video_outputs/'):
            os.mkdir('./video_outputs/')
        try:
          name = args['direction'].split('/')[-1]
        except:
          name = args['direction']
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./video_outputs/' + name, fourcc, 30, (512,256))
        
        while cap.isOpened():
            prevTime = time.time()
            ret, image = cap.read()
            if not net:
              break
            t_image = cv2.resize(image,(512,256))
            x , y = net.predict(t_image)
            # fits = np.array([np.polyfit(_y, _x, 1) for _x, _y in zip(x, y)])
            # fits = util.adjust_fits(fits)
            image_points = net.get_image_points()
            # mask = net.get_mask_lane(fits)
            cur_time = time.time()
            fps = 1/(cur_time - prevTime)
            s = "FPS : "+ str(fps)
            # image_lane = net.get_image_lane()
            cv2.putText(image_points, s, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            # cv2.imshow("image",image_points)
            out.write(image_points)
            key = cv2.waitKey(1)
            if not ret or key == ord('q'):
                break
        out.release()
