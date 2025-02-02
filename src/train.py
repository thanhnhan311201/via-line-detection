#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import glob2
import os
import argparse
import sys
import cv2
import torch
import agent
import numpy as np
from data_loader import Generator
from parameters import Parameters
import test
import evaluation
import util
import copy

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training(epoch_model, loss_model, flag):
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(epoch_model, f"tensor({loss_model})")

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()
        #torch.backends.cudnn.benchmark=True

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    sampling_list = None
    loss_though_epoch = 0
    min_loss = 9999
    if flag == True:
      begin_epoch = epoch_model + 1
    else:
      begin_epoch = 0
    for epoch in range(begin_epoch, p.n_epoch):
        lane_agent.training_mode()
        for inputs, target_lanes, target_h, test_image, data_list in loader.Generate(sampling_list):
            #training
            #util.visualize_points(inputs[0], target_lanes[0], target_h[0])
            print("epoch : " + str(epoch))
            print("step : " + str(step))
            try:
              loss_p = lane_agent.train(inputs, target_lanes, target_h, epoch, lane_agent, data_list)
            except:
              continue
            torch.cuda.synchronize()
            loss_p = loss_p.cpu().data
            loss_though_epoch = loss_p
                
            if step%1000 == 0:
                lane_agent.save_model(int(step/1000), loss_p)
                testing(lane_agent, test_image, step, loss_p)
            step += 1
        if loss_though_epoch < min_loss:
            try:
                best_model = glob2.glob('savefile/best*')[0]
                best_loss = float(best_model.split('_')[1][7:-1])
                if loss_though_epoch < best_loss:
                    os.remove(best_model)
                    print(f'Best model: ({epoch}, {loss_though_epoch})')
                    lane_agent.save_model('best', loss_though_epoch)
                    min_loss = loss_though_epoch
            except:
                print(f'Best model: ({epoch}, {loss_though_epoch})')
                lane_agent.save_model('best', loss_though_epoch)
                min_loss = loss_though_epoch
        lane_agent.save_model(int(epoch), loss_though_epoch)
        sampling_list = copy.deepcopy(lane_agent.get_data_list())
        lane_agent.sample_reset()

        #evaluation:turn it off when training.
        # if epoch >= 0 and epoch%1 == 0:
        #     print("evaluation")
        #     lane_agent.evaluate_mode()
        #     th_list = [0.8]
        #     index = [3]
        #     lane_agent.save_model(int(step/100), loss_p)

            # for idx in index:
            #     print("generate result")
            #     test.evaluation(loader, lane_agent, index = idx, name="test_result_"+str(epoch)+"_"+str(idx)+".json")

        #     for idx in index:
        #         print("compute score")
        #         with open("/home/kym/Dropbox/eval_result2_"+str(idx)+"_.txt", 'a') as make_file:
        #             make_file.write( "epoch : " + str(epoch) + " loss : " + str(loss_p.cpu().data) )
        #             make_file.write(evaluation.LaneEval.bench_one_submit("test_result_"+str(epoch)+"_"+str(idx)+".json", "test_label.json"))
        #             make_file.write("\n")
        #         with open("eval_result_"+str(idx)+"_.txt", 'a') as make_file:
        #             make_file.write( "epoch : " + str(epoch) + " loss : " + str(loss_p.cpu().data) )
        #             make_file.write(evaluation.LaneEval.bench_one_submit("test_result_"+str(epoch)+"_"+str(idx)+".json", "test_label.json"))
        #             make_file.write("\n")

        if int(step)>700000:
            break

def testing(lane_agent, test_image, step, loss):
    lane_agent.evaluate_mode()

    _, _, ti = test.test(lane_agent, np.array([test_image]))

    cv2.imwrite('test_result/result_'+str(step)+'_'+str(loss)+'.png', ti[0])

    lane_agent.training_mode()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_model', type=str, default='32_tensor(1.1001)_lane_detection_network.pkl')
    parser.add_argument('-m' ,'--model_weight', type=str)
    args = vars(parser.parse_args())

    flag = False
    if args['model_weight']:
      model = args['model_weight']
      flag = True
    else:
      model = args['pretrained_model']
    epoch_model, loss_model = int(model.split('_')[0]), model.split('_')[1][7:-1]
    Training(epoch_model, loss_model, flag)
