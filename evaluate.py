from utils.calibration import Calibrator
import argparse
import os
from utils.model import Spice_model
import torch
import sys
import pandas as pd
import mir_eval
from tqdm import tqdm
import numpy as np
import time
import gc

class Evaluation():
    def __init__(self, model_name, test_data, cal_sample):
        self.test_data_path = os.path.join("./data_test/",test_data)
        self.model_path = os.path.join("./rev_1k_checkpoints/",model_name)
        if os.path.isfile(self.test_data_path):
            print("******* Loading test Data *******\n")
            self.test_data = pd.read_pickle(self.test_data_path).to_numpy()
            print("******* Test Data Loaded *******\n")
        else:
            sys.exit("Fatal : Test data not exist")
        
        if os.path.isfile(self.model_path):
            try:
                print("******* Loading the model *******\n")
                self.model = Spice_model([1, 64, 128, 256, 512, 512, 512], 
                                        [512, 512, 512, 256, 128, 64, 1], 
                                        [True, True, True, True, True, True])
                checkpoint = torch.load(self.model_path, 'cuda' if 
                                        torch.cuda.is_available() else 'cpu')
                self.model.load_state_dict(checkpoint['state_dict'])
                print("******* Model load successful *******\n")
            except Exception as e:
                print("******* Could not load model *******\n{}".format(e))
        else:
            sys.exit("Fatal : Model does not exist")
        cal = Calibrator(self.model, cal_sample)
        self.PT_OFFSET, self.PT_SLOPE = cal.get_values()
        
    def _output2hz(self,pitch_output):
          FMIN = 10.0    
          BINS_PER_OCTAVE = 12.0  
          cqt_bin = pitch_output * self.PT_SLOPE + self.PT_OFFSET;
          return FMIN*2**(cqt_bin/BINS_PER_OCTAVE)
    
    def evaluate(self):
        print("******* Inference Started *******\n")
        # data_part = self.test_data[:100,]
#         label = []
#         yt_hat = []
#         voice = []
#         for row in tqdm(self.test_data):
#             pitch_h1,conf_h1,x_hat1 = self.model(torch.from_numpy(row[1:129].reshape(1,128)).float())
#             yt_hat.append(pitch_h1.detach().numpy())
#             voice.append(row[-2])
#             label.append(row[-1])
            
#         y_hat = np.apply_along_axis(self._output2hz,0,yt_hat)
#         y_hat_cent = np.apply_along_axis(mir_eval.melody.hz2cents, 0, y_hat)
#         label_cent = np.apply_along_axis(mir_eval.melody.hz2cents, 0, label)   
            
        
        pitch_h1,conf_h1,x_hat1 = self.model(torch.from_numpy(self.test_data[:,1:129].reshape(len(self.test_data),128)).float())

        y_hat = np.apply_along_axis(self._output2hz,0,pitch_h1.detach().numpy())
        y_hat_cent = np.apply_along_axis(mir_eval.melody.hz2cents, 0, y_hat)
        label_cent = np.apply_along_axis(mir_eval.melody.hz2cents, 0, self.test_data[:,-1:])
        voice = self.test_data[:,-2:-1]
        
        
        voice = np.array(voice).reshape(len(voice),1)
        label_cent = np.array(label_cent).reshape(len(label_cent),1)
        y_hat_cent= np.array(y_hat_cent).reshape(len(y_hat_cent),1)
        rpa = mir_eval.melody.raw_pitch_accuracy(voice, label_cent, voice, y_hat_cent, cent_tolerance=50)
        return rpa
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SPICE')
    parser.add_argument('--model', '--model', default='checkpoint_MIR_noconf.ckp', help='No of Calibration Samples')
    parser.add_argument('--M', '--cal_samples',type=int, default=5, help='No of Calibration Samples')
    parser.add_argument('--test', '--test_dataset', default='MIR1k.pkl', help='Dataset to Load for Tesing')

    args = parser.parse_args()
    
    print(args)
    
    eva = Evaluation(args.model,  args.test, args.M) 
    print(gc.collect())
    RPA = eva.evaluate()
    print("RPA: {}\nModel :{}\nTest Data:{}\n".format(RPA, args.model,  args.test))



