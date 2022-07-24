import pickle
import os
from fsl.config import config_SL, config_meta, config_meta_miniImageNet
from fsl.Framework import Learner

def meta_test():

    config = pickle.load(open('../result/visual_meta_train 0/config.pkl', 'rb'))
    config.path_params = '../result/visual_meta_train 0/model/IPLncPP-FSL, Epoch[250.000].pt'

    config.device = 0

    learner = Learner.Learner(config)
    learner.setIO('test')
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.adjust_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.test_model()

if __name__ == '__main__':
    SL_test()
