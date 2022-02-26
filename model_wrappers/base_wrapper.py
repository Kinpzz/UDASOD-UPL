from typing import Dict
import torch
from apex import amp

# define interface of trainer
class Base_Wrappepr():
    name = 'LDF'
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.scheduler = None
        self.model = None
        self.optimizer = None
    
    def setup_optimizer(self) -> None:
        raise NotImplementedError
    
    def setup_scheduler(self) -> None:
        raise NotImplementedError

    def handle_batch(self, batch, global_step = None) -> Dict:
        raise NotImplementedError
    
    def setup_tbwriter(self, tb_writer):
        self.tb_writer = tb_writer
        

    def setup_amp(self):
        ## model must be in gpu
        assert self.cfg.SOLVER.AMP
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O2')
        return self.model
        
    def init_model(self):
        raise NotImplementedError

    def pred_img(self, image:torch.Tensor, shape = None):
        raise NotImplementedError


    def output2img(self, pred:torch.Tensor):
        raise NotImplementedError


    def postprocess_np(self, a_output, shape):
        raise NotImplementedError
