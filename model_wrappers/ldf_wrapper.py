import torch
import torch.nn.functional as F
from models.ldf import LDF
from models.ldf_vgg import LDF_VGG
from lib.losses import get_ldf_loss_msr
from model_wrappers.base_wrapper import Base_Wrappepr
import logging
from lib.misc import get_scheduler
from apex import amp


class LDF_Wrapper(Base_Wrappepr):
    name = 'LDF'
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        if cfg.MODEL.NAME == 'LDF_VGG':
            self.model = LDF_VGG(cfg)
        elif cfg.MODEL.NAME == 'LDF':
            self.model = LDF(cfg)
        self.tb_writer = None
        self.optimizer = None
        self.scheduler = None


    def setup_scheduler(self):
        assert self.optimizer != None
        cfg = self.cfg
        cfg.epoch = cfg.SOLVER.WARMUP_EPOCH + cfg.SOLVER.EPOCH_PER_ROUND * (cfg.SOLVER.ROUND_NUM - 1)
        cfg.tot_steps = cfg.epoch * cfg.SOLVER.ITER_PER_EPOCH
        self.scheduler = get_scheduler(self.optimizer, cfg)

    def setup_tbwriter(self, tb_writer):
        self.tb_writer = tb_writer


    def setup_optimizer(self):
        base, head = [], []
        for name, param in self.model.named_parameters():
            if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
                logging.info(name)
            elif 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        self.optimizer = torch.optim.SGD( # update base and backbone differently
            [
                {'params':base, 'lr': self.cfg.SOLVER.BASE_LR},
                {'params':head, 'lr': self.cfg.SOLVER.LR}
            ],
            momentum=self.cfg.SOLVER.MOMEN, 
            weight_decay=self.cfg.SOLVER.DECAY, 
            nesterov=True
        )

    def setup_amp(self):
        ## model must be in gpu
        assert self.cfg.SOLVER.AMP
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O2')
        return self.model
        
    def init_model(self):
        self.model.init_base(self.cfg.MODEL.BAKCBONE_PATH) # init model
        self.model.init_head()


    def handle_batch(self, batch, global_step = None):
        assert self.optimizer != None
        assert self.scheduler != None
        if self.tb_writer:
            self.tb_writer.add_scalar('train/lr_base', self.optimizer.param_groups[0]['lr'], global_step=global_step)
            self.tb_writer.add_scalar('train/lr_head', self.optimizer.param_groups[1]['lr'], global_step=global_step)
        ## calulate 
        image, mask, body, detail, var = batch
        outb1, outd1, out1, outb2, outd2, out2 = self.model(image)
        loss, sub_losses = get_ldf_loss_msr(
            outb1, outd1, out1, 
            outb2, outd2, out2, 
            mask, body, detail, 
            var, self.cfg
        )

        self.optimizer.zero_grad()
        if self.cfg.SOLVER.AMP:
            with amp.scale_loss(loss, self.optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        
        # for sl in out_dict['sub_loss'].keys():
        if self.tb_writer:
            for sl in sub_losses.keys():
                self.tb_writer.add_scalar(f'train_sub_loss/{sl}', sub_losses[sl], global_step=global_step)
            self.tb_writer.add_scalar('train/overall_loss', loss, global_step=global_step)

        return {
            'all_out': [outb1, outd1, out1, outb2, outd2, out2],
            'final_out': out2,
            'loss': loss, # for back propogation
            'sub_loss': sub_losses,
        }

    def pred_img(self, image:torch.Tensor, shape = None):
        """
        image: bn, c, h, w 
        """
        _, _, _, outb2, outd2, out2 = self.model(image, shape)

        return {
            'mask': out2,
            'detail': outd2,
            'body': outb2,
        }

    def output2img(self, pred:torch.Tensor):
        """
        args:
        - pred[bn, c, h, w]
        return:
        - image[bn, c, h, w]: range in (0, 1)
        """
        # pred = F.interpolate(pred.unsqueeze(0), size=shape, mode='bilinear')
        # pred = torch.sigmoid(pred[0][0]).cpu().numpy() * 255 # H, W
        return pred.sigmoid()

    def postprocess_np(self, input, shape):
        """ output postprocessing for `one` image
        args:
        input_tensor{torch.Tensor}[c, h, w]: model output
        shape{List[2]}: output shape(h, w)
        return:
        output 
        """
        # if shape != None:
        output = F.interpolate(input.unsqueeze(0), size=shape, mode='bilinear')
        output = torch.sigmoid(output[0][0]).cpu().numpy() * 255 # H, W
        return output