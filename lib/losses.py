import torch

def iou_loss_ms(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

## loss conbined with variance map
def get_ldf_loss_msr(outb1, outd1, out1, outb2, outd2, out2, mask, body, detail, var, cfg):
    bce_loss_layer = torch.nn.BCEWithLogitsLoss(reduction='none')
    lossb1 = bce_loss_layer(outb1, body)
    lossd1 = bce_loss_layer(outd1, detail)
    loss1  = bce_loss_layer(out1, mask) + iou_loss_ms(out1, mask) # b, 1, h, w

    lossb2 = bce_loss_layer(outb2, body)
    lossd2 = bce_loss_layer(outd2, detail)
    loss2  = bce_loss_layer(out2, mask) + iou_loss_ms(out2, mask)
    
    exp_var = torch.exp(-var * cfg.SOLVER.PIXEL_WEIGHT)

    loss1, loss2, lossb1, lossb2, lossd1, lossd2 = \
        [torch.mean(t_loss * exp_var) for t_loss in [loss1, loss2, lossb1, lossb2, lossd1, lossd2]]

    loss   = (lossb1 + lossd1 + loss1 + lossb2 + lossd2 + loss2)/2
    return loss, {
                'lossb1':lossb1.item(), 'lossd1':lossd1.item(), 
                'loss1':loss1.item(), 'lossb2':lossb2.item(), 
                'lossd2':lossd2.item(), 'loss2':loss2.item()
            }