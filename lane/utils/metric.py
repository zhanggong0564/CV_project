import numpy as np

def compute_iou(pred,gt,result):
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    for i in range(8):
        single_gt = gt==i
        single_pred = pred==i
        tem_tp = np.sum(single_gt*single_pred)
        tem_ta = np.sum(single_pred)+np.sum(single_gt)-tem_tp
        result['TP'][i]+=tem_tp
        result['TA'][i]+=tem_ta
    return result
