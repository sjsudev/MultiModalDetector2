from .deep_sort import DeepSort
from .mydeep_sort import MyDeepSort

__all__ = ['MyDeepSort', 'DeepSort']
#__all__ = ['MyDeepSort', 'DeepSort', 'build_tracker']

# def build_tracker(cfg, use_cuda):
#     return DeepSort(cfg.DEEPSORT.REID_CKPT, 
#                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
#                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
#                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
    








