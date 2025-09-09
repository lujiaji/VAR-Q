import torch
import numpy as np

class VAR_Q:
    def __init__(self, quant_bits=8, quant_method='per-tensor', blk_idx=0):
        self.quant_bits = quant_bits,
        self.quant_method = quant_method
        self.cur_blk_idx = blk_idx
        self.bound_min = -2 ** (quant_bits - 1)
        self.bound_max =  2 ** (quant_bits - 1) - 1
        self.cached_item, self.cached_scale = None, None

    def quant(self, item):
        # whole tensor quantization
        range_half = item.abs().amax()
        scale = range_half / (self.bound_max + 1)
        self.quantized_item = torch.round(item / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
        self.scale = scale
    
    def dequant(self):
        return (self.quantized_item.float()*self.scale).to(torch.float32)

    def use_var_q(self,item):
        self.quant()
        return self.dequant()