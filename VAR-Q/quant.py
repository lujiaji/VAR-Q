import torch

class VAR_Q:
    def __init__(
            self, quant_bits=8, qkv_format='BLHc', #(B:batch_size, L:seq_len, H:heads, c:head_dim)
                 quant_method='per-tensor', #available choice: ['per-tensor', 'VAR-Q', 'per-head+per-dim', 'per-scale']
                 blk_idx=0
        ):
        self.quant_bits = quant_bits
        if qkv_format == 'BLHc':
            self.dim_cat = 1
            self.dim_map = {
                'VAR-Q': (0,1),
                'per-head+per-dim': (0, 1),
                'per-scale': (0, 2, 3)
            }
        elif qkv_format == 'BHLc':
            self.dim_cat = 2
            self.dim_map = {
                'VAR-Q': (0,2),
                'per-head+per-dim': (0, 2),
                'per-scale': (0, 1, 3)
            }
        else:
            raise ValueError(f"[VAR-Q]: Invalid format of Q, K, V: {qkv_format}")
        self.quant_method = quant_method
        self.cur_blk_idx = blk_idx
        self.qkv_format = qkv_format
        self.bound_min = -2 ** (quant_bits - 1)
        self.bound_max =  2 ** (quant_bits - 1) - 1
        self.cached_item, self.cached_scale = None, None
        self.quantized_item, self.scale = None, None

    def quant(self, item):
        # whole tensor quantization
        if self.quant_method == 'per-tensor':
            range_half = item.abs().amax()
            scale = range_half / (self.bound_max + 1)
            self.quantized_item = torch.round(item / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
            self.scale = scale

        elif self.quant_method in ('VAR-Q', 'per-scale'):
            range_half = item.abs().amax(dim=self.dim_map[self.quant_method], keepdim=True)
            scale = range_half / (self.bound_max + 1)
            self.quantized_item = torch.round(item / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
            self.scale = scale

        elif self.quant_method == 'per-head+per-dim':
            if self.quantized_item is None or self.scale is None:
                range_half = item.abs().amax(dim=self.dim_map[self.quant_method], keepdim=True)
                scale = range_half / (self.bound_max + 1)
                self.quantized_item = torch.round(item / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
                self.scale = scale
            else:
                cached_item_deq = (self.quantized_item.float()*self.scale).to(torch.float32)
                item=torch.cat((cached_item_deq, item), dim=self.dim_cat)
                range_half = item.abs().amax(dim=self.dim_map[self.quant_method], keepdim=True)
                scale = range_half / (self.bound_max + 1)
                self.quantized_item = torch.round(item / scale).clamp(self.bound_min, self.bound_max).to(torch.int8)
                self.scale = scale

        else:
            raise ValueError(f"[VAR-Q]: Invalid quantization method: {self.quant_method}")
    
    def dequant(self):
        return (self.quantized_item.float()*self.scale).to(torch.float32)

    def cache(self):
        if self.quant_method in ('VAR-Q', 'per-scale'):
            if self.quantized_item.dim() != self.scale.dim():
                self.scale = self.scale.expand(-1, self.quantized_item.shape[1], -1, -1) if self.qkv_format == 'BLHc' else self.scale.expand(-1, -1, self.quantized_item.shape[2], -1)
            if self.cached_item is None:
                self.cached_item, self.cached_scale = self.quantized_item, self.scale
            else:
                self.cached_item = torch.cat((self.cached_item, self.quantized_item), dim=self.dim_cat)
                self.cached_scale = torch.cat((self.cached_scale, self.scale), dim=self.dim_cat)

    def use_var_q(self, item):
        self.quant(item)
        self.cache()
        return self.dequant()