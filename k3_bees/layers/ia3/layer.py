from keras import layers, ops

from k3_bees.api_export import k3_export


@k3_export('k3_bees.layers.IA3')
class IA3(layers.Layer):
    def __init__(
        self, 
        in_layer, 
        multiplier=1.0, 
        pre=False, 
        **kwargs
    ):
        super().__init__()
        
        self.shape = ops.shape(in_layer.kernel)
        if in_layer.__class__.__name__ == 'Conv2D':
            in_dim = self.shape[2]
            out_dim = in_layer.filters
            if pre:
                train_dim = in_dim
            else:
                train_dim = out_dim
            self.weight = self.add_weight((1, 1, 1, train_dim), initializer='zeros')
        else:
            in_dim = self.shape[0]
            out_dim = in_layer.units
            if pre:
                train_dim = in_dim
            else:
                train_dim = out_dim
            
            self.weight = self.add_weight((train_dim,), initializer='zeros')

        self.multiplier = multiplier
        self.in_call = None
        self.in_layers = [in_layer]
        self.pre = pre

    def apply_to(self):
        self.in_call = self.in_layers[0].call
        self.in_layers[0].call = self.call

    def call(self, x):
        if self.pre:
            x = x * (1 + self.weight * self.multiplier)
        out = self.in_call(x)
        dtype = out.dtype
        if not self.pre:
            out = out * (1 + self.weight * self.multiplier)
            out = ops.cast(out, dtype)
        return out