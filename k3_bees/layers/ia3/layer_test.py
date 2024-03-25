
import pytest
import keras

from keras import layers, ops
from k3_bees.layers.ia3.layer import IA3

@pytest.mark.parametrize(
    "in_layer, multiplier, pre, in_features, out_features",
    [
        ('dense', 1.0, False, 32, 56),
        ('dense', 1.0, True, 32, 56),
        ('conv2d', 2.0, False, 32, 56),
        ('conv2d', 2.0, True, 32, 56),
    ]
)
def test_ia3(in_layer, multiplier, pre, in_features, out_features):
    if in_layer == 'dense':
        in_layer = layers.Dense(out_features)
        input_shape = (10, in_features)
    elif in_layer == 'conv2d':
        in_layer = layers.Conv2D(out_features, 3)
        input_shape = (3, 12, 12, in_features)
    inputs = keras.random.uniform((input_shape))
    expected_shape = ops.shape(in_layer(inputs))
    ia3 = IA3(in_layer, multiplier, pre)
    ia3.apply_to()
    output = ia3(inputs)
    assert ops.shape(output) == expected_shape