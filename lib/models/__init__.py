

from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .ScnnStdcBisenetv1 import ScnnStdcBiSeNetV1


model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
    'ScnnStdcBiSeNetV1': ScnnStdcBiSeNetV1,
}
