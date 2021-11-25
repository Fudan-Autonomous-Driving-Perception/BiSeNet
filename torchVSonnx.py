import cv2
import torch
import argparse
import numpy as np
import onnxruntime
from onnxruntime.datasets import get_example

import lib.transform_cv2 as T
from lib.models import model_factory

# args
parse = argparse.ArgumentParser()
parse.add_argument('--onnxmodel', dest='onnx_model_path', type=str, default='./res/model.onnx',)
parse.add_argument('--torchmodel', dest='torch_model_path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()

to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223),
    std=(0.2112, 0.2148, 0.2115),
)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 获取 input
dummy_input = cv2.resize(cv2.imread(args.img_path), (512, 512))
dummy_input = dummy_input[:, :, ::-1]
dummy_input = to_tensor(dict(im=dummy_input, lb=None))['im'].unsqueeze(0)

# onnx 模型
example_model = get_example(args.onnxmodel)
sess = onnxruntime.InferenceSession(example_model)
onnx_out = sess.run(None, {'input_image': to_numpy(dummy_input)})

# torch 模型
net = model_factory['bisenetv1'](10, aux_mode='pred')
net.load_state_dict(torch.load(args.torchmodel, map_location='cpu'), strict=False)
net.eval()
with torch.no_grad():
    torch_out = net(dummy_input)

# onnx and torch 输出结果
onnxout = onnx_out[0]
torchout = torch_out.numpy()

# 随机调色板 palette
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

pred = palette[onnxout]
cv2.imwrite('res_onnx.jpg', pred[0])
pred = palette[torchout]
cv2.imwrite('res_torch.jpg', pred[0])
