import torch, torchvision, os
os.makedirs("fp32", exist_ok=True)
jobs=[("lraspp_mobilenet_v3_large",torchvision.models.segmentation.lraspp_mobilenet_v3_large,512),
      ("deeplabv3_mobilenet_v3_large",torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,512)]
for name,fn,size in jobs:
    m=fn(weights="DEFAULT").eval()
    x=torch.randn(1,3,size,size)
    out=f"fp32/{name}_{size}.onnx"
    with torch.no_grad():
        torch.onnx.export(m,x,out,input_names=["input"],output_names=["out"],
                          opset_version=13,dynamo=False,
                          dynamic_axes=None)
    print("exported",out,os.path.getsize(out))
