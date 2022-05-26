import torch
import models
import torch.onnx

#Function to Convert to ONNX
def Convert_ONNX(model,dummy_input):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
#    dummy_input = torch.randn(1, input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "mobilevit.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=12,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')
img = torch.randn(1, 3, 256, 256,requires_grad=True)
net = models.MobileViT_S()

# XXS: 1.3M 、 XS: 2.3M 、 S: 5.6M
print("MobileViT-S params: ", sum(p.numel() for p in net.parameters()))
print(f"Output shape: {net(img).shape}")
Convert_ONNX(net,img)
