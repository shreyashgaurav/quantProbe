import json #To write the manifest.json
import os #For path building
import struct
import numpy as np #Reshaping tesors and writing raw bytes
import torch #DL framework for model
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights #models and pretrained weights enum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #abs pathto the project root
MODELS_DIR = os.path.join(ROOT, "models") # (weights + manifest)
DATA_DIR = os.path.join(ROOT, "data") #for reference output
WEIGHTS_DIR = os.path.join(MODELS_DIR, "weights") #contsins per layer BIN files
os.makedirs(WEIGHTS_DIR, exist_ok=True) #Creates if missing
os.makedirs(DATA_DIR, exist_ok=True) #Creates if missing


# Serializes a PyTorch tensor into a raw binary file of 32-bit floating-point values
# and returns metadata (shape and size) about the tensor
def dump_tensor(t: torch.tensor, path: str):
    arr = t.detach().cpu().contiguous().numpy().astype("<f4") #detach from autograd, move to CPU, make memory row-major, convert to numpy, cast to LE float32('<f4')
    arr.tofile(path) #Write the raw bytes straight to the disk
    return list(arr.shape), int(arr.size) #For sanity checjk - Not needed though

# This function mathematically combines a Conv2D layer and a BatchNorm2D layer into a single Conv2D layer with modified weights and biases
def fold_conv_bn(conv: nn.Conv2d, bn:nn.BatchNorm2d):
    w = conv.weight.clone() #copy the conv weight tensor to avoid mutation of the model
    if conv.bias is not None:
        b = conv.bias.clone() #if bias exists then copy it in b
    else:
        b = torch.zeros(conv.out_channels) #else start from a zero bias

    # Reading Batch Norm parameters
    gamma = bn.weight #batchnorm's learned scale
    beta = bn.bias #batchnorm's learned shift
    mean = bn.running_mean #running mean collected during training per channel
    var = bn.running_var #Running variance collected during training
    eps = bn.eps #small const to avoid div by zeros


    scale = gamma / torch.sqrt(var + eps) #the per outut channnel folding factor
    w_folded = w * scale.reshape(-1, 1, 1, 1)
    b_folded = (b - mean) * scale + beta
    return w_folded, b_folded

def main():
    print("Loading the pretrained MobileNetV2 ...")
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 #ImageNet-pretrained weitghs
    model = mobilenet_v2(weights=weights) #Building the model and loading the weights
    model.eval() #Switching to inference mode

    # the manifest skeleton
    manifest = {
        "model": "mobilenet_v2",
        "input_shape": [1, 3, 224, 224], #Expected input
        "num_classes": 1000, #ImageNet has 1000 classes
        "layers": [],
        "notes": "Conv+BN folded, relu6 where marked. LE float32, row-major.",
    }

    # layer_idx = 0

    flat = []

    for _, m in model.features.named_modules(): #iterating every submodules of the feature extractor
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU6, nn.ReLU)):
            flat.append(m)
    i = 0
    conv_count = 0

    while i < len(flat):
        m = flat[i]
        if isinstance(m, nn.Conv2d):
            conv = m
            bn = None
            act = None

            if i + 1 < len(flat) and isinstance(flat[i+1], nn.BatchNorm2d):
                bn = flat[i + 1]

                if i + 2 < len(flat) and isinstance(flat[i+2], (nn.ReLU6, nn.ReLU)):
                    act = flat[i + 2]
                    i += 3
                else:
                    i += 2
            else:
                i += 1

            if bn is not None:
                w_folded, b_folded = fold_conv_bn(conv, bn)
            else:
                w_folded = conv.weight.clone()
                b_folded = conv.bias.clone() if conv.bias is not None \
                    else torch.zeros(conv.out_channels)
            name = f"conv{conv_count}"
            wshape, _ = dump_tensor(
                w_folded, os.path.join(WEIGHTS_DIR, f"{conv_count:03d}_{name}_w.bin"))
            bshape, _ = dump_tensor(
                b_folded, os.path.join(WEIGHTS_DIR, f"{conv_count:03d}_{name}_b.bin"))
            stride = list(conv.stride)
            padding = list(conv.padding)
            groups = conv.groups
            manifest["layers"].append({
                "idx": conv_count,
                "name": name,
                "type": "conv2d",
                "weight_file": f"weights/{conv_count:03d}_{name}_w.bin",
                "bias_file": f"weights/{conv_count:03d}_{name}_b.bin",
                "weight_shape": wshape,
                "bias_shape": bshape,
                "stride": stride,
                "padding": padding,
                "groups": groups,
                "activation": "relu6" if act is not None else "none",
            })
            conv_count += 1
        else:
            i += 1

    classifier_linear = None

    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            classifier_linear = m
    if classifier_linear is not None:
        wshape, _ = dump_tensor(
            classifier_linear.weight,
            os.path.join(WEIGHTS_DIR, "classifier_w.bin"))
        bshape, _ = dump_tensor(
            classifier_linear.bias,
            os.path.join(WEIGHTS_DIR, "classifier_b.bin"))
        manifest["layers"].append({
            "idx": conv_count,
            "name": "classifier",
            "type": "linear",
            "weight_file": "weights/classifier_w.bin",
            "bias_file": "weights/classifier_b.bin",
            "weight_shape": wshape,   # [out_features, in_features]
            "bias_shape": bshape,
            "activation": "none",
        })

    manifest["num_conv_layers"] = conv_count

    torch.manual_seed(0)
    ref_input = torch.randn(1, 3, 224, 224)
    dump_tensor(ref_input, os.path.join(DATA_DIR, "ref_input.bin"))

    with torch.no_grad():
        ref_output = model(ref_input)
    dump_tensor(ref_output, os.path.join(DATA_DIR, "ref_output.bin"))

    # Capture a few early intermediate activations via forward hooks, so the
    # C++ forward pass can be validated layer-by-layer, not just end-to-end.
    activations = {}

    def make_hook(tag):
        def hook(_module, _inp, out):
            activations[tag] = out.detach().clone()
        return hook

    handles = []
    # first conv block output, and first inverted-residual output
    handles.append(model.features[0].register_forward_hook(make_hook("feat0")))
    handles.append(model.features[1].register_forward_hook(make_hook("feat1")))
    handles.append(model.features[2].register_forward_hook(make_hook("feat2")))

    with torch.no_grad():
        _ = model(ref_input)
    for h in handles:
        h.remove()

    ref_meta = {"activations": {}}
    for tag, t in activations.items():
        shape, _ = dump_tensor(t, os.path.join(DATA_DIR, f"ref_{tag}.bin"))
        ref_meta["activations"][tag] = {"file": f"ref_{tag}.bin", "shape": shape}
    with open(os.path.join(DATA_DIR, "ref_meta.json"), "w") as f:
        json.dump(ref_meta, f, indent=2)

    with open(os.path.join(MODELS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported {conv_count} conv layers + classifier.")
    print(f"Manifest: {os.path.join(MODELS_DIR, 'manifest.json')}")
    print(f"Reference input/output + {len(activations)} activations in {DATA_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()