//
// Created by shrey on 20-Jul-26.
//

#ifndef QUANTPROBE_LOADER_HPP
#define QUANTPROBE_LOADER_HPP

#endif //QUANTPROBE_LOADER_HPP


#pragma once
#include "tensor.hpp"
#include <string>
#include <vector>

// One layer's metadata + loaded parameters, mirroring a manifest entry.
struct LayerInfo {
    int idx = 0;
    std::string name;
    std::string type;         // "conv2d" or "linear"
    std::string activation;   // "relu6" or "none"
    std::vector<int> stride;  // conv only: {sh, sw}
    std::vector<int> padding; // conv only: {ph, pw}
    int groups = 1;           // conv only

    Tensor weight;            // loaded weight tensor
    Tensor bias;              // loaded bias tensor
};

// The whole model: an ordered list of layers plus a couple of globals.
struct Model {
    std::string name;
    std::vector<int> input_shape;
    int num_classes = 0;
    std::vector<LayerInfo> layers;
};

// Read a raw little-endian float32 .bin into a Tensor of the given shape.
// Throws if the file size doesn't match product(shape)*4 bytes.
Tensor load_bin(const std::string& path, const std::vector<int>& shape);

// Parse manifest.json (at models/manifest.json) and load every layer's
// weight/bias .bin. `models_dir` is the folder containing manifest.json.
Model load_model(const std::string& models_dir);