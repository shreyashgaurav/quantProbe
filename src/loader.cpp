//
// Created by shrey on 20-Jul-26.
//
#include "loader.hpp"
#include "third_party/json.hpp"

#include <fstream>
#include <stdexcept>
#include <iostream>

using nlohmann::json;

Tensor load_bin(const std::string& path, const std::vector<int>& shape) {
    Tensor t(shape);  // allocates data sized to product(shape)

    std::ifstream f(path, std::ios::binary | std::ios::ate);  // open at end to get size
    if (!f) throw std::runtime_error("Cannot open bin file: " + path);

    std::streamsize bytes = f.tellg();          // file size in bytes
    f.seekg(0, std::ios::beg);                   // rewind to start

    std::streamsize expected = static_cast<std::streamsize>(t.numel()) * 4; // float32 = 4 bytes
    if (bytes != expected) {
        throw std::runtime_error(
            "Size mismatch for " + path + ": file has " + std::to_string(bytes) +
            " bytes, expected " + std::to_string(expected) +
            " (shape " + t.shape_str() + ")");
    }

    f.read(reinterpret_cast<char*>(t.data.data()), bytes);  // read raw bytes straight into the buffer
    if (!f) throw std::runtime_error("Failed reading " + path);
    return t;
}

static std::vector<int> to_int_vec(const json& j) {
    std::vector<int> v;
    for (const auto& e : j) v.push_back(e.get<int>());
    return v;
}

Model load_model(const std::string& models_dir) {
    std::string manifest_path = models_dir + "/manifest.json";
    std::ifstream mf(manifest_path);
    if (!mf) throw std::runtime_error("Cannot open manifest: " + manifest_path);

    json m;
    mf >> m;  // parse the whole manifest

    Model model;
    model.name = m.value("model", "");
    model.num_classes = m.value("num_classes", 0);
    if (m.contains("input_shape")) model.input_shape = to_int_vec(m["input_shape"]);

    for (const auto& L : m["layers"]) {
        LayerInfo info;
        info.idx = L.value("idx", 0);
        info.name = L.value("name", "");
        info.type = L.value("type", "");
        info.activation = L.value("activation", "none");
        if (L.contains("stride"))  info.stride  = to_int_vec(L["stride"]);
        if (L.contains("padding")) info.padding = to_int_vec(L["padding"]);
        info.groups = L.value("groups", 1);

        std::vector<int> wshape = to_int_vec(L["weight_shape"]);
        std::vector<int> bshape = to_int_vec(L["bias_shape"]);

        info.weight = load_bin(models_dir + "/" + L["weight_file"].get<std::string>(), wshape);
        info.bias   = load_bin(models_dir + "/" + L["bias_file"].get<std::string>(),   bshape);

        model.layers.push_back(std::move(info));
    }

    return model;
}