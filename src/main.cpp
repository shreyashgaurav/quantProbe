//
// Created by shrey on 21-Jul-26.
//

#include "loader.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string models_dir = "models";           // default; assumes you run from project root
    if (argc > 1) models_dir = argv[1];           // or pass a path explicitly

    try {
        Model model = load_model(models_dir);

        std::cout << "Loaded model: " << model.name
                  << " | classes: " << model.num_classes
                  << " | layers: " << model.layers.size() << "\n\n";

        // Print the first few layers so we can eyeball shapes vs the manifest.
        int shown = 0;
        for (const auto& L : model.layers) {
            std::cout << "[" << L.idx << "] " << L.name
                      << "  type=" << L.type
                      << "  w=" << L.weight.shape_str()
                      << "  b=" << L.bias.shape_str();
            if (L.type == "conv2d") {
                std::cout << "  groups=" << L.groups
                          << "  act=" << L.activation;
            }
            std::cout << "\n";
            if (++shown >= 6) { std::cout << "  ...\n"; break; }
        }

        // Print the classifier (last layer) explicitly.
        const auto& last = model.layers.back();
        std::cout << "\nlast layer: " << last.name
                  << "  w=" << last.weight.shape_str()
                  << "  b=" << last.bias.shape_str() << "\n";

        // A tiny numeric spot-check: first 3 weights of conv0.
        const auto& c0 = model.layers.front();
        std::cout << "\nconv0 first 3 weights: "
                  << c0.weight.data[0] << ", "
                  << c0.weight.data[1] << ", "
                  << c0.weight.data[2] << "\n";

        std::cout << "\nOK: model loaded successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}