//Used openvino toolkit
#include <iostream>
#include <vector>
#include <inference_engine.hpp>
#include <ie_blob.h>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace std;
using namespace InferenceEngine;

int main() {
    try {
        // Create Inference Engine core
        Core ie;

        // Read ONNX model
        CNNNetwork network = ie.ReadNetwork("path/to/your/model.onnx");

        // Load model to the device
        ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "CPU");

        // Create inference request
        InferRequest inferRequest = executableNetwork.CreateInferRequest();

        // Read audio file into an array (replace with your own method or library)
        // For simplicity, assuming a 1D array of floats
        std::vector<float> audioData;
        // TODO: Read audio file and populate audioData

        // Set input blob
        Blob::Ptr inputBlob = inferRequest.GetBlob("input_audio");
        MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
        auto minputHolder = minput->wmap();
        float *inputData = minputHolder.as<float *>();
        std::copy(audioData.begin(), audioData.end(), inputData);

        // Perform inference
        inferRequest.Infer();

        // Get output blob
        Blob::Ptr outputBlob = inferRequest.GetBlob("output_magnitude");
        MemoryBlob::Ptr moutput = as<MemoryBlob>(outputBlob);
        auto moutputHolder = moutput->wmap();
        float *outputData = moutputHolder.as<float *>();

        // Print output magnitude
        for (size_t i = 0; i < outputBlob->size(); i++) {
            std::cout << "Magnitude[" << i << "] = " << outputData[i] << std::endl;
        }

        // Set intra-processing threads to 10
        ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "10"}}, "CPU");

    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
