//Used openvino toolkit
#include <iostream>
#include <vector>
#include <inference_engine.hpp>
#include <ie_blob.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <sndfile.h>  // Include the libsndfile header

using namespace std;
using namespace InferenceEngine;

int main() {
    try {
        // Create Inference Engine core
        Core ie;

        // Read ONNX model
        CNNNetwork network = ie.ReadNetwork("D:/ZOHO INTERN/pytorch/audio_to_magnitude.onnx");

        // Load model to the device
        ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "CPU");

        // Create inference request
        InferRequest inferRequest = executableNetwork.CreateInferRequest();

        // Read audio file into an array using libsndfile
        const char* audioFile = "D:/ZOHO INTERN/pytorch/arun.wav";
        SF_INFO sfInfo;
        SNDFILE* sndfile = sf_open(audioFile, SFM_READ, &sfInfo);

        if (!sndfile) {
            cerr << "Error: Failed to open audio file" << endl;
            return 1;
        }

        // Get the number of frames in the audio file
        int numFrames = static_cast<int>(sfInfo.frames);

        // Read audio data into a vector of floats
        vector<float> audioData(numFrames);
        sf_readf_float(sndfile, audioData.data(), numFrames);

        // Close the audio file
        sf_close(sndfile);

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
