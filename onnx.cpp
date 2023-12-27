#include <iostream>
#include <vector>
#include <sndfile.h>

int main() {
    // Specify the path to the audio file
    const char* audioFilePath = "D:/ZOHO INTERN/pytorch/arun.wav";

    // Open the audio file using libsndfile
    SF_INFO sfInfo;
    SNDFILE* sndFile = sf_open(audioFilePath, SFM_READ, &sfInfo);
    
    if (!sndFile) {
        std::cerr << "Error opening audio file." << std::endl;
        return 1;
    }

    // Read audio data into a vector
    std::vector<float> audioData(sfInfo.frames * sfInfo.channels);
    sf_readf_float(sndFile, audioData.data(), sfInfo.frames);

    // Close the audio file
    sf_close(sndFile);

    // Print some information about the audio file
    std::cout << "Audio file: " << audioFilePath << std::endl;
    std::cout << "Sample rate: " << sfInfo.samplerate << " Hz" << std::endl;
    std::cout << "Channels: " << sfInfo.channels << std::endl;
    std::cout << "Frames: " << sfInfo.frames << std::endl;

    // Now you can use the 'audioData' vector for further processing

    return 0;
}
