// Note to Visual Studio / Windows users: you must set the working directory manually on the project file
// to $(ProjectDir)../ since these settings are not saved directly in project. The loader
// will be unable to find the example assets unless the proper working directory is set.

#if defined(_MSC_VER)
#pragma comment(lib, "dsound.lib")
#endif

#include "AudioFile.h"

#include "libnyquist/Decoders.h"
#include "libnyquist/Encoders.h"

#include <thread>

using namespace nqr;

#ifdef USE_CUDA
#include "../cuda/mdct_cuda.hpp"
#endif

extern "C" int test_opus_ifft(int nfft, float* fin, float* fout);

int main(int argc, const char** argv) try
{
#ifdef USE_CUDA
    printCudaVersion();
#endif

    const int desiredSampleRate = 48000;
    const int desiredChannelCount = 2;

    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    NyquistIO loader;

    if (argc > 1)
    {
        printf("Loading %s\n", argv[1]);
        std::string cli_arg = std::string(argv[1]);
        loader.Load(fileData.get(), cli_arg);
    }
    else
    {
        // Circular libnyquist testing
        //loader.Load(fileData.get(), "libnyquist_example_output.opus");

        // 1-channel wave
        //loader.Load(fileData.get(), "test_data/1ch/44100/8/test.wav");
        //loader.Load(fileData.get(), "test_data/1ch/44100/16/test.wav");
        //loader.Load(fileData.get(), "test_data/1ch/44100/24/test.wav");
        //loader.Load(fileData.get(), "test_data/1ch/44100/32/test.wav");
        //loader.Load(fileData.get(), "test_data/1ch/44100/64/test.wav");

        // 2-channel wave
        //loader.Load(fileData.get(), "test_data/2ch/44100/8/test.wav");
        //loader.Load(fileData.get(), "test_data/2ch/44100/16/test.wav");
        //loader.Load(fileData.get(), "test_data/2ch/44100/24/test.wav");
        //loader.Load(fileData.get(), "test_data/2ch/44100/32/test.wav");
        //loader.Load(fileData.get(), "test_data/2ch/44100/64/test.wav");

        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat_44_16_mono-ima4-reaper.wav");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat_44_16_stereo-ima4-reaper.wav");

        // Multi-channel wave
        //loader.Load(fileData.get(), "test_data/ad_hoc/6_channel_44k_16b.wav");

        // 1 + 2 channel ogg
        //loader.Load(fileData.get(), "test_data/ad_hoc/LR_Stereo.ogg");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestLaugh_44k.ogg");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat.ogg");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeatMono.ogg");
        //loader.Load(fileData.get(), "test_data/ad_hoc/BlockWoosh_Stereo.ogg");

        // 1 + 2 channel flac
        //loader.Load(fileData.get(), "test_data/ad_hoc/KittyPurr8_Stereo_Dithered.flac");
        //loader.Load(fileData.get(), "test_data/ad_hoc/KittyPurr16_Stereo.flac");
        //loader.Load(fileData.get(), "test_data/ad_hoc/KittyPurr16_Mono.flac");
        //loader.Load(fileData.get(), "test_data/ad_hoc/KittyPurr24_Stereo.flac");

        //auto memory = ReadFile("test_data/ad_hoc/KittyPurr24_Stereo.flac"); // broken
        //loader.Load(fileData.get(), "flac", memory.buffer); // broken

        // 2-channel opus
        //loader.Load(fileData.get(), "test_data/short.opus");
        loader.Load(fileData.get(), "test_data/sb-reverie.opus");

        // 8 channel opus
        //loader.Load(fileData.get(), "test_data/Rachel8ch.opus");


        // 1 + 2 channel wavpack
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat_Float32.wv");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat_Float32_Mono.wv");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat_Int16.wv");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat_Int24.wv");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat_Int32.wv");
        //loader.Load(fileData.get(), "test_data/ad_hoc/TestBeat_Int24_Mono.wv");

        // In-memory wavpack
        // auto memory = ReadFile("test_data/ad_hoc/TestBeat_Float32.wv");
        // loader.Load(fileData.get(), "wv", memory.buffer);

        // 1 + 2 channel musepack
        //loader.Load(fileData.get(), "test_data/ad_hoc/44_16_stereo.mpc");
        //loader.Load(fileData.get(), "test_data/ad_hoc/44_16_mono.mpc");

        // In-memory ogg
        //auto memory = ReadFile("test_data/ad_hoc/BlockWoosh_Stereo.ogg");
        //loader.Load(fileData.get(), "ogg", memory.buffer);

        // In-memory Mp3
        //auto memory = ReadFile("test_data/ad_hoc/acetylene.mp3");
        //loader.Load(fileData.get(), "mp3", memory.buffer);
    }

    /* Test Recording Capabilities of AudioDevice
    fileData->samples.reserve(44100 * 5);
    fileData->channelCount = 1;
    fileData->frameSize = 32;
    fileData->lengthSeconds = 5.0;
    fileData->sampleRate = 44100;
    std::cout << "Starting recording ..." << std::endl;
    myDevice.Record(fileData->sampleRate * fileData->lengthSeconds, fileData->samples);
    */

#ifdef USE_CUDA
    cleanupCudaBuffers();
#endif

    {  // save to wave file
        AudioFile<float> a;
        a.setNumChannels(fileData->channelCount);
        a.setNumSamplesPerChannel(fileData->samples.size() / fileData->channelCount);
        a.setSampleRate(fileData->sampleRate);

        float sum = 0;
        for (int i = 0; i < fileData->channelCount; i++) {
            for (int j = 0; j < a.getNumSamplesPerChannel(); j++) {
                a.samples[i][j] = (fileData->samples[j * fileData->channelCount + i]);
                sum += a.samples[i][j];
            }
        }
        printf("len: %ld sum: %f\n", fileData->samples.size(), sum);
        if (
            (static_cast<int>(sum) != 403 || fileData->samples.size() != 21472602)
            && (static_cast<int>(sum) != 40 || fileData->samples.size() != 127712488)
            && (static_cast<int>(sum) != 719 || fileData->samples.size() != 21472602)
            )
        {
            printf("wrong results!  save to wave file\n");
            a.save("opusdec.wav", AudioFileFormat::Wave);
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
catch (const UnsupportedExtensionEx& e)
{
    std::cerr << "Caught: " << e.what() << std::endl;
}
catch (const LoadPathNotImplEx& e)
{
    std::cerr << "Caught: " << e.what() << std::endl;
}
catch (const LoadBufferNotImplEx& e)
{
    std::cerr << "Caught: " << e.what() << std::endl;
}
catch (const std::exception& e)
{
    std::cerr << "Caught: " << e.what() << std::endl;
}
