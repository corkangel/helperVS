#include <iostream>
#include <random>
#include <cassert>
#include <array>

#include <fstream>
#include <iterator>

#include "model.h"
#include "render.h"

// each image is 32 x 32 x 3
const int imageArraySize = 32 * 32 * 3;
const int numImages = 400;
const int numCategories = 10;

// each image in the data is this many bytes
const int imageDataSize = 1 + 1024*3;

// constant to covert from 255 to float in 0-to-1 range
const float convert255 = float(1) / float(255);

void loadImages(const char* filename, matrix& images, matrix& categories)
{
    std::ifstream input(filename, std::ios::binary );
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

    // flatten and scale image data from the file into a matrix
    for (int i=0; i < numImages; i++)
    {
        images[i].resize(1024 * 3);

        unsigned char* imagePtr = &buffer[i*imageDataSize];
        unsigned char* r = &imagePtr[1];
        unsigned char* g = &imagePtr[1+1024];
        unsigned char* b = &imagePtr[1+1024+1024];

        for (int pixel = 0; pixel < 1024; pixel++)
        {
            const int o = pixel*3;
            images[i][o+0] = (*r++) * convert255;
            images[i][o+1] = (*g++) * convert255;
            images[i][o+2] = (*b++) * convert255;
        }

        // hot encode categories
        categories[i].resize(numCategories);
        unsigned char category = *imagePtr;
        for (int c=0; c < numCategories; c++)
        {
            categories[i][c] = (int(category) == c) ? 1.0f : 0.0f;
        }
    }
}


class ImageTask : public hhTask
{
public:

    void Configure(hhModel& model) override
    {
        learningRate = 0.0001f;
        epochs = 1;
        batchSize = 32;

        AddLayer(hhLayerType::Input, imageArraySize);
        AddLayer(hhLayerType::Relu, 200, hhTaskOperation::NormalizeWeights | hhTaskOperation::NormalizeValues);
        AddLayer(hhLayerType::Relu, 150, hhTaskOperation::NormalizeWeights | hhTaskOperation::NormalizeValues);
        AddLayer(hhLayerType::Softmax, numCategories);

        inputs.resize(numImages);
        targets.resize(numImages);        
        loadImages("../Resources/Data/data_batch_1.bin", inputs, targets);
    }
};

const bool gDrawImages = false;

int main(int, char**)
{
    // stable random values
    srand(101010101);

    ImageTask task;
    hhModel model;
    model.Configure(task);

    matrix testImages(numImages);
    matrix testCategories(numImages);
    loadImages("../Resources/Data/test_batch.bin", testImages, testCategories);

    renderWindow rw;

    // for rendering only.
    std::vector<unsigned char> drawBuffer;
    if (1)
    {
        std::ifstream input = std::ifstream("../Resources/Data/test_batch.bin", std::ios::binary);
        drawBuffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(input), {});
    }

    // stable random set of test indices
    const int numTests = 16;    
    std::array<int, numTests> testIds;
    for (int j=0; j< numTests; j++)
    testIds[j] = rand() % numImages;
    std::array<int, numTests> testedPredictions;

    float loss = 0;
    int trainingRuns = 0;
    bool running = 1;
    while (running)
    {
        if (!gDrawImages)
        {
            model.Train();

            trainingRuns++;
            if (1)
            {
                column predictions(numCategories);

                int numCorrect = 0;

                // calculate accuracy on a subset of test images
                for (int t = 0; t < numTests; t++)
                {
                    const int index = testIds[t];

                    column predictions = model.Predict(testImages[index]);

                    int predictedCategory = argmax(predictions);
                    int testCategory = argmax(testCategories[index]);

                    if (predictedCategory == testCategory)
                        numCorrect += 1;

                    testedPredictions[t] = predictedCategory;
                }
                loss = float(numCorrect) / numTests;
                printf("loss = %f\n", loss);
            }
        }

        rw.ProcessEvents(running);
        rw.BeginDisplay();
        rw.DisplayTitle(model.numEpochs, model.lastTrainError, "Images");

        for (int t = 0; t < numTests; t++)
        {
            const int index = testIds[t];
            const int iz = 1 + 1024 * 3;
            unsigned char* r = &drawBuffer[iz * index + 1];
            unsigned char* g = &drawBuffer[iz * index + 1 + (1024)];
            unsigned char* b = &drawBuffer[iz * index + 1 + (1024) + (1024)];
            rw.DisplayImage(&r[0], &g[0], &b[0], 40, 160 + t * 40);

            char tmp[100];
            snprintf(tmp, sizeof(tmp), "%d / %d  (pred/act) index: %d", testedPredictions[t], argmax(testCategories[testIds[t]]), testIds[t]);
            rw.DrawString(tmp, 100, 160 + t * 40);
            rw.DrawRect(90, 160 + t * 40, 10, 10, sf::Color::White);
        }

        if (gDrawImages)
        {
            const int numImages = 20;
            for (int y = 0; y < numImages; y++)
            {
                for (int x = 0; x < numImages; x++)
                {
                    const int iz = 1 + 1024 * 3;
                    int o = y * numImages + x;
                    unsigned char* r = &drawBuffer[iz * o + 1];
                    unsigned char* g = &drawBuffer[iz * o + 1 + (1024)];
                    unsigned char* b = &drawBuffer[iz * o + 1 + (1024) + (1024)];
                    rw.DisplayImage(&r[0], &g[0], &b[0], 20 + x * 40, 160 + y * 40);
                }
            }
        }

        rw.EndDisplay();
    }
}
