
#pragma once

#include "task.h"

class hhLayer
{
public:
    virtual ~hhLayer() = default;

    hhLayer(const int numNeurons, const int numInputs, const hhTaskOperation op);

    virtual void Forward(const column& input) = 0;
    virtual float Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets) { return 0.0f;}

    const int numNeurons;
    const int numInputs;
    const hhTaskOperation operation;

    column activationValue;
    column errors;
    column biases;
    matrix weights;  

    void InitializeWeights(const float avg, const float dist);
    void NormalizeWeights(); 
    void NormalizeValues();
};

class hhInputLayer : public hhLayer
{
public:
    hhInputLayer(const int numNeurons, const int numInputs, const hhTaskOperation op);

    void Forward(const column& input) override;
};

class hhDenseLayer : public hhLayer
{
public:
    hhDenseLayer(const int numNeurons, const int numInputs, const hhTaskOperation op);

    void UpdateWeightsAndBiases(const hhLayer& previous, float learningRate);
};

class hhSigmoidLayer : public hhDenseLayer
{
public:
    hhSigmoidLayer(const int numNeurons, const int numInputs, const hhTaskOperation op);
    void Forward(const column& input) override;
    float Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets) override;
};

class hhReluLayer : public hhDenseLayer
{
public:
    hhReluLayer(const int numNeurons, const int numInputs, const hhTaskOperation op);
    void Forward(const column& input) override;
    float Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets) override;
};

class hhSoftmaxLayer : public hhDenseLayer
{
public:
    hhSoftmaxLayer(const int numNeurons, const int numInputs, const hhTaskOperation op);
    void Forward(const column& input) override;
    float Backward(const hhLayer& previous, hhLayer* next, float learningRate, const column& targets) override;
};



class hhModel
{
public:

    void Configure(hhTask& task);
    hhLayer* AddLayer(const hhLayerType type, const int numNeurons, const hhTaskOperation op);

    void Forward(const column& input);
    float Backward(const column& targets);
    void Train();

    const column& Predict(const column& input);

    hhTask* task = nullptr;

    int numEpochs = 0;
    float lastTrainError = 0;
    float lastTrainTime = 0;

    std::vector<hhLayer*> layers;
    std::vector<int> indicies;
};


