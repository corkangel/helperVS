#include "utils.h"

class hhModel;

enum class hhLayerType
{
    None,
    Input,
    Sigmoid,
    Relu,
    Softmax,
};

enum class hhTaskOperation
{
    Default,
    Normalize
};

struct hhTaskLayer
{
    hhLayerType type;
    int numNeurons;
    hhTaskOperation op;
};

class hhTask
{
public:

    virtual ~hhTask() = default;

    void AddLayer(hhLayerType type, int numNeurons, hhTaskOperation op = hhTaskOperation::Default)
    {
        layers.push_back({type, numNeurons, op});
    }

    matrix inputs;
    matrix targets;
    
    float learningRate;
    int epochs;
    int batchSize;

    std::vector<hhTaskLayer> layers;

    virtual void Configure(hhModel& model) = 0;
    virtual void Render(hhModel& model) {};    
};

