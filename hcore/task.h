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

enum class hhTaskOperation : int
{
    Default,
    NormalizeWeights = 0x01,
    NormalizeValues = 0x02,
};

inline hhTaskOperation operator|(hhTaskOperation a, hhTaskOperation b)
{
    return static_cast<hhTaskOperation>(static_cast<int>(a) | static_cast<int>(b));
}

inline hhTaskOperation operator&(hhTaskOperation a, hhTaskOperation b)
{
    return static_cast<hhTaskOperation>(static_cast<int>(a) & static_cast<int>(b));
}


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

