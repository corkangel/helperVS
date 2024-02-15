
A neural network from scratch in C++.

Configurable in code:

```
class ColorTask : public hhTask
{
public:

    void Configure(hhModel& model) override
    {
        learningRate = 0.01f;
        epochs = 40;
        batchSize = 0;
        inputs = {
            {.1f, .1f},
            {.1f, .3f},
            {.2f, .7f},
            {.1f, .9f}};

        targets = {
            {.9f, .0f, .0f},         
            {.9f, .9f, .9f},         
            {.0f, .0f, .9f},                 
            {.0f, .9f, .0f}};

        AddLayer(hhLayerType::Input, 2);
        AddLayer(hhLayerType::Relu, 5, hhTaskOperation::Default);
        AddLayer(hhLayerType::Sigmoid, 3, hhTaskOperation::Default);
    }
};
```

The demo application trains on some grid positions and colors, and predicts a color gradient from them.

![](helper.png?raw=true)

