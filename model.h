#ifndef MODEL_H
#define MODEL_H

#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

class Model
{
public:
    Model();
    Model(std::string);
    Model(const Model &);
    virtual ~Model();

    virtual int* predict(std::vector<cv::Mat> &)= 0;
    at::Tensor forward(const at::Tensor &);
    bool updateWeights(std::string);

    torch::jit::script::Module getModule() const;
    bool cuda();
    void cpu();
    bool isUploaded() const;
    torch::DeviceType getDevice() const;
    std::string getPath() const;
    at::Tensor ToTensor(cv::Mat img, bool unsqueeze=false, int unsqueeze_dim = 0) const;

protected:
    torch::jit::script::Module module;
    std::string path;
    torch::DeviceType device;
    bool uploaded;

};

#endif // MODEL_H
