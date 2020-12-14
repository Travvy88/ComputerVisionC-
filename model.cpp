#include "model.h"

Model::Model()
{
    uploaded = false;
}

Model::Model(std::string path_)
{
    if (updateWeights(path_))
    {
        device = at::kCPU;
        path = path_;
        module.eval();
    }

}

Model::Model(const Model &other)
{
    module = other.getModule();
    uploaded = other.isUploaded();
    path = other.getPath();
    device = other.getDevice();
    module.eval();
}

Model::~Model()
{
    module.to(at::kCPU);
}

at::Tensor Model::forward(const at::Tensor &tensor)
{
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor.to(device));
    at::Tensor output = module.forward(inputs).toTensor();
    output.to(at::kCPU);

    return output;
}

bool Model::updateWeights(std::string path_)
{

    std::ifstream in(path_, std::ios_base::binary);

    try {
        module = torch::jit::load(in);
    }  catch (const c10::Error& e) {
        std::cout << "\n DONT UPLOADED \n ";
        uploaded = false;
        return false;
    }

    module = torch::jit::load(in);
    uploaded = true;
    path = path_;
    return true;
}

torch::jit::script::Module Model::getModule() const
{
    return module;
}

bool Model::cuda()
{
        if (torch::cuda::cudnn_is_available())
        {
            device = at::kCUDA;
            module.to(device);
            return true;
        }
        else
        {
            device = at::kCPU;
            module.to(device);
            return false;
        }
}

void Model::cpu()
{
    device = at::kCPU;
    module.to(device);
}

bool Model::isUploaded() const
{
    return uploaded;
}

c10::DeviceType Model::getDevice() const
{
    return device;
}

std::string Model::getPath() const
{
    return path;
}

at::Tensor Model::ToTensor(cv::Mat img, bool unsqueeze, int unsqueeze_dim) const
{

        at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);
        tensor_image = tensor_image.permute({ 2,0,1 });
        if (unsqueeze)
        {
            tensor_image.unsqueeze_(unsqueeze_dim);

        }

        tensor_image = tensor_image.toType(c10::kFloat).div(255);

        return tensor_image;


}

