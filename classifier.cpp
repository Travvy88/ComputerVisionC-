#include "classifier.h"



int *Classifier::predict(std::vector<cv::Mat> & vec)
{
    at::Tensor tensor;
    for (int i = 0; i < vec.size(); i++)
    {

        cv::Mat img = vec[i];
        cv::resize(img, img, cv::Size(224, 224));

        at::Tensor tensor_image = ToTensor(img, true);
        if (i == 0)
            tensor = tensor_image;
        else
            tensor = at::cat({tensor, tensor_image});
    }

    at::Tensor output = forward(tensor);
    at::Tensor maxes = std::get<1>(output.max(1));
    int *data = new int[maxes.size(0)];

    for (int i = 0; i < maxes.size(0);i++)
           *(data+i) = maxes[i].item<int>();

    return data;
}
