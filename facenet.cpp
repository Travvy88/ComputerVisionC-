#include "facenet.h"

int *FaceNet::predict(std::vector<cv::Mat> & vec)
{
    at::Tensor tensor;
    for (int i = 0; i < vec.size(); i++)
    {


        cv::Mat img = vec[i];
        cv::Mat gray;

           //-- Detect faces
        /*
        std::vector<cv::Rect> faces;
        cv::CascadeClassifier face_cascade;
        cv::cvtColor(img, gray,  cv::COLOR_BGR2GRAY);
        cv::equalizeHist( gray, gray );
        face_cascade.detectMultiScale( gray, faces );
        if (faces.size() > 0)
        {
            img = img(faces[0]);
        }
        */
        cv::resize(img, img, cv::Size(112, 112));

        at::Tensor tensor_image = ToTensor(img, true);
        if (i == 0)
            tensor = tensor_image;
        else
            tensor = at::cat({tensor, tensor_image});
    }

    at::Tensor output = forward(tensor);

    at::Tensor dist = (output - embedding).pow(2).sum(1).pow(0.5);
    at::Tensor result = (dist - 8 > 19);
    //std::cout << result;
    int* data = new int[result.size(0)];
    for (int i = 0; i < result.size(0); i++)
        *(data + i) = result[i].item<int>();

    return data;
}

void FaceNet::makeEmbedding(cv::Mat &img)
{
    cv::resize(img, img, cv::Size(112, 112));
    at::Tensor tensor_image = ToTensor(img, true);
    embedding = forward(tensor_image);

}
