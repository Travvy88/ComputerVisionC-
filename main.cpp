#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include "classifier.h"
#include "model.h"
#include "facenet.h"
#include <filesystem>
namespace fs = std::filesystem;
int main()
{


    std::string path = "C:\\Users\\kdjdj\\Desktop\\C\\Labs 2 course\\TorchTest_3\\Photos";
    std::string emb_path = "C:\\Users\\kdjdj\\Desktop\\C\\Labs 2 course\\TorchTest_3\\Photos\\000069.jpg";

    std::vector<std::string> paths, names;
    for (const auto & entry : fs::directory_iterator(path))
    {
        paths.push_back(entry.path().string());
        names.push_back(entry.path().filename().string());
    }

    std::vector<cv::Mat> imgs;
    for (int i = 0; i < paths.size(); i++) {
        cv::Mat img = cv::imread(paths[i], cv::IMREAD_COLOR);
        imgs.push_back(img);
    }


    Classifier a("C:\\Users\\kdjdj\\Desktop\\C\\Labs 2 course\\TorchTest_3\\traced_resnet_model.pt");
    a.cuda();
    int *data1 = a.predict(imgs);

    FaceNet b("C:\\Users\\kdjdj\\Desktop\\C\\Labs 2 course\\TorchTest_3\\tracedMobileFaceNet.pth");
    b.cuda();
    cv::Mat emb = cv::imread(emb_path, cv::IMREAD_COLOR);
    b.makeEmbedding(emb);
    int *data2 = b.predict(imgs);
    for (int i = 0; i < imgs.size(); i++)
    {
        std::string sex;
        std::string found = "";
        if (data1[i] == 0)
            sex = "Male";
        else
            sex = "Female";
        if (data2[i] == 1)
            found = " - Found";

        std::cout << names[i] << " - " << sex << found << std::endl;
    }

}

