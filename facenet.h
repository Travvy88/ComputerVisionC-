#ifndef FACENET_H
#define FACENET_H

#include <torch/script.h>
#include "model.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

class FaceNet: public Model
{
public:
    FaceNet(std::string path_): Model(path_) {}
    virtual int* predict(std::vector<cv::Mat> &);
    void makeEmbedding(cv::Mat &);

private:
   at::Tensor embedding;

};

#endif // FACENET_H
