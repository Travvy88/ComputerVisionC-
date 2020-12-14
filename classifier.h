#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <torch/script.h>
#include "model.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

class Classifier: public Model
{
public:
    Classifier(std::string path_): Model(path_) {}
    int *predict(std::vector<cv::Mat> &);

};

#endif // CLASSIFIER_H
