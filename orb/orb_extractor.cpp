#include <pybind11/pybind11.h>
#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>

#include "src/ORB_extractor.h"
#include "opencv_type_casters.h"

using namespace std;
using namespace cv;
using namespace VO_ORB;

PYBIND11_MODULE(orb_slam2_extractor, m)
{   
    m.doc() = "pybind11 plugin for ORB SLAM2 Extractor";
    pybind11::class_<ORBExtractor>(m, "ORBExtractor" )
        .def(pybind11::init<int, float, int, int, int>(), "nfeatures"_a, "scaleFactor"_a, "nlevels"_a, "iniThFAST"_a=20, "minThFAST"_a=10)
        .def( "ExtarctFeatures", 
            [](ORBExtractor& vo, cv::Mat  &image)
            {
                cv::Mat mask = cv::Mat();
                std::vector<cv::KeyPoint> vResKeypoints;
                cv::Mat descriptors;
                vo.ExtractFeatures(image, mask, vResKeypoints, descriptors);
                return std::make_tuple(vResKeypoints, descriptors);
            });
}