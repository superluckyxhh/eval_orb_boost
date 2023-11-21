#include <iostream>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include "src/ORB_extractor.h"

using namespace std;
using namespace cv;

int main() {
    cv::Mat img, img_gray, img_show;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    std::string image_root = "/home/xiaohunan/WorkSpace/MyCode/VISUAL_ODOMETRY/images/test_images/";
    std::string save_image_root = "/home/xiaohunan/WorkSpace/MyCode/VISUAL_ODOMETRY/images/plot_images/";
    std::string image_name = "6.ppm";
    std::string image_path = image_root + image_name;
    std::string save_image_path = save_image_root + image_name.substr(0, 1) + ".png";
    cout << save_image_path << endl;

    img = cv::imread(image_path);
    img.copyTo(img_show);
    img_gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    
    VO_ORB::ORBExtractor orb = VO_ORB::ORBExtractor(1000, 1.2f, 8, 20, 15);
    orb.ExtractFeatures(img_gray, cv::Mat(), keypoints, descriptors);
    
    // Plot
    for (size_t i = 0; i < keypoints.size(); ++i) {
        cv::KeyPoint kpt = keypoints[i];
        cv::circle(img_show, kpt.pt, 2, cv::Scalar(0, 255, 0), 2);
    }
    cout << "[VO:ORB] Keypoints shape: " << keypoints.size() << endl;
    cout << "[VO:ORB] Descriptor shape: " << descriptors.size() << endl;
    cv::imwrite(save_image_path, img_show);
    return 0;
}