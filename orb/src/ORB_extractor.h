#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv4/opencv2/core/core.hpp>

namespace VO_ORB
{
    class ExtractorNode
    {
    public:
        ExtractorNode():split_node_flag(false){};
        ~ExtractorNode(){};
        // 分裂 4 个节点
        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        // ExtractorNode 类的迭代器，用于在 list 中迭代元素
        std::list<ExtractorNode>::iterator lit;

        // 存放特征点的当前节点
        std::vector<cv::KeyPoint> vKeys;

        // 节点的左上，右上，左下，右下坐标
        cv::Point2i UL, UR, BL, BR;

        // true：当前节点不再分；
        // false：当前节点可再分；
        bool split_node_flag;
    };

    class ORBExtractor
    {
    public:
        // 构造函数
        ORBExtractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);
        // 析构函数
        ~ORBExtractor(){}

        std::vector<cv::Mat> mvImagePyramid;
        void ExtractFeatures(cv::Mat image, cv::Mat mask, std::vector<cv::KeyPoint>& vResKeypoints, cv::_OutputArray _descriptors);
    
    protected:
        // 提取特征的相关参数
        int nfeatures;
        float scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

        // 提取描述符的pattern
        std::vector<cv::Point> pattern;

        // 存储每一层特征点的个数
        std::vector<int> mnFeaturesPerLevel;
        std::vector<int> umax;

        // 存储每一层尺度因子和逆尺度相关信息
        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSIgma2;

        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);
        std::vector<cv::KeyPoint> DistributeOctTree(
            const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
            const int &maxX, const int &minY, const int &maxY, const int &N, const int &level);
        float ComputeSingleOrientation(const cv::Mat& image, cv::Point2f pt, const std::vector<int> &umax);
        void ComputeOrbDescriptor(cv::Mat& image, cv::KeyPoint& kpt, const cv::Point* pattern, uchar* desc);

    };
};


#endif