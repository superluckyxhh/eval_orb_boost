#include <iostream>
#include <string>
#include <vector>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include "ORB_extractor.h"

using namespace std;
using namespace cv;

namespace VO_ORB
{
    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    // 为高斯模糊而预留的区域
    const int EDGE_THRESHOLD = 19;

    // Descriptor pattern
    static int bit_pattern_31_[256*4] =
    {
        8,-3, 9,5/*mean (0), correlation (0)*/,
        4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
        -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
        7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
        2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
        1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
        -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
        -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
        -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
        10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
        -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
        -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
        7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
        -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
        -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
        -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
        12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
        -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
        -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
        11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
        4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
        5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
        3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
        -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
        -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
        -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
        -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
        -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
        -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
        5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
        5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
        1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
        9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
        4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
        2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
        -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
        -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
        4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
        0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
        -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
        -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
        -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
        8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
        0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
        7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
        -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
        10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
        -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
        10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
        -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
        -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
        3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
        5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
        -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
        3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
        2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
        -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
        -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
        -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
        -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
        6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
        -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
        -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
        -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
        3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
        -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
        -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
        2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
        -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
        -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
        5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
        -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
        -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
        -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
        10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
        7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
        -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
        -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
        7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
        -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
        -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
        -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
        7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
        -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
        1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
        2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
        -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
        -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
        7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
        1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
        9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
        -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
        -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
        7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
        12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
        6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
        5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
        2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
        3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
        2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
        9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
        -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
        -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
        1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
        6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
        2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
        6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
        3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
        7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
        -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
        -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
        -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
        -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
        8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
        4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
        -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
        4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
        -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
        -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
        7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
        -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
        -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
        8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
        -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
        1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
        7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
        -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
        11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
        -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
        3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
        5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
        0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
        -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
        0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
        -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
        5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
        3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
        -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
        -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
        -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
        6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
        -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
        -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
        1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
        4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
        -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
        2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
        -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
        4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
        -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
        -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
        7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
        4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
        -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
        7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
        7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
        -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
        -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
        -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
        2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
        10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
        -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
        8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
        2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
        -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
        -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
        -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
        5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
        -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
        -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
        -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
        -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
        -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
        2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
        -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
        -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
        -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
        -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
        6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
        -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
        11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
        7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
        -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
        -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
        -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
        -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
        -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
        -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
        -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
        -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
        1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
        1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
        9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
        5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
        -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
        -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
        -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
        -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
        8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
        2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
        7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
        -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
        -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
        4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
        3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
        -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
        5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
        4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
        -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
        0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
        -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
        3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
        -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
        8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
        -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
        2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
        10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
        6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
        -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
        -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
        -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
        -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
        -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
        4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
        2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
        6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
        3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
        11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
        -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
        4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
        2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
        -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
        -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
        -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
        6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
        0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
        -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
        -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
        -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
        5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
        2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
        -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
        9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
        11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
        3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
        -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
        3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
        -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
        5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
        8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
        7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
        -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
        7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
        9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
        7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
        -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
    };

    ORBExtractor::ORBExtractor(int _nfeatures, float _scaleFactor, 
        int _nlevels, int _iniThFAST, int _minThFAST):
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
        iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {   
        // 计算每一层的尺度和逆尺度
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0] = 1.f;
        mvLevelSigma2[0] = 1.f;

        for (int i = 1; i < nlevels; i++)
        {
            mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
            mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSIgma2.resize(nlevels);

        for (int i = 0; i < nlevels; i++)
        {
            mvInvScaleFactor[i] = 1.f / mvScaleFactor[i];
            mvInvLevelSIgma2[i] = 1.f / mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);
        mnFeaturesPerLevel.resize(nlevels);
        
        // 利用公式推导出每层应该分布的特征点数量
        float factor = 1.f / scaleFactor;
        // 计算出第一层应该分配的特征点数量
        float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));
        // 后续每一层特征点数量根据比例从第一层换算即可
        int sumFeatures = 0;
        for (int i = 0; i < nlevels - 1; i++)
        {
            mnFeaturesPerLevel[i] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[i];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

        // 对pattern进行操作
        // 将长度为 1024 的int型一维数组，变成了一个长度为 512 的Point类型的一维数组
        const int npoints = 512;
        const cv::Point* pattern0 = (const Point*) bit_pattern_31_;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

        // umax 表示 1/4 圆中行 u 轴坐标值
        umax.resize(HALF_PATCH_SIZE + 1);
        int vmax = cvFloor(HALF_PATCH_SIZE * std::sqrt(2.f) / 2.f + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * std::sqrt(2.f) / 2.f);
        const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE; 

        for (int v = 0; v <= vmax; v++)
        {   
            int u_coord = cvRound(sqrt(hp2 - v * v));
            umax[v] = u_coord;
        }
        
        for (int v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; v--)
        {
            while (umax[v0] == umax[v0 + 1])
            {
                ++v0;
            }
            umax[v] = v0;
            ++v0; 
        }
    }

    void ORBExtractor::ExtractFeatures(cv::Mat image, cv::Mat mask, std::vector<cv::KeyPoint>& vResKeypoints, cv::_OutputArray _descriptors)
    {
        if (image.empty())
        {
            cout << "Failed to load image" << endl;
            return;
        }
        assert(image.type() == CV_8UC1);
        
        // Step 1. 计算图像金字塔
        cout << "############### Step 1. Compute Image Pyramid ###############" << endl;
        ComputePyramid(image);

        // Step 2. 检测特征点，并使用四叉树分配每层特征点数量
        cout << "############### Step 2. Detecting and distributing ###############" << endl;
        // 多层金字塔，每层多个特征，用两个 vector 装载所有特征点
        std::vector<std::vector<cv::KeyPoint>> allKeypoints;
        ComputeKeyPointsOctTree(allKeypoints);


        cout << "############### Step 3. Describing ###############" << endl;
        cv::Mat descriptors;
        // 所有层数的特征点之和
        int nkeypoints = 0;
        for (int level = 0; level < nlevels; level++)
            nkeypoints += (int)allKeypoints[level].size();
        
        if (nkeypoints == 0)
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 32, CV_8U);
            descriptors = _descriptors.getMat();
        }

        vResKeypoints.clear();
        vResKeypoints.reserve(nkeypoints);
        
        int offset = 0;
        for (int level = 0; level < nlevels; level++)
        {
            std::vector<cv::KeyPoint> keypoints = allKeypoints[level];
            int nkeypointsLevel = (int)keypoints.size();
            if (nkeypointsLevel == 0)
                continue;
            
            cv::Mat workingMat = mvImagePyramid[level].clone();
            cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 3, cv::BORDER_REFLECT_101);
            // 取出本层特征点数量的描述符
            cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);

            for (size_t i = 0; i < keypoints.size(); i++)
                ComputeOrbDescriptor(workingMat, keypoints[i], &pattern[0], desc.ptr((int)i));

            offset += nkeypointsLevel;
            // 对非第一层的特征点进行缩放
            if (level != 0)
            {
                float scale = mvScaleFactor[level];
                for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(), keypointEnd = keypoints.end();
                    keypoint != keypointEnd; keypoint++)
                    keypoint->pt *= scale;
            }
            vResKeypoints.insert(vResKeypoints.end(), keypoints.begin(), keypoints.end());
        }
        return ;
    }

    void ORBExtractor::ComputePyramid(cv::Mat image)
    {
        for (int i = 0; i < nlevels; i++)
        {
            cout << "--------- level: " << i << " ---------" << endl;
            float scale = mvInvScaleFactor[i];
            cout << "inv scale factor: " << scale << endl;

            cv::Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
            cout << "scale size: "<< sz << endl;
        
            cv::Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            cout << "whole size (add edge): " << wholeSize << endl;

            cv::Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[i] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
            
            if (i == 0)
            {
                cv::copyMakeBorder(image, temp, 
                    EDGE_THRESHOLD, EDGE_THRESHOLD, 
                    EDGE_THRESHOLD, EDGE_THRESHOLD, 
                    cv::BORDER_REFLECT_101);
            }
            else
            {
                cv::resize(mvImagePyramid[i - 1], mvImagePyramid[i], sz, 0, 0, cv::INTER_LINEAR);
                cv::copyMakeBorder(mvImagePyramid[i - 1], temp, 
                    EDGE_THRESHOLD, EDGE_THRESHOLD, 
                    EDGE_THRESHOLD, EDGE_THRESHOLD, 
                    cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
            }
        }
    }

    void ORBExtractor::ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints)
    {
        // allKeypoints 中每一个元素是 vector, 存放本层提取到的特征点
        allKeypoints.resize(nlevels);
        // 每个网格的尺寸均为 30 个像素
        const int cell_size = 30;

        // 对每层金字塔图像循环提取特征点
        for (int level = 0; level < nlevels; level++)
        {   
            cout << "--------- " << "level: " << level << " ---------" << endl;
            cout << "detecting keypoints..." << endl;
            // 取出本层图像和对应尺寸
            cv::Mat image = mvImagePyramid[level];

            // 提取“边界范围”
            const int minBorderX = EDGE_THRESHOLD - 3;
            const int minBorderY = EDGE_THRESHOLD - 3;
            const int maxBorderX = image.cols - EDGE_THRESHOLD + 3;
            const int maxBorderY = image.rows - EDGE_THRESHOLD + 3;

            // 临时存放本层特征点，申请 10 倍内存
            std::vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures * 10);

            const float width = (float)(maxBorderX - minBorderX);
            const float height = (float)(maxBorderY - minBorderY);
            
            // 行列网格数
            const int nCols = width / cell_size;
            const int nRows = height / cell_size;
            // 每个网格的宽度
            const int wCell = std::ceil(width / nCols);
            const int hCell = std::ceil(height / nRows);
            
            // 在该层按行列的每个网格提取 FAST 特征点
            for (int i = 0; i < nRows; i++)
            {
                // 每个行网格的起始 Y 值，最大的 Y 值
                const float iniY = minBorderY + i * hCell;
                float maxY = iniY + hCell + 6;
                if (iniY >= maxBorderY - 3)
                    continue;
                if (maxY > maxBorderY)
                    maxY = maxBorderY;
                
                for (int j = 0; j < nCols; j++)
                {
                    const float iniX = minBorderX + j * wCell;
                    float maxX = iniX + wCell + 6;
                    if (iniX >= maxBorderX - 6)
                        continue;
                    if (maxX > maxBorderX)
                        maxX = maxBorderX;

                    // 每个网格建一个 vector 存储小格子的特征点
                    std::vector<cv::KeyPoint> vKeysCell;
                    // 在每个格子里提取角点
                    cv::FAST(image.rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, iniThFAST, true);
                    if (vKeysCell.empty())
                    {
                        cv::FAST(image.rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, minThFAST, true);
                    }

                    if (!vKeysCell.empty())
                    {
                        // 将特征点从网格坐标系转到本层图像的“边界范围”
                        for (std::vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                        {
                            (*vit).pt.x += j * wCell;
                            (*vit).pt.y += i * hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }
                }
            }
            cout << "FAST num keypoints: " << vToDistributeKeys.size() << endl;

            // 引用传递，后续修改 keypoints ，对应的传入值同样会修改
            std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
            keypoints.reserve(nfeatures);

            // FAST 角点数量未知，使用四叉树过滤每层特征点，使均为分布
            // vToDistributeKeys: 每层未经过均匀化的特征点
           
            // 这个函数好像Distribute没有起作用
            keypoints = DistributeOctTree(
                vToDistributeKeys, minBorderX, maxBorderX, 
                minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);
            
            cout << "distribute num keypoints: " << keypoints.size() << endl;
            
            // 计算 FAST 时 PatchSize, 可以理解为特征点所代表的范围
            const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
            
            // 因为是在扩充半径图像上进行 FAST 特征检测的，所以需要将特征点坐标加上边缘扩充的部分
            const int nkps = keypoints.size();
            for(int i = 0; i < nkps ; i++)
            {
                keypoints[i].pt.x += minBorderX;
                keypoints[i].pt.y += minBorderY;
                keypoints[i].octave = level;
                // 计算方向的 patch,也被称为特征点半径
                keypoints[i].size = scaledPatchSize;
            }
        }

        // 循环计算每层特征点的方向
        for (int level = 0; level < nlevels; level++)
        {   
            cout << "--------- " << "level: " << level << " ---------" << endl;
            cout << "compute orientation..." << endl;
            cv::Mat image = mvImagePyramid[level];
            std::vector<cv::KeyPoint> keypoints = allKeypoints[level];

            // 对第 i 层，循环所有特征点，计算方向
            for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(), keypointEnd = keypoints.end(); 
                keypoint != keypointEnd; keypoint++)
                // 方向用角度表示，范围 [0, 360)，顺时针
                keypoint->angle = ComputeSingleOrientation(image, keypoint->pt, umax);
            cout << "compute orientation done" << endl;
        }
    }

    /*
        四叉树分配特征点的步骤：
        1. 若图像宽度较大，则先将图像分成 w / h 份节点 node；
        2. 若节点内特征点数目大于 1，把每个节点分成 4 个节点。若节点内特征点为空，则删掉该节点；
        3. 新分裂的节点内特征点数目大于 1，再分裂成 4 个节点，以此类推；
        4. 如果达到终止条件，或无法再分裂，则停止；
        5. 从每个节点中选择选择一个质量最好的特征点；
        终止条件：节点 node 总数 > 分配特征点数量
    */
    std::vector<cv::KeyPoint> ORBExtractor::DistributeOctTree(
        const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
        const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
    {
        // 初始node的个数
        const int nIni = std::round(static_cast<float>(maxX - minX) / (maxY - minY));

        // 计算 x 方向上初始节点的水平宽度
        const float hX = static_cast<float>(maxX - minX) / nIni;
           
        // 新建 ExtractorNode 类的 list，用于存放节点 Node
        std::list<ExtractorNode> lNodes;

        // 新建 ExtractorNode 指针类型的 vector，用于存放每次循环向 lNodes 添加元素后，其最后元素的地址
        std::vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);

        // 只对 x 方向上的节点划分，因为循环中所有的 Y 坐标值都是相同的
        for (int i = 0; i < nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
            ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
            ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
            ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        // 遍历所有特征点，按照 x 方向上的位置放到对应的节点中
        for (size_t i = 0; i < vToDistributeKeys.size(); i++)
        {
            const cv::KeyPoint &kp = vToDistributeKeys[i];
            int node_idx = kp.pt.x / hX;
            // 类指针用->访问成员变量，类用.访问成员变量
            vpIniNodes[node_idx]->vKeys.push_back(kp);
        }

        // 构建迭代器，指向 lNodes 的第一个元素
        std::list<ExtractorNode>::iterator lit = lNodes.begin();
        // 循环遍历节点，处理只含一个特征点或者不含特征点情况
        while (lit != lNodes.end())
        { 
            // 当前节点只含有 1 个特征点，不继续分裂该节点
            if (lit->vKeys.size() == 1)
            {
                lit->split_node_flag = true;
                lit++;
            }
            else if (lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        // 分裂是否完成，是否停止迭代
        bool split_finish = false;
        int iteration = 0;
        // 存储尺寸和指向该节点的指针
        std::vector<std::pair<int, ExtractorNode*>> vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size() * 4);

        while (!split_finish)
        {
            iteration++;
            // 当前总节点数
            int preSize = lNodes.size();
            lit = lNodes.begin();
            // 特征点大于 1 的节点数
            int nToExpand = 0;

            vSizeAndPointerToNode.clear();
            // 循环分裂节点，并重新分配给每个节点分配特征点
            while (lNodes.end() != lit)
            {
                if (lit->split_node_flag)
                {
                    lit++;
                    continue;
                }
                // 可以再分裂
                else
                {
                    ExtractorNode n1, n2, n3, n4;
                    lit->DivideNode(n1, n2, n3, n4);

                    // 分别对 4 个子节点执行相同操作
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        // 若此时特征点数仍大于 1，说明非叶子节点
                        if (n1.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                            // 后加入的 node 放到 node list的最前面，保证进入新的迭代时，“后加入的先分配”
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    // 删除父节点
                    lit = lNodes.erase(lit);
                    continue;
                }
            }

            // 判断是否满足终止条件
            if ((int)lNodes.size() >= N || (int)lNodes.size() == preSize)
                split_finish = true;
            // 若当前节点数加3*nToExpand大于预分配特征点数，则说明分裂节点使得特征点不够均匀
            else if((int)lNodes.size() + nToExpand * 3 > N)
            {
                while (!split_finish)
                {
                    preSize = lNodes.size();
                    std::vector<std::pair<int, ExtractorNode*>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();
                    // 越往后的节点包含的特征点数量越多，即越可以分裂
                    std::sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());

                    for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                    {
                        ExtractorNode n1, n2, n3, n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                        if (n1.vKeys.size() > 0)
                        {
                            lNodes.push_front(n1);
                            if (n1.vKeys.size() > 1)
                            {
                                nToExpand++;
                                vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        if (n2.vKeys.size() > 0)
                        {
                            lNodes.push_front(n2);
                            if (n2.vKeys.size() > 1)
                            {
                                nToExpand++;
                                vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        if (n3.vKeys.size() > 0)
                        {
                            lNodes.push_front(n3);
                            if (n3.vKeys.size() > 1)
                            {
                                nToExpand++;
                                vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        if (n4.vKeys.size() > 0)
                        {
                            lNodes.push_front(n4);
                            if (n4.vKeys.size() > 1)
                            {
                                nToExpand++;
                                vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        // 删除父节点
                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);
                        if ((int)lNodes.size() >= N)
                            break;
                    }
                    
                    if ((int)lNodes.size() >= N || (int)lNodes.size() == preSize)
                        split_finish = true;
                }
            }
        }

        // 存放结果
        std::vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(nfeatures);
        for (std::list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
        {
            std::vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
            cv::KeyPoint* pKp = &vNodeKeys[0];
            float maxResponse = pKp->response;

            for (size_t k = 1; k < vNodeKeys.size(); k++)
            {
                if (maxResponse < vNodeKeys[k].response)
                {
                    pKp = &vNodeKeys[k];
                    maxResponse = vNodeKeys[k].response;
                }
            }
            // 将最大响应值特征装入结果
            vResultKeys.push_back(*pKp);
        }

        return vResultKeys;
    }


    void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        // 计算上一个节点一半长度
        const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
        const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

        // 按照一半长度，重新对坐标赋值
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x + halfX, UL.y);
        n1.BL = cv::Point2i(UL.x, UL.y + halfY);
        n1.BR = cv::Point2i(UL.x + halfX,UL.y + halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x, UL.y + halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x, BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        // 依次遍历四个子节点，并将父节点特征点分配到不同子节点
        for(size_t i = 0; i < vKeys.size(); i++)
        {
            const cv::KeyPoint &kp = vKeys[i];
            if(kp.pt.x < n1.UR.x)
            {
                if(kp.pt.y < n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if(kp.pt.y < n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        // 如果哪个节点里面不能再分了，就作标记
        if(n1.vKeys.size() == 1)
            n1.split_node_flag = true;
        if(n2.vKeys.size() == 1)
            n2.split_node_flag = true;
        if(n3.vKeys.size() == 1)
            n3.split_node_flag = true;
        if(n4.vKeys.size() == 1)
            n4.split_node_flag = true;
    }

    float ORBExtractor::ComputeSingleOrientation(const cv::Mat& image, cv::Point2f pt, const std::vector<int> &umax)
    {
        int m_01 = 0;
        int m_10 = 0;

        // 取出特征点对应该像素的指针
        const uchar* pcenter= &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

        // 当在园中轴线(v = 0)时
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; u++)
            m_10 += u * pcenter[u];
        
        // 获取从该行到下一行所需字节数
        int step = (int)image.step1();
        for (int v = 1; v <= HALF_PATCH_SIZE; v++)
        {
            int v_sum = 0;
            int d = umax[v];

            // 选定一个纵坐标 v，开始计算
            for (int u = -d; u <= d; u++)
            {
                int v_plus = pcenter[u + v * step];
                int v_minus = pcenter[u - v * step];
                v_sum = v_plus - v_minus;
                m_10 = u * (v_plus + v_minus);
            }
            m_01 += v * v_sum;
        }
        return cv::fastAtan2((float)m_01, (float)m_10);
    }

    const float factorPI = (float)(CV_PI / 180.f);
    void ORBExtractor::ComputeOrbDescriptor(cv::Mat& image, cv::KeyPoint& kpt, const cv::Point* pattern, uchar* desc)
    {
        float angle = (float)kpt.angle * factorPI;
        float a = (float)std::cos(angle);
        float b = (float)std::sin(angle);
        const uchar* pcenter = &image.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
        // 一个数据所占字节长度
        const int step = (int)image.step;
        
        #define GET_VALUE(idx) \
            pcenter[\
            cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
            cvRound(pattern[idx].x * a - pattern[idx].y * b)]
        
        for (int i = 0; i < 32; i++, pattern += 16)
        {
            int t0, t1, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;
            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;
            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;
            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;
            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;
            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;
            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;
            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            desc[i] = (uchar)val;
        }
        #undef GET_VALUE
    }
};