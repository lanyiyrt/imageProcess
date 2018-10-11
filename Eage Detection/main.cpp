#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <cmath>

/*
基本的思路是：
1.对图片进行预处理
2.使用霍夫变换求图像中的所有满足条件的直线
3.求直线与直线之间的交点，然后求取直线之间的夹角，对于近似于90度的夹角进行记录，主要记录的是直线直线特征K b值和交点信息。
4.判断交点所在的像限以及直线的斜率，然后根据这两个特征求解

*/

#define PARAMETER_CONTROL "【袋装衣服边缘检测控制面板】"
#define WINDOW_RESULT "【袋装衣服边缘检测显示面板】" 


using namespace cv;
using namespace std;
typedef Vec<double, 4> VecLinePara;  //直线参数,k1 b1,k2,b2
typedef Vec<Point, 4> Vec4Point;

//定义需要的全局变量
Mat src;  
Mat imgROI;
Mat grayImg;
Mat Eage;
Mat dst;

//ROIPara
int ROI_X = 400;
int ROI_Y = 250;
int ROI_width = 1200;
int ROI_heigth = 720;

//Canny边缘检测的高低阈值
int cannyHighThreshold;    //0-100
int cannyLowThreshold;     //0-100
//HoughLinesp中的交点阈值、最小线长度、最小线间距
int houghThreshold;
int houghMinLinLength;
int houghmMaxLineGap;

void on_Trackbar(int, void*);
//求两条直线的交点
Point CrossPoint(const Vec4i  line1, const Vec4i   line2, VecLinePara &lineParameter);
Point CenterPoint(const Vec4i line1, const Vec4i line2);
Point DrawRectangle(Point pt, double K, double b);
Point CenterPointProcess(Point CenterPoint);
int main()
{
	//1.读入图片
	src = imread("H:/git/imageProcess/Eage Detection/imgSource/7.jpg");
	if (src.empty()){
		std::cout << "图片加载失败，请重试！" << endl;
		waitKey(0);
		
	}
	//2.ROI提取
	imgROI = src(Rect(ROI_X, ROI_Y, ROI_width, ROI_heigth));

	//2.将图像转换成灰度图像
	cvtColor(imgROI, grayImg, CV_BGR2GRAY);

	//3.对灰度图片进行滤波
	int g_nStructElementSize = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));
	erode(grayImg, grayImg, element);
	cv::imwrite("腐蚀.jpg", grayImg);
	GaussianBlur(grayImg, grayImg, Size(3, 3), 0);
	
	//命名显示窗口
	namedWindow(PARAMETER_CONTROL, 0);

	//创建滑动条空件
	cannyLowThreshold = 25;
	createTrackbar("低阈值100 ", PARAMETER_CONTROL, &cannyLowThreshold, 100, on_Trackbar);
	//cannyHighThreshold = 60;
	//createTrackbar("高阈值100 ", PARAMETER_CONTROL, &cannyHighThreshold, 100, on_Trackbar);
	houghThreshold = 90;
	createTrackbar("阈值150 ", PARAMETER_CONTROL, &houghThreshold, 150, on_Trackbar);
	houghMinLinLength = 80;
	createTrackbar("线长300 ", PARAMETER_CONTROL, &houghMinLinLength, 300, on_Trackbar);
	houghmMaxLineGap = 2;
	createTrackbar("间隙10 ", PARAMETER_CONTROL, &houghmMaxLineGap, 10, on_Trackbar);
	
	on_Trackbar(cannyLowThreshold, 0);

	waitKey(0);
}

void on_Trackbar(int, void*){
	//程序计时
	clock_t start, finish;
	start = clock();
	double totaltime;
	//4.使用CANNY对边缘进行检测
	Canny(grayImg, Eage, cannyLowThreshold, cannyLowThreshold * 3);
	namedWindow("检测到的边缘", 0);
	imshow("检测到的边缘", Eage);
	
	//5.对边缘图片进行高斯滤波
	GaussianBlur(Eage, Eage, Size(5, 5), 0);

	//6.对处理过后的图片进行二值化
	threshold(Eage, Eage, 50, 255, CV_THRESH_BINARY);

	//7.使用霍夫变换求图像中的所有满足阈值条件的直线
	vector<Vec4i> lines;
	// 检测直线，最小投票为90，线条不短于50，间隙不小于10,这里参数调整为150和5
	HoughLinesP(Eage, lines, 1, CV_PI / 180, houghThreshold, houghMinLinLength, houghmMaxLineGap);
	
	//这里可以写一个函数 void process(const vector<Vec4i> lines);
	
	vector<Vec4i>::const_iterator line1Iterator = lines.begin(); 
	vector<Vec4i>::const_iterator line2Iterator;
	vector<Point> points;
	Point point(0, 0);
	Point prePoint(0, 0);
	//定义直线的参数
	double K, b;
	int width = Eage.cols;
	int heigh = Eage.rows;
	dst = imgROI.clone();
	while (line1Iterator != lines.end())
	{
		line2Iterator = line1Iterator + 1;
		while (line2Iterator != lines.end()){
			point = CenterPoint(*line1Iterator, *line2Iterator);
			if (point != Point(0, 0)){
				if (prePoint != Point(0, 0)){
					prePoint.x = (prePoint.x + point.x) / 2;
					prePoint.y = (prePoint.y + point.y) / 2;
				}
				else{
					prePoint = point;
				}
			}
			line2Iterator++;
		}
		
		//画出检测到的直线
		int x1 = (*line1Iterator)[0];
		int y1 = (*line1Iterator)[1];
		int x2 = (*line1Iterator)[2];
		int y2 = (*line1Iterator)[3];

		K = (double)(y2 - y1) / (x2 - x1);
		b = y1 - K * x1;
		int y_end = K * (Eage.cols - 1) + b;

		Point pt1(x1, y1);
		Point pt2(x2, y2);
		Point pt3(0, int(b));
		Point pt4(width - 1, y_end);
		
		cv::line(dst, pt3, pt4, Scalar(0, 255, 0), 2);
		
		++line1Iterator;
	}

	//将得到的中心点画在图像中
	circle(dst, prePoint, 20, Scalar(0, 0, 255), -1, 8);





	//优化中心点的坐标

	





	CenterPointProcess(prePoint);





	cout << "衣服位置的圆心为：[" << prePoint.x << " , " << prePoint.y << "]" << endl;
	//在图片中画两个垂直的线
	//cv::line(dst, Point(0, dst.rows / 2), Point(dst.cols, dst.rows / 2), Scalar(255, 255, 255), 2);
	//cv::line(dst, Point(dst.cols / 2, 0), Point(dst.cols / 2, dst.rows), Scalar(255, 255, 255), 2);
 
	//addWeighted(imgROI, 0, dst, 1, 0., imgROI);
	//imshow("src", src);

	imwrite("效果图.jpg", dst);
	imshow(WINDOW_RESULT, dst);

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("此程序的运行时间为%f秒 \n", totaltime);

}

/*函数功能：
首先是计算两条直线之间的夹角，如果夹角在设定的范围内，则计算斜率，交点，使用DrawRectangle（）函数把框子画出来，或者是把中心点位置求解出来。
*/
Point CenterPoint(const Vec4i line1, const Vec4i line2){
	Point crossPoint(0, 0);
	Point centerPoint(0, 0);
	double k1, k2, b1, b2;
	//计算传入的两条直线的夹角
	double theta1 = atan2(line1[3] - line1[1], line1[2] - line1[0]);
	double theta2 = atan2(line2[3] - line2[1], line2[2] - line2[0]);
	double theta = theta1 - theta2;

	if (theta > CV_PI){
		theta -= 2 * CV_PI;
	}
	else if (theta < -CV_PI){
		theta += 2 * CV_PI;
	}
	theta = abs(theta * 180.0 / CV_PI);

	//对于夹角范围在80° - 100°之间的两条直线进行求解。
	if (theta >= 80 && theta <= 100){
		VecLinePara lineParameter;
		crossPoint = CrossPoint(line1, line2, lineParameter);
		if (crossPoint != Point(0, 0)){
			k1 = lineParameter[0];
			b1 = lineParameter[1];
			k2 = lineParameter[2];
			b2 = lineParameter[3];

			//求解两次，取平均值，这样的目的是抵消偏向某一条边产生的误差
			Point point1 = DrawRectangle(crossPoint, k1, b1);
			Point point2 = DrawRectangle(crossPoint, k2, b2);
			if (point1 != Point(0, 0) && point2 != Point(0, 0)){
				centerPoint.x = (point1.x + point2.x) / 2;
				centerPoint.y = (point1.y + point2.y) / 2;
			}
		}
	}
	return centerPoint;
}

//求解两条直线的交点，返回交点，和直线的参数K, b
Point CrossPoint(const Vec4i  line1, const Vec4i   line2, VecLinePara &lineParameter)  //计算两条直线的交点。直线由整数向量形式提供。
{
	Point pt(0,0);          
	double k1, k2, b1, b2;
	//如果直线的斜率不存在
	if ((line1[0] == line1[2]) || (line2[0] == line2[2])){
		return pt;
	}
	else     //求出斜截式方程。然后让k1x + b1 = k2x + b2，解出x，再算出y即可
	{
		k1 = double(line1[3] - line1[1]) / (line1[2] - line1[0]);      b1 = double(line1[1] - k1*line1[0]);
		k2 = double(line2[3] - line2[1]) / (line2[2] - line2[0]);      b2 = double(line2[1] - k2*line2[0]);
		pt.x = (int)((b2 - b1) / (k1 - k2));  //算出x
		pt.y = (int)(k1* pt.x + b1); //算出y

		//过滤掉交点在边界外边的情况
		if (pt.x > 0 && pt.x < src.cols && pt.y > 0 && pt.y < src.rows){
			lineParameter[0] = k1;
			lineParameter[1] = b1;
			lineParameter[2] = k2;
			lineParameter[3] = b2;
		}
		else{
			pt = Point(0, 0);
		}
	} 
	return pt;
}

Point DrawRectangle(Point pt, double K, double b){
	int halfWidth = imgROI.cols / 2;
	int halfHeight = imgROI.rows / 2;

	//把两个边都设置为575.0，这样就避免了一些问题出现
	double bagWidth = 600;
	double bagHeight = 500;

	double lineLength = 575.0;

	Point left_up_Point(0, 0), left_down_Point(0, 0), right_up_Point(0, 0), right_down_Point(0, 0);
	Point centerPoint(0, 0);

	/*将图像等分成四个象限，判断一下点在那个象限*/

	double theta = abs(atan(K));
	cout << "使用的直线与X轴的夹角为: " << atan(K) * 180.0 / CV_PI << endl;

	double heightCos = bagHeight * cos(theta);
	double heightSin = bagHeight * sin(theta);
	double widthCos = bagWidth * cos(theta);
	double widthSin = bagWidth * sin(theta);

	//1，第一象限
	if (pt.x < halfWidth && pt.y < halfHeight){
		left_up_Point = pt;
		if (K < 0){ 
			if (theta > CV_PI / 4){

				left_down_Point.x = left_up_Point.x + (K < 0 ? -heightCos : +heightCos);
				left_down_Point.y = left_up_Point.y + heightSin;

				right_up_Point.x = left_up_Point.x + widthSin;
				right_up_Point.y = left_up_Point.y + widthCos;

				right_down_Point.x = left_down_Point.x + widthSin;
				right_down_Point.y = left_down_Point.y + widthCos;

				left_down_Point.x = left_up_Point.x - heightCos;
				left_down_Point.y = left_up_Point.y + heightSin;

				right_up_Point.x = left_up_Point.x + widthSin;
				right_up_Point.y = left_up_Point.y + widthCos;

				right_down_Point.x = left_down_Point.x + widthSin;
				right_down_Point.y = left_down_Point.y + widthCos;
			}
			else{
				left_down_Point.x = left_up_Point.x + heightSin;
				left_down_Point.y = left_up_Point.y + heightCos;
				cout << left_down_Point << endl;

				right_up_Point.x = left_up_Point.x + widthCos;
				right_up_Point.y = left_up_Point.y - widthSin;
				cout << right_up_Point << endl;

				right_down_Point.x = left_down_Point.x + widthCos;
				right_down_Point.y = left_down_Point.y - widthSin;
			}
		}
		else{
			if (theta < CV_PI / 4){
				left_down_Point.x = left_up_Point.x - heightSin;
				left_down_Point.y = left_up_Point.y + heightCos;

				right_up_Point.x = left_up_Point.x + widthCos;
				right_up_Point.y = left_up_Point.y + widthSin;

				right_down_Point.x = left_down_Point.x + widthCos;
				right_down_Point.y = left_down_Point.y + widthSin;
			}
			else{
				left_down_Point.x = left_up_Point.x + heightCos;
				left_down_Point.y = left_up_Point.y + heightSin;

				right_up_Point.x = left_up_Point.x + widthSin;
				right_up_Point.y = left_up_Point.y - widthCos;

				right_down_Point.x = left_down_Point.x + widthSin;
				right_down_Point.y = left_down_Point.y - widthCos;
			}
		}
	}

	//2.第二象限
	if (pt.x >= halfWidth && pt.y < halfHeight){
		right_up_Point = pt;
		if (K < 0){
			if (theta > CV_PI / 4){
				right_down_Point.x = right_up_Point.x - heightCos;
				right_down_Point.y = right_up_Point.y + heightSin;

				left_up_Point.x = right_up_Point.x - widthSin;
				left_up_Point.y = right_up_Point.y - widthCos;

				left_down_Point.x = right_down_Point.x - widthSin;
				left_down_Point.y = right_down_Point.y - widthCos;
			}
			else{
				right_down_Point.x = right_up_Point.x + heightSin;
				right_down_Point.y = right_up_Point.y + heightCos;

				left_up_Point.x = right_up_Point.x - widthCos;
				left_up_Point.y = right_up_Point.y + widthSin;

				left_down_Point.x = right_down_Point.x - widthCos;
				left_down_Point.y = right_down_Point.y + widthSin;
			}
		}
		else{
			if (theta < CV_PI / 4){
				right_down_Point.x = right_up_Point.x - heightSin;
				right_down_Point.y = right_up_Point.y + heightCos;

				left_up_Point.x = right_up_Point.x - widthCos;
				left_up_Point.y = right_up_Point.y - widthSin;

				left_down_Point.x = right_down_Point.x - widthCos;
				left_down_Point.y = right_down_Point.y - widthSin;
			}
			else{
				right_down_Point.x = right_up_Point.x + heightCos;
				right_down_Point.y = right_up_Point.y + heightSin;

				left_up_Point.x = right_up_Point.x - widthSin;
				left_up_Point.y = right_up_Point.y + widthCos;

				left_down_Point.x = right_down_Point.x - widthSin;
				left_down_Point.y = right_down_Point.y + widthCos;
			}
		}

	}

	//3.第三象限
	if (pt.x < halfWidth && pt.y >= halfHeight){
		left_down_Point = pt;
		if (K < 0){
			if (theta > CV_PI / 4){
				left_up_Point.x = left_down_Point.x + heightCos;
				left_up_Point.y = left_down_Point.y - heightSin;

				right_down_Point.x = left_down_Point.x + widthSin;
				right_down_Point.y = left_down_Point.y + widthCos;

				right_up_Point.x = left_up_Point.x + widthSin;
				right_up_Point.y = left_up_Point.y + widthCos;
			}
			else{
				left_up_Point.x = left_down_Point.x - heightSin;
				left_up_Point.y = left_down_Point.y - heightCos;

				right_down_Point.x = left_down_Point.x + widthCos;
				right_down_Point.y = left_down_Point.y - widthSin;

				right_up_Point.x = left_up_Point.x + widthCos;
				right_up_Point.y = left_up_Point.y - widthSin;
			}
		}
		else{
			if (theta < CV_PI / 4){
				left_up_Point.x = left_down_Point.x + heightSin;
				left_up_Point.y = left_down_Point.y - heightCos;

				right_down_Point.x = left_down_Point.x + widthCos;
				right_down_Point.y = left_down_Point.y + widthSin;

				right_up_Point.x = left_up_Point.x + widthCos;
				right_up_Point.y = left_up_Point.y + widthSin;
			}
			else{
				left_up_Point.x = left_down_Point.x - heightCos;
				left_up_Point.y = left_down_Point.y - heightSin;

				right_down_Point.x = left_down_Point.x + widthSin;
				right_down_Point.y = left_down_Point.y - widthCos;

				right_up_Point.x = left_up_Point.x + widthSin;
				right_up_Point.y = left_up_Point.y - widthCos;
			}
		}
	}

	//4.第四象限
	 

	if (pt.x >= halfWidth && pt.y >= halfHeight){
		circle(dst, pt, 5, Scalar(255, 0, 0),5);
		right_down_Point = pt;
		if (K < 0){
			cout << "K 值为：" << K << endl;
			if (theta > CV_PI / 4){
				right_up_Point.x = right_down_Point.x + heightCos;
				right_up_Point.y = right_down_Point.y - heightSin;

				left_down_Point.x = right_down_Point.x - widthSin;
				left_down_Point.y = right_down_Point.y - widthCos;

				left_up_Point.x = right_up_Point.x - widthSin;
				left_up_Point.y = right_up_Point.y - widthCos;
			}
			else{
				right_up_Point.x = right_down_Point.x - heightSin;
				right_up_Point.y = right_down_Point.y - heightCos;

				left_down_Point.x = right_down_Point.x - widthCos;
				left_down_Point.y = right_down_Point.y + widthSin;

				left_up_Point.x = right_up_Point.x - widthCos;
				left_up_Point.y = right_up_Point.y + widthSin;
				//cout << "[" << right_down_Point.x << "," << right_down_Point.y << "]" << endl;
			}
		}
		else{
			cout << "K 值为：" << K << endl;
			if (theta < CV_PI / 4){
				right_up_Point.x = right_down_Point.x + heightSin;
				right_up_Point.y = right_down_Point.y - heightCos;

				left_down_Point.x = right_down_Point.x - widthCos;
				left_down_Point.y = right_down_Point.y - widthSin;

				left_up_Point.x = right_up_Point.x - widthCos;
				left_up_Point.y = right_up_Point.y - widthSin;
			}
			else{
				right_up_Point.x = right_down_Point.x - heightCos;
				right_up_Point.y = right_down_Point.y - heightSin;

				left_down_Point.x = right_down_Point.x - widthSin;
				left_down_Point.y = right_down_Point.y + widthCos;

				left_up_Point.x = right_up_Point.x - widthSin;
				left_up_Point.y = right_up_Point.y + widthCos;
			}
		}

	}

	if (left_up_Point != Point(0, 0) && left_down_Point != Point(0, 0) && right_up_Point != Point(0, 0) && right_down_Point != Point(0, 0)){
		
		cv::line(dst, right_up_Point, right_down_Point, Scalar(255,0,255),3);
		cv::line(dst, right_down_Point, left_down_Point, Scalar(255, 0, 255), 3);
		cv::line(dst, left_down_Point, left_up_Point, Scalar(255, 0, 255), 3);
		cv::line(dst, left_up_Point, right_up_Point, Scalar(255, 0, 255), 3);
		centerPoint.x = (left_up_Point.x + right_down_Point.x) / 2;
		centerPoint.y = (left_up_Point.y + right_down_Point.y) / 2;
	}
	
	return centerPoint;
}


//zheyi
Point CenterPointProcess(Point CenterPoint){


	//暂时先不用这部分
	
	//将要读入的文件写如TXT文档中，便于批量操作
	ofstream fout;
	fout.open("均方差结果.txt", std::ios::out);
	//判断文件是否打开成功
	if (!fout.is_open()){
	return 0;
	}

	Mat colorImg = imgROI.clone();
	Vec3b color_point = colorImg.at<Vec3b>(CenterPoint.y, CenterPoint.x);
	int colorB = color_point[0];
	int colorG = color_point[1];
	int colorR = color_point[2];

	fout << "中心点的像素是：" << color_point << endl;



	for (int i = CenterPoint.y; i < 720; i++){
		Vec3b color1 = colorImg.at<Vec3b>(i, CenterPoint.x);
		int color_b = color1[0];
		int color_g = color1[1];
		int color_r = color1[2];
		double result = sqrt(pow((colorB - color_b), 2) + pow((colorB - color_b), 2) + pow((colorB - color_b), 2));
		fout << "坐标位置为：" << "[" << i << "," << CenterPoint.x << "]  ";
		fout << color1 ;
		fout << result << endl;
	}

	fout.close();

	//cout << color1 << endl;

	//求解两点之间颜色的方差
	//double result = sqrt(pow((colorB - color_b), 2) + pow((colorB - color_b), 2) + pow((colorB - color_b), 2));

	//cout << "两个点之间的颜色均方差" << result << endl;


	return CenterPoint;

}