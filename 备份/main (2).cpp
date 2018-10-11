#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
基本的思路是：
1.对图片进行预处理
2.使用霍夫变换求图像中的所有满足条件的直线
3.求直线与直线之间的交点，然后求取直线之间的夹角，对于近似于90度的夹角进行记录，主要记录的是直线直线特征K b值和交点信息。
4.判断交点所在的像限以及直线的斜率，然后根据这两个特征求解
*/

#define WINDOW_NAME "【袋装衣服边缘检测效果图】"
using namespace cv;
using namespace std;
typedef Vec<double, 4> VecLinePara;  //直线参数,k1 b1,k2,b2
typedef Vec<Point, 4> Vec4Point;

//定义需要的全局变量
Mat src;  
Mat grayImg;
Mat Eage;
Mat dst;

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
float getAngelOfTwoVector(Point &pt1, Point &pt2, Point &c);
Point drawRectangle(Point pt, float K, float b);
int main()
{
	//暂时先不用这部分
	/*
	//将要读入的文件写如TXT文档中，便于批量操作
	ofstream fout;
	fout.open("图片源文件列表.txt", std::ios::out | std::ios::app);
	//判断文件是否打开成功
	if (!fout.is_open()){
		return 0;
	}
	//定义图片的数量,并想文件中写入图片原文件绝对位置信息
	int imgNum = 1;
	while (imgNum <= 12){
		fout << "F:/项目/边缘检测/Eage Detection/Eage Detection/测试图像/" << imgNum << ".jpg" << endl;
		imgNum++;
	}
	fout.close();
	*/

	//1.读入图片
	src = imread("F:/项目/边缘检测/Eage Detection/Eage Detection/测试图像/23.jpg");
	if (src.empty()){
		cout << "图片加载失败，请重试！" << endl;
		return -1;
	}

	int WIDTH = src.cols;
	int HEIGHT = src.rows;

	//2.将图像转换成灰度图像
	cvtColor(src, grayImg, CV_BGR2GRAY);
	imwrite("未腐蚀.jpg", grayImg);

	//3.对灰度图片进行滤波

	int g_nStructElementSize = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));
	erode(grayImg, grayImg, element);
	imwrite("腐蚀.jpg", grayImg);
	GaussianBlur(grayImg, grayImg, Size(3, 3), 0);
	
	//命名显示窗口
	namedWindow(WINDOW_NAME, 0);

	//创建滑动条空件
	cannyLowThreshold = 20;
	createTrackbar("低阈值100 ", WINDOW_NAME, &cannyLowThreshold, 100, on_Trackbar);
	cannyHighThreshold = 60;
	createTrackbar("高阈值100 ", WINDOW_NAME, &cannyHighThreshold, 100, on_Trackbar);
	houghThreshold = 90;
	createTrackbar("阈值150 ", WINDOW_NAME, &houghThreshold, 150, on_Trackbar);
	houghMinLinLength = 120;
	createTrackbar("线长300 ", WINDOW_NAME, &houghMinLinLength, 300, on_Trackbar);
	houghmMaxLineGap = 2;
	createTrackbar("间隙10 ", WINDOW_NAME, &houghmMaxLineGap, 10, on_Trackbar);
	
	on_Trackbar(cannyLowThreshold, 0);

	/*
	dilate(Eage, Eage, element, Point(-1, -1), 3, 1, Scalar(1));
	namedWindow("膨胀后的图片", 0);
	imshow("膨胀后的图片", Eage);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Eage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Mat result = Mat::zeros(Eage.size(), CV_8U);
	drawContours(result, contours, -1, Scalar(255, 0, 255));
	namedWindow("提取轮廓图片", 0);
	imshow("提取轮廓图片", result);

	threshold(result, result, 0, 255, CV_THRESH_BINARY);
	*/

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
	//namedWindow("高斯滤波", 0);
	//imshow("高斯滤波", Eage);

	//6.对处理过后的图片进行二值化
	threshold(Eage, Eage, 50, 255, CV_THRESH_BINARY);
	//namedWindow("二值化的图片", 0);
	//imshow("二值化的图片", Eage);

	//7.使用霍夫变换求图像中的所有满足阈值条件的直线
	vector<Vec4i> lines;
	// 检测直线，最小投票为90，线条不短于50，间隙不小于10,这里参数调整为150和5
	HoughLinesP(Eage, lines, 1, CV_PI / 180, houghThreshold, houghMinLinLength, houghmMaxLineGap);
	
	//这里可以写一个函数 void process(const vector<Vec4i> lines);
	
	vector<Vec4i>::const_iterator line1Iterator = lines.begin(); 
	vector<Vec4i>::const_iterator line2Iterator;
	//vector<Point> points;
	Point point(0, 0);
	Point prePoint(0, 0);
	//定义直线的参数
	double K, b;
	int a = 0, c = 0 ,d = 0;
	int width = Eage.cols;
	int heigh = Eage.rows;
	dst = src.clone();
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
		
		int x1 = (*line1Iterator)[0];
		int y1 = (*line1Iterator)[1];
		int x2 = (*line1Iterator)[2];
		int y2 = (*line1Iterator)[3];

		K = (double)(y2 - y1) / (x2 - x1);
		b = y1 - K * x1;

		int y_end = K * (Eage.cols - 1) + b;
		std::cout << "K = (y1 - y2) / (x1 -x2) = (" << y1 << " - " << y2 << ") / (" << x1 << " - " << y2 << ")" << std::endl;
		std::cout << "y = " << K << " x + " << b << std::endl;
		String text = "y = " + to_string(K) + " x + " + to_string(b);

		Point pt1((*line1Iterator)[0], (*line1Iterator)[1]);
		Point pt2((*line1Iterator)[2], (*line1Iterator)[3]);
		Point pt3(0, int(b));
		Point pt4(width - 1, y_end);

		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 2, 8);
		putText(dst, text, pt1, FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2);
		line(dst, pt1, pt2, Scalar(0, 255, 0), 4); //  线条宽度设置为2
		line(dst, pt3, pt4, Scalar(0, 0, 0), 2);
		
		++line1Iterator;
	}

	//将得到的中心点画在图像中
	circle(dst, prePoint, 20, Scalar(0, 0, 255), -1, 8);
	cout << "衣服位置的圆心为：[" << prePoint.x << " , " << prePoint.y << "]" << endl;
 
	/*cout << "通过Houghlinesp检测到的直线数量为：a = " << a << std::endl;
	printf("通过直线计算得到的交点个数为：b = %d ,在外边的点为：d = %d \n", c,d);*/
	imwrite("效果图.jpg", dst);
	imshow(WINDOW_NAME, dst);

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("此程序的运行时间为%f秒 \n", totaltime);

}

/*函数功能：
首先是计算两条直线之间的夹角，如果夹角在设定的范围内，则计算斜率，交点，使用drawRectangle（）函数把框子画出来，或者是把中心点位置求解出来。
*/
Point CenterPoint(const Vec4i line1, const Vec4i line2){
	Point pt(0, 0);
	double k1, k2, b1, b2;
	double theta1 = atan2(line1[3] - line1[1], line1[2] - line1[0]);
	double theta2 = atan2(line2[3] - line2[1], line2[2] - line2[0]);
	double theta = theta1 - theta2;

	if (theta > CV_PI)
		theta -= 2 * CV_PI;
	if (theta < -CV_PI)
		theta += 2 * CV_PI;
	theta = abs(theta * 180.0 / CV_PI);

	if (theta >= 80 && theta <= 100){
		VecLinePara lineParameter;
		pt = CrossPoint(line1, line2, lineParameter);
		if (pt != Point(0, 0)){
			k1 = lineParameter[0];
			b1 = lineParameter[1];
			k2 = lineParameter[2];
			b2 = lineParameter[3];
			//float K = abs(k1) > abs(k2) ? k1 : k2;
			//float b = K == k1 ? b1 : b2;
			//画框求中心点；
			//求解两次，取平均值
			Point point1 = drawRectangle(pt, k1, b1);
			Point point2 = drawRectangle(pt, k2, b2);
			pt.x = (point1.x + point2.x) / 2;
			pt.y = (point1.y + point2.y) / 2;
		}
	}
	return pt;
}

Point CrossPoint(const Vec4i  line1, const Vec4i   line2, VecLinePara &lineParameter)  //计算两条直线的交点。直线由整数向量形式提供。
{
	Point pt(0,0);          
	double k1, k2, b1, b2;
	if ((line1[0] == line1[2]) || (line2[0] == line2[2])){//如果第一条直线斜率不存在
		return pt;
	}
	else     //求出斜截式方程。然后让k1x + b1 = k2x + b2，解出x，再算出y即可
	{
		k1 = double(line1[3] - line1[1]) / (line1[2] - line1[0]);      b1 = double(line1[1] - k1*line1[0]);
		k2 = double(line2[3] - line2[1]) / (line2[2] - line2[0]);      b2 = double(line2[1] - k2*line2[0]);
		pt.x = (int)((b2 - b1) / (k1 - k2));  //算出x
		pt.y = (int)(k1* pt.x + b1); //算出y
		if (pt.x > 0 && pt.x < src.cols && pt.y > 0 && pt.y < src.rows){
			lineParameter[0] = k1;
			lineParameter[1] = b1;
			lineParameter[2] = k2;
			lineParameter[3] = b2;
			return pt;
		}
		else{
			pt = Point(0, 0);
			return pt;
		}
	}	
}

//其中pt1和pt2为两条直线上的点，c为公共交点
//float getangeloftwovector(point &pt1, point &pt2, point &c)
//{
//	float theta = atan2(pt1.x - c.x, pt1.y - c.y) - atan2(pt2.x - c.x, pt2.y - c.y);
//	if (theta > cv_pi)
//		theta -= 2 * cv_pi;
//	if (theta < -cv_pi)
//		theta += 2 * cv_pi;
//
//	theta = theta * 180.0 / cv_pi;
//	return theta;
//}


//30号主要写这个程序
Point drawRectangle(Point pt, float K, float b){
	int halfWidth = src.cols / 2;
	int halfHeight = src.rows / 2;
	//把两个边都设置为575.0，这样就避免了一些问题出现
	float bagWidth = 575.0;
	float bagHeight = 575.0;

	Point left_up_Point, left_down_Point, right_up_Point, right_down_Point;

	/*将图像等分成四个象限，判断一下点在那个象限*/

	double theta = abs(atan(K));
	cout << "使用的直线与X轴的夹角为: " << atan(K) * 180.0 / CV_PI << endl;


	//1，第一象限
	if (pt.x < halfWidth && pt.y < halfHeight){
		if (K < 0){ 
			left_up_Point = pt;

			left_down_Point.x = pt.x - bagHeight * cos(theta);
			left_down_Point.y = pt.y + bagHeight * sin(theta);
			cout << left_down_Point << endl;

			right_up_Point.x = pt.x + bagWidth * sin(theta);
			right_up_Point.y = pt.y + bagWidth * cos(theta);
			cout << right_up_Point << endl;

			right_down_Point.x = left_down_Point.x + bagWidth * sin(theta);
			right_down_Point.y = left_down_Point.y + bagWidth * cos(theta);
		}
		else{
			left_up_Point = pt;

			left_down_Point.x = pt.x - bagHeight * sin(theta);
			left_down_Point.y = pt.y + bagHeight * cos(theta);
			cout << left_down_Point << endl;

			right_up_Point.x = pt.x + bagWidth * cos(theta);
			right_up_Point.y = pt.y + bagWidth * sin(theta);
			cout << right_up_Point << endl;

			right_down_Point.x = left_down_Point.x + bagWidth * cos(theta);
			right_down_Point.y = left_down_Point.y + bagWidth * sin(theta);
		}
	}
	cv::line(dst, right_up_Point, right_down_Point, Scalar(255,0,255),3);
	cv::line(dst, right_down_Point, left_down_Point, Scalar(255, 0, 255), 3);
	cv::line(dst, left_down_Point, left_up_Point, Scalar(255, 0, 255), 3);
	cv::line(dst, left_up_Point, right_up_Point, Scalar(255, 0, 255), 3);
	pt.x = (left_up_Point.x + right_down_Point.x) / 2;
	pt.y = (left_up_Point.y + right_down_Point.y) / 2;
	return pt;
}
