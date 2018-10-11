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
Point CrossPoint(const Vec4i line1, const Vec4i line2);
float getAngelOfTwoVector(Point &pt1, Point &pt2, Point &c);
void drawRectangle(Point pt, float K, float b);
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
	cannyLowThreshold = 10;
	createTrackbar("低阈值100 ", WINDOW_NAME, &cannyLowThreshold, 100, on_Trackbar);
	cannyHighThreshold = 60;
	createTrackbar("高阈值100 ", WINDOW_NAME, &cannyHighThreshold, 100, on_Trackbar);
	houghThreshold = 100;
	createTrackbar("阈值150 ", WINDOW_NAME, &houghThreshold, 150, on_Trackbar);
	houghMinLinLength = 200;
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
	clock_t start, finish;
	start = clock();
	double totaltime;
	//4.使用CANNY对边缘进行检测
	Canny(grayImg, Eage, cannyLowThreshold, cannyHighThreshold);
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
	
	vector<Vec4i>::const_iterator line1Iterator = lines.begin(); 
	vector<Vec4i>::const_iterator line2;
	vector<Point> points;
	Point point;
	//定义直线的参数
	double K, b;
	int a = 0, c = 0;
	int width = Eage.cols;
	int heigh = Eage.rows;
	dst = src.clone();
	while (line1Iterator != lines.end())
	{
		line2 = line1Iterator + 1;
		while (line2 != lines.end()){
			point = CrossPoint(*line1Iterator, *line2);
			//在图片外面的交点就忽略不计
			if (point.x >= 0 && point.x < width && point.y >= 0 && point.y < heigh){
				points.push_back(point);
			}
			line2++;
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
		Point pt3(0, b);
		Point pt4(width - 1, y_end);

		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 2, 8);
		putText(dst, text, pt1, FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 2);
		line(dst, pt1, pt2, Scalar(0, 255, 0), 4); //  线条宽度设置为2
		line(dst, pt3, pt4, Scalar(0, 0, 0), 2);
		++line1Iterator;
		a++;
	}

	//将得到的点画在图像中
	vector<Point>::const_iterator pointIt = points.begin();
	while (pointIt != points.end()){
		circle(dst, *pointIt, 10, Scalar(0, 0, 255), 2);
		cout << *pointIt << endl;
		pointIt++;
		c++;
	}

	Point pt4(0, 0);
	Point pt5(1500, 1500);
	line(dst, pt4, pt5, Scalar(255, 255, 255), 4);

	cout << "通过Houghlinesp检测到的直线数量为：a = " << a << std::endl;
	printf("通过直线计算得到的交点个数为：b = %d \n", c);
	imwrite("效果图.jpg", dst);
	imshow(WINDOW_NAME, dst);

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("此程序的运行时间为%f秒 \n", totaltime);

}
Point CrossPoint(const Vec4i  line1, const Vec4i   line2)  //计算两条直线的交点。直线由整数向量形式提供。
{
	Point pt;          
	double k1, k2, b1, b2;
	if (line1[0] == line1[2]){//如果第一条直线斜率不存在
		pt.x = line1[0];
		pt.y = line2[1] == line2[3] ? line2[1] :
		double(line2[1] - line2[3])*(pt.x - line2[0]) / (line2[0] - line2[2]) + line2[1];
		
	}
	else if (line2[0] == line2[2]){//如果第二条直线斜率不存在
		pt.x = line2[0];
		pt.y = line1[1] == line1[3] ? line1[1] :
		double(line1[1] - line1[3])*(pt.x - line1[0]) / (line1[0] - line1[2]) + line1[1];
	}
	else     //求出斜截式方程。然后让k1x + b1 = k2x + b2，解出x，再算出y即可
	{
		k1 = double(line1[3] - line1[1]) / (line1[2] - line1[0]);      b1 = double(line1[1] - k1*line1[0]);
		k2 = double(line2[3] - line2[1]) / (line2[2] - line2[0]);      b2 = double(line2[1] - k2*line2[0]);
		pt.x = (b2 - b1) / (k1 - k2);  //算出x
		pt.y = k1* pt.x + b1; //算出y
		
		//计算两个直线的夹角 atan2(double y, double x);
		float theta1 = atan2(line1[3] - line1[1], line1[2] - line1[0]);
		float theta2 = atan2(line2[3] - line2[1], line2[2] - line2[0]);
		float theta = theta1 - theta2;
		theta1 = theta1 * (180.0 / CV_PI);
		theta2 = theta2 * (180.0 / CV_PI);

		cout << "两条直线各自的夹角为" << theta1 << "  " << theta2 << endl;

		if (theta > CV_PI)
			theta -= 2 * CV_PI;
		if (theta < -CV_PI)
			theta += 2 * CV_PI;
		theta = abs(theta * 180.0 / CV_PI);

		//符合条件的两条直线被认为形成了一个直角，该直角被认为是衣物的一角
		if (theta >= 80 && theta <= 100){
			float K = abs(k1) > abs(k2) ? k1 : k2;
			float b = K == k1 ? b1 : b2;
			drawRectangle(pt, K, b);
		}


		cout << "该直线与其他直线的夹角为：" << theta << endl;
	}
	return pt;
}

//其中pt1和pt2为两条直线上的点，c为公共交点
float getAngelOfTwoVector(Point &pt1, Point &pt2, Point &c)
{
	float theta = atan2(pt1.x - c.x, pt1.y - c.y) - atan2(pt2.x - c.x, pt2.y - c.y);
	if (theta > CV_PI)
		theta -= 2 * CV_PI;
	if (theta < -CV_PI)
		theta += 2 * CV_PI;

	theta = theta * 180.0 / CV_PI;
	return theta;
}

void drawRectangle(Point pt, float K, float b){
	int halfWidth = src.cols / 2;
	int halfHeight = src.rows / 2;
	float bagWidth = 510.0;
	float bagHeight = 700.0;
	Point left_up_Point, left_down_Point, right_up_Point, right_down_Point;

	cout << pt << " halfw: " << halfWidth << " halfHeight：" << halfHeight << endl;

	//将图像等分成四个象限，判断一下点在那个象限
	if (pt.x < halfWidth && pt.y < halfHeight){
		if (K < 0){ 
			left_up_Point = pt;
			float theta = abs(atan(K));

			cout << theta << endl;
			left_down_Point.x = pt.x - bagHeight * cos(theta);
			left_down_Point.y = pt.y + bagHeight * sin(theta);
			cout << left_down_Point << endl;

			right_up_Point.x = pt.x + bagWidth * sin(theta);
			right_up_Point.y = pt.y + bagWidth * cos(theta);
			cout << right_up_Point << endl;

			right_down_Point.x = left_down_Point.x + bagWidth * sin(theta);
			right_down_Point.y = left_down_Point.y + bagWidth * cos(theta);
		}
	}

	line(dst, right_up_Point, right_down_Point, Scalar(255,0,255),3);
	line(dst, right_down_Point, left_down_Point, Scalar(255, 0, 255), 3);
	line(dst, left_down_Point, left_up_Point, Scalar(255, 0, 255), 3);
	line(dst, left_up_Point, right_up_Point, Scalar(255, 0, 255), 3);
}
