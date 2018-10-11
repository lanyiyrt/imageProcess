#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
������˼·�ǣ�
1.��ͼƬ����Ԥ����
2.ʹ�û���任��ͼ���е���������������ֱ��
3.��ֱ����ֱ��֮��Ľ��㣬Ȼ����ȡֱ��֮��ļнǣ����ڽ�����90�ȵļнǽ��м�¼����Ҫ��¼����ֱ��ֱ������K bֵ�ͽ�����Ϣ��
4.�жϽ������ڵ������Լ�ֱ�ߵ�б�ʣ�Ȼ������������������
*/

#define WINDOW_NAME "����װ�·���Ե���Ч��ͼ��"
using namespace cv;
using namespace std;
typedef Vec<double, 4> VecLinePara;  //ֱ�߲���,k1 b1,k2,b2
typedef Vec<Point, 4> Vec4Point;

//������Ҫ��ȫ�ֱ���
Mat src;  
Mat grayImg;
Mat Eage;
Mat dst;

//Canny��Ե���ĸߵ���ֵ
int cannyHighThreshold;    //0-100
int cannyLowThreshold;     //0-100
//HoughLinesp�еĽ�����ֵ����С�߳��ȡ���С�߼��
int houghThreshold;
int houghMinLinLength;
int houghmMaxLineGap;

void on_Trackbar(int, void*);
//������ֱ�ߵĽ���
Point CrossPoint(const Vec4i  line1, const Vec4i   line2, VecLinePara &lineParameter);
Point CenterPoint(const Vec4i line1, const Vec4i line2);
float getAngelOfTwoVector(Point &pt1, Point &pt2, Point &c);
Point drawRectangle(Point pt, float K, float b);
int main()
{
	//��ʱ�Ȳ����ⲿ��
	/*
	//��Ҫ������ļ�д��TXT�ĵ��У�������������
	ofstream fout;
	fout.open("ͼƬԴ�ļ��б�.txt", std::ios::out | std::ios::app);
	//�ж��ļ��Ƿ�򿪳ɹ�
	if (!fout.is_open()){
		return 0;
	}
	//����ͼƬ������,�����ļ���д��ͼƬԭ�ļ�����λ����Ϣ
	int imgNum = 1;
	while (imgNum <= 12){
		fout << "F:/��Ŀ/��Ե���/Eage Detection/Eage Detection/����ͼ��/" << imgNum << ".jpg" << endl;
		imgNum++;
	}
	fout.close();
	*/

	//1.����ͼƬ
	src = imread("F:/��Ŀ/��Ե���/Eage Detection/Eage Detection/����ͼ��/23.jpg");
	if (src.empty()){
		cout << "ͼƬ����ʧ�ܣ������ԣ�" << endl;
		return -1;
	}

	int WIDTH = src.cols;
	int HEIGHT = src.rows;

	//2.��ͼ��ת���ɻҶ�ͼ��
	cvtColor(src, grayImg, CV_BGR2GRAY);
	imwrite("δ��ʴ.jpg", grayImg);

	//3.�ԻҶ�ͼƬ�����˲�

	int g_nStructElementSize = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));
	erode(grayImg, grayImg, element);
	imwrite("��ʴ.jpg", grayImg);
	GaussianBlur(grayImg, grayImg, Size(3, 3), 0);
	
	//������ʾ����
	namedWindow(WINDOW_NAME, 0);

	//�����������ռ�
	cannyLowThreshold = 20;
	createTrackbar("����ֵ100 ", WINDOW_NAME, &cannyLowThreshold, 100, on_Trackbar);
	cannyHighThreshold = 60;
	createTrackbar("����ֵ100 ", WINDOW_NAME, &cannyHighThreshold, 100, on_Trackbar);
	houghThreshold = 90;
	createTrackbar("��ֵ150 ", WINDOW_NAME, &houghThreshold, 150, on_Trackbar);
	houghMinLinLength = 120;
	createTrackbar("�߳�300 ", WINDOW_NAME, &houghMinLinLength, 300, on_Trackbar);
	houghmMaxLineGap = 2;
	createTrackbar("��϶10 ", WINDOW_NAME, &houghmMaxLineGap, 10, on_Trackbar);
	
	on_Trackbar(cannyLowThreshold, 0);

	/*
	dilate(Eage, Eage, element, Point(-1, -1), 3, 1, Scalar(1));
	namedWindow("���ͺ��ͼƬ", 0);
	imshow("���ͺ��ͼƬ", Eage);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Eage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Mat result = Mat::zeros(Eage.size(), CV_8U);
	drawContours(result, contours, -1, Scalar(255, 0, 255));
	namedWindow("��ȡ����ͼƬ", 0);
	imshow("��ȡ����ͼƬ", result);

	threshold(result, result, 0, 255, CV_THRESH_BINARY);
	*/

	waitKey(0);
}

void on_Trackbar(int, void*){
	//�����ʱ
	clock_t start, finish;
	start = clock();
	double totaltime;
	//4.ʹ��CANNY�Ա�Ե���м��
	Canny(grayImg, Eage, cannyLowThreshold, cannyLowThreshold * 3);
	namedWindow("��⵽�ı�Ե", 0);
	imshow("��⵽�ı�Ե", Eage);
	
	//5.�Ա�ԵͼƬ���и�˹�˲�
	GaussianBlur(Eage, Eage, Size(5, 5), 0);
	//namedWindow("��˹�˲�", 0);
	//imshow("��˹�˲�", Eage);

	//6.�Դ�������ͼƬ���ж�ֵ��
	threshold(Eage, Eage, 50, 255, CV_THRESH_BINARY);
	//namedWindow("��ֵ����ͼƬ", 0);
	//imshow("��ֵ����ͼƬ", Eage);

	//7.ʹ�û���任��ͼ���е�����������ֵ������ֱ��
	vector<Vec4i> lines;
	// ���ֱ�ߣ���СͶƱΪ90������������50����϶��С��10,�����������Ϊ150��5
	HoughLinesP(Eage, lines, 1, CV_PI / 180, houghThreshold, houghMinLinLength, houghmMaxLineGap);
	
	//�������дһ������ void process(const vector<Vec4i> lines);
	
	vector<Vec4i>::const_iterator line1Iterator = lines.begin(); 
	vector<Vec4i>::const_iterator line2Iterator;
	//vector<Point> points;
	Point point(0, 0);
	Point prePoint(0, 0);
	//����ֱ�ߵĲ���
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
		line(dst, pt1, pt2, Scalar(0, 255, 0), 4); //  �����������Ϊ2
		line(dst, pt3, pt4, Scalar(0, 0, 0), 2);
		
		++line1Iterator;
	}

	//���õ������ĵ㻭��ͼ����
	circle(dst, prePoint, 20, Scalar(0, 0, 255), -1, 8);
	cout << "�·�λ�õ�Բ��Ϊ��[" << prePoint.x << " , " << prePoint.y << "]" << endl;
 
	/*cout << "ͨ��Houghlinesp��⵽��ֱ������Ϊ��a = " << a << std::endl;
	printf("ͨ��ֱ�߼���õ��Ľ������Ϊ��b = %d ,����ߵĵ�Ϊ��d = %d \n", c,d);*/
	imwrite("Ч��ͼ.jpg", dst);
	imshow(WINDOW_NAME, dst);

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("�˳��������ʱ��Ϊ%f�� \n", totaltime);

}

/*�������ܣ�
�����Ǽ�������ֱ��֮��ļнǣ�����н����趨�ķ�Χ�ڣ������б�ʣ����㣬ʹ��drawRectangle���������ѿ��ӻ������������ǰ����ĵ�λ����������
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
			//���������ĵ㣻
			//������Σ�ȡƽ��ֵ
			Point point1 = drawRectangle(pt, k1, b1);
			Point point2 = drawRectangle(pt, k2, b2);
			pt.x = (point1.x + point2.x) / 2;
			pt.y = (point1.y + point2.y) / 2;
		}
	}
	return pt;
}

Point CrossPoint(const Vec4i  line1, const Vec4i   line2, VecLinePara &lineParameter)  //��������ֱ�ߵĽ��㡣ֱ��������������ʽ�ṩ��
{
	Point pt(0,0);          
	double k1, k2, b1, b2;
	if ((line1[0] == line1[2]) || (line2[0] == line2[2])){//�����һ��ֱ��б�ʲ�����
		return pt;
	}
	else     //���б��ʽ���̡�Ȼ����k1x + b1 = k2x + b2�����x�������y����
	{
		k1 = double(line1[3] - line1[1]) / (line1[2] - line1[0]);      b1 = double(line1[1] - k1*line1[0]);
		k2 = double(line2[3] - line2[1]) / (line2[2] - line2[0]);      b2 = double(line2[1] - k2*line2[0]);
		pt.x = (int)((b2 - b1) / (k1 - k2));  //���x
		pt.y = (int)(k1* pt.x + b1); //���y
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

//����pt1��pt2Ϊ����ֱ���ϵĵ㣬cΪ��������
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


//30����Ҫд�������
Point drawRectangle(Point pt, float K, float b){
	int halfWidth = src.cols / 2;
	int halfHeight = src.rows / 2;
	//�������߶�����Ϊ575.0�������ͱ�����һЩ�������
	float bagWidth = 575.0;
	float bagHeight = 575.0;

	Point left_up_Point, left_down_Point, right_up_Point, right_down_Point;

	/*��ͼ��ȷֳ��ĸ����ޣ��ж�һ�µ����Ǹ�����*/

	double theta = abs(atan(K));
	cout << "ʹ�õ�ֱ����X��ļн�Ϊ: " << atan(K) * 180.0 / CV_PI << endl;


	//1����һ����
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
