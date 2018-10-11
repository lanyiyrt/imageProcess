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
������˼·�ǣ�
1.��ͼƬ����Ԥ����
2.ʹ�û���任��ͼ���е���������������ֱ��
3.��ֱ����ֱ��֮��Ľ��㣬Ȼ����ȡֱ��֮��ļнǣ����ڽ�����90�ȵļнǽ��м�¼����Ҫ��¼����ֱ��ֱ������K bֵ�ͽ�����Ϣ��
4.�жϽ������ڵ������Լ�ֱ�ߵ�б�ʣ�Ȼ������������������

*/

#define PARAMETER_CONTROL "����װ�·���Ե��������塿"
#define WINDOW_RESULT "����װ�·���Ե�����ʾ��塿" 


using namespace cv;
using namespace std;
typedef Vec<double, 4> VecLinePara;  //ֱ�߲���,k1 b1,k2,b2
typedef Vec<Point, 4> Vec4Point;

//������Ҫ��ȫ�ֱ���
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
Point DrawRectangle(Point pt, double K, double b);
Point CenterPointProcess(Point CenterPoint);
int main()
{
	//1.����ͼƬ
	src = imread("H:/git/imageProcess/Eage Detection/imgSource/7.jpg");
	if (src.empty()){
		std::cout << "ͼƬ����ʧ�ܣ������ԣ�" << endl;
		waitKey(0);
		
	}
	//2.ROI��ȡ
	imgROI = src(Rect(ROI_X, ROI_Y, ROI_width, ROI_heigth));

	//2.��ͼ��ת���ɻҶ�ͼ��
	cvtColor(imgROI, grayImg, CV_BGR2GRAY);

	//3.�ԻҶ�ͼƬ�����˲�
	int g_nStructElementSize = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));
	erode(grayImg, grayImg, element);
	cv::imwrite("��ʴ.jpg", grayImg);
	GaussianBlur(grayImg, grayImg, Size(3, 3), 0);
	
	//������ʾ����
	namedWindow(PARAMETER_CONTROL, 0);

	//�����������ռ�
	cannyLowThreshold = 25;
	createTrackbar("����ֵ100 ", PARAMETER_CONTROL, &cannyLowThreshold, 100, on_Trackbar);
	//cannyHighThreshold = 60;
	//createTrackbar("����ֵ100 ", PARAMETER_CONTROL, &cannyHighThreshold, 100, on_Trackbar);
	houghThreshold = 90;
	createTrackbar("��ֵ150 ", PARAMETER_CONTROL, &houghThreshold, 150, on_Trackbar);
	houghMinLinLength = 80;
	createTrackbar("�߳�300 ", PARAMETER_CONTROL, &houghMinLinLength, 300, on_Trackbar);
	houghmMaxLineGap = 2;
	createTrackbar("��϶10 ", PARAMETER_CONTROL, &houghmMaxLineGap, 10, on_Trackbar);
	
	on_Trackbar(cannyLowThreshold, 0);

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

	//6.�Դ�������ͼƬ���ж�ֵ��
	threshold(Eage, Eage, 50, 255, CV_THRESH_BINARY);

	//7.ʹ�û���任��ͼ���е�����������ֵ������ֱ��
	vector<Vec4i> lines;
	// ���ֱ�ߣ���СͶƱΪ90������������50����϶��С��10,�����������Ϊ150��5
	HoughLinesP(Eage, lines, 1, CV_PI / 180, houghThreshold, houghMinLinLength, houghmMaxLineGap);
	
	//�������дһ������ void process(const vector<Vec4i> lines);
	
	vector<Vec4i>::const_iterator line1Iterator = lines.begin(); 
	vector<Vec4i>::const_iterator line2Iterator;
	vector<Point> points;
	Point point(0, 0);
	Point prePoint(0, 0);
	//����ֱ�ߵĲ���
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
		
		//������⵽��ֱ��
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

	//���õ������ĵ㻭��ͼ����
	circle(dst, prePoint, 20, Scalar(0, 0, 255), -1, 8);





	//�Ż����ĵ������

	





	CenterPointProcess(prePoint);





	cout << "�·�λ�õ�Բ��Ϊ��[" << prePoint.x << " , " << prePoint.y << "]" << endl;
	//��ͼƬ�л�������ֱ����
	//cv::line(dst, Point(0, dst.rows / 2), Point(dst.cols, dst.rows / 2), Scalar(255, 255, 255), 2);
	//cv::line(dst, Point(dst.cols / 2, 0), Point(dst.cols / 2, dst.rows), Scalar(255, 255, 255), 2);
 
	//addWeighted(imgROI, 0, dst, 1, 0., imgROI);
	//imshow("src", src);

	imwrite("Ч��ͼ.jpg", dst);
	imshow(WINDOW_RESULT, dst);

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("�˳��������ʱ��Ϊ%f�� \n", totaltime);

}

/*�������ܣ�
�����Ǽ�������ֱ��֮��ļнǣ�����н����趨�ķ�Χ�ڣ������б�ʣ����㣬ʹ��DrawRectangle���������ѿ��ӻ������������ǰ����ĵ�λ����������
*/
Point CenterPoint(const Vec4i line1, const Vec4i line2){
	Point crossPoint(0, 0);
	Point centerPoint(0, 0);
	double k1, k2, b1, b2;
	//���㴫�������ֱ�ߵļн�
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

	//���ڼнǷ�Χ��80�� - 100��֮�������ֱ�߽�����⡣
	if (theta >= 80 && theta <= 100){
		VecLinePara lineParameter;
		crossPoint = CrossPoint(line1, line2, lineParameter);
		if (crossPoint != Point(0, 0)){
			k1 = lineParameter[0];
			b1 = lineParameter[1];
			k2 = lineParameter[2];
			b2 = lineParameter[3];

			//������Σ�ȡƽ��ֵ��������Ŀ���ǵ���ƫ��ĳһ���߲��������
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

//�������ֱ�ߵĽ��㣬���ؽ��㣬��ֱ�ߵĲ���K, b
Point CrossPoint(const Vec4i  line1, const Vec4i   line2, VecLinePara &lineParameter)  //��������ֱ�ߵĽ��㡣ֱ��������������ʽ�ṩ��
{
	Point pt(0,0);          
	double k1, k2, b1, b2;
	//���ֱ�ߵ�б�ʲ�����
	if ((line1[0] == line1[2]) || (line2[0] == line2[2])){
		return pt;
	}
	else     //���б��ʽ���̡�Ȼ����k1x + b1 = k2x + b2�����x�������y����
	{
		k1 = double(line1[3] - line1[1]) / (line1[2] - line1[0]);      b1 = double(line1[1] - k1*line1[0]);
		k2 = double(line2[3] - line2[1]) / (line2[2] - line2[0]);      b2 = double(line2[1] - k2*line2[0]);
		pt.x = (int)((b2 - b1) / (k1 - k2));  //���x
		pt.y = (int)(k1* pt.x + b1); //���y

		//���˵������ڱ߽���ߵ����
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

	//�������߶�����Ϊ575.0�������ͱ�����һЩ�������
	double bagWidth = 600;
	double bagHeight = 500;

	double lineLength = 575.0;

	Point left_up_Point(0, 0), left_down_Point(0, 0), right_up_Point(0, 0), right_down_Point(0, 0);
	Point centerPoint(0, 0);

	/*��ͼ��ȷֳ��ĸ����ޣ��ж�һ�µ����Ǹ�����*/

	double theta = abs(atan(K));
	cout << "ʹ�õ�ֱ����X��ļн�Ϊ: " << atan(K) * 180.0 / CV_PI << endl;

	double heightCos = bagHeight * cos(theta);
	double heightSin = bagHeight * sin(theta);
	double widthCos = bagWidth * cos(theta);
	double widthSin = bagWidth * sin(theta);

	//1����һ����
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

	//2.�ڶ�����
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

	//3.��������
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

	//4.��������
	 

	if (pt.x >= halfWidth && pt.y >= halfHeight){
		circle(dst, pt, 5, Scalar(255, 0, 0),5);
		right_down_Point = pt;
		if (K < 0){
			cout << "K ֵΪ��" << K << endl;
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
			cout << "K ֵΪ��" << K << endl;
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


	//��ʱ�Ȳ����ⲿ��
	
	//��Ҫ������ļ�д��TXT�ĵ��У�������������
	ofstream fout;
	fout.open("��������.txt", std::ios::out);
	//�ж��ļ��Ƿ�򿪳ɹ�
	if (!fout.is_open()){
	return 0;
	}

	Mat colorImg = imgROI.clone();
	Vec3b color_point = colorImg.at<Vec3b>(CenterPoint.y, CenterPoint.x);
	int colorB = color_point[0];
	int colorG = color_point[1];
	int colorR = color_point[2];

	fout << "���ĵ�������ǣ�" << color_point << endl;



	for (int i = CenterPoint.y; i < 720; i++){
		Vec3b color1 = colorImg.at<Vec3b>(i, CenterPoint.x);
		int color_b = color1[0];
		int color_g = color1[1];
		int color_r = color1[2];
		double result = sqrt(pow((colorB - color_b), 2) + pow((colorB - color_b), 2) + pow((colorB - color_b), 2));
		fout << "����λ��Ϊ��" << "[" << i << "," << CenterPoint.x << "]  ";
		fout << color1 ;
		fout << result << endl;
	}

	fout.close();

	//cout << color1 << endl;

	//�������֮����ɫ�ķ���
	//double result = sqrt(pow((colorB - color_b), 2) + pow((colorB - color_b), 2) + pow((colorB - color_b), 2));

	//cout << "������֮�����ɫ������" << result << endl;


	return CenterPoint;

}