/*
xiaocong_fu@hotmail.com   

Hand gesture recognition.
Step 1, segmentation:
-Hand segmentation using skin color;
-Arm part above wrist is segmented and only pale is keeped.
Step 2, recognition:
-Hu moments is extracted as features of hand gestures;
-ANN is used to train features and do the recognition.

Hand gestures supported:
fist, 1-5, paper, lizaed, spock.

g++ gestRecog.cpp -o gestRecog `pkg-config --cflags --libs opencv`
./gestRecog
*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/core.hpp> 
#include <string>
#include <sys/time.h>
#include <opencv2/ml/ml.hpp> 

#define N 900
#define M1 7
#define M2 9

using namespace std;
using namespace cv;

//transfer to binary image
int cvSkinOtsu(Mat &src,Mat &dst);
//otsu, a threshold selection method from gray-level histograms
int cvThresholdOtsu(Mat &src, Mat &dst);
//select the biggest area to remove noise
void selectArea(Mat& src,Mat& dst,int thresholdValue, vector<Point>& selectImg_contour);
//calculate the max inscribed circle of hand
pair<Point, double> findMaxInscribedCircle(vector<Point> &contour, Mat &inputFrame);
//fill rotatedrect with white color
void fillMask(Mat &src, Point2f rect_points[]);
//draw min enclosing rectangle
void drawMinarearect(Mat &scr, Point2f rect_points[]);
//calculate new points to segment arm
inline void caculate_tandm(Mat &scr, RotatedRect &minRect, Point2f rect_points[], pair<Point, double> &cr1);

int main(int argc,char** argv)
{

    int res = 0;
    int nCamera = 0;
       
    Mat inputFrame;     //source image from camera
    int count_tmp = 0;
    char tempfile[100] = {'\0'};


    VideoCapture capture;
    cout<<"Indicate a camera number:"<<endl;
    cin>>nCamera;
    capture.open(nCamera);
    capture.read(inputFrame);
    Mat dst(inputFrame.size().height, inputFrame.size().width, CV_8UC1);
    Mat selectImg(inputFrame.size().height, inputFrame.size().width, CV_8UC1);

    vector<vector<Point> > mode_contours;
    vector<Point> selectImg_contour;

    RotatedRect minRect;  //min enclosing rectangle
    Point2f rect_points[4]; //min enclosing rectangle point
    
    struct timeval start, end;
    

    ////////////////////////////////////////////////////////////////
    //import training data
    
    float input[N][M1];    //train input
    float output[N][M2];   //train output

    
    ifstream fin("trainHandInputSeg.txt");    
    ifstream fout("trainHandOutputSeg.txt");

    
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M1; j++)
        {
            fin>>input[i][j];
        }
    }
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M2; j++)
        {
            fout>>output[i][j];
        }
    }
    fin.close();
    fout.close();     

    cout<<"Import succeeded"<<endl;
    
    
    ////////////////////////////////////////////////////////////////
    //create ANN and start training
    CvANN_MLP bp;   //create network
    CvANN_MLP_TrainParams params;   //set up parameters
    params.train_method = CvANN_MLP_TrainParams::BACKPROP;  //set up training method: back propagation
    params.bp_dw_scale = 0.1;   //weight update rate
    params.bp_moment_scale = 0.1;   //weight update impulse
    Mat dataInput(N, M1, CV_32FC1, input);     //create a matrice to store train input data 
    //cout << "trainingDataInput = " << trainingDataInput << endl;
    
    Mat dataOutput(N, M2, CV_32FC1, output);   //create a matrice to store train output data 
    //cout<<"trainingDataOutput"<<trainingDataOutput<<endl;
    
    Mat layerSizes = (Mat_<int>(1,3) <<M1, 10, M2);   //set up ANN level, 3 levels here, 7 nodes in input level, 10 nodes in hidden level and 6 nodes in output level
    //void CvANN_MLP::create(const Mat& layerSizes, int activateFunc=CvANN_MLP::SIGMOID_SYM, double fparam1=0; fparam2=0);
    bp.create(layerSizes, CvANN_MLP::SIGMOID_SYM);
    //int CvANN_MLP::train(const Mat& input, const Mat& output, const Mat& sampleWeights, const Mat&sampleIdx=Mat(), CvANN_MLP_trainParams, int flags=0);

    bp.train(dataInput, dataOutput, Mat(), Mat(), params);  
    cout<<"Train succeeded"<<endl;
    ////////////////////////////////////////////////////////////////////
    
   
    
    while(1)
    {
    	gettimeofday( &start, NULL);
    	int nFinger;
        capture.read(inputFrame);

        res = cvSkinOtsu(inputFrame, dst);

        if (res > 0)
        {

            selectArea(dst, selectImg, res, selectImg_contour);

            minRect = minAreaRect(Mat(selectImg_contour));
            minRect.points(rect_points);
            drawMinarearect(inputFrame, rect_points);
            pair<Point, double> cr1 = findMaxInscribedCircle(selectImg_contour, selectImg);
            circle(inputFrame, cr1.first, cr1.second, Scalar(0, 0, 255), 2, 8, 0);
            caculate_tandm(inputFrame, minRect, rect_points, cr1);
            Mat mask(inputFrame.size().height, inputFrame.size().width, CV_8UC1, Scalar(0));
            fillMask(mask, rect_points);
            imshow("mask", mask);
            Mat roi(inputFrame.size().height, inputFrame.size().width, CV_8UC1, Scalar(0));
            selectImg.copyTo(roi, mask);
            imshow("roi", roi);                 
           

            //calculate 'Hu moment' of binary image, as feature value of ANN
            Moments m = moments(roi, 1);    //Moments moments(InputArray array, bool binaryImage = false)
	        double hu[7];
	        HuMoments(m, hu);   
	        float testInput[7]= {0, 0, 0, 0, 0, 0, 0};
            for(int i = 0; i < 7; i++)
            {
                testInput[i] = hu[i];    
            }
                
            Mat testDataInput(1, 7, CV_32FC1, testInput);
      
            Mat testDataOutput;    
            bp.predict(testDataInput, testDataOutput);  
            //cout<<"test data output: "<<endl<<testDataOutput<<endl;
            
            
            double minval,maxval;
            Point ptmin,ptmax;
            Mat dataRow = testDataOutput.rowRange(0,1);  
            minMaxLoc(dataRow, &minval, &maxval, &ptmin, &ptmax);
            nFinger = ptmax.x;
            cout<<"Number of fingers"<<nFinger<<endl; 
            switch(nFinger)
            {
                case 0: putText(inputFrame, "Fist", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                        
                        imshow("src", inputFrame);

                        break;
                case 1: putText(inputFrame, "One", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                       
                        imshow("src", inputFrame);

                        break;
                case 2: putText(inputFrame, "Two", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                        
                        imshow("src", inputFrame);

                        break;
                case 3: putText(inputFrame, "Three", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                       
                        imshow("src", inputFrame);

                        break;
                case 4: putText(inputFrame, "Four", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                       
                        imshow("src", inputFrame);

                        break;
                case 5: putText(inputFrame, "Five", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                      
                        imshow("src", inputFrame);

                        break;
                case 6: putText(inputFrame, "Paper", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                      
                        imshow("src", inputFrame);

                        break;
                 case 7: putText(inputFrame, "Lizard", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                      
                        imshow("src", inputFrame);

                        break;
                 case 8: putText(inputFrame, "Spock", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                      
                        imshow("src", inputFrame);

                        break;
                default:putText(inputFrame, "Error", Point(40, 60), CV_FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,255), 3, 8, false);
                       
                        imshow("src", inputFrame);
                    
            }   
            
            imwrite("inputFrame.jpg", inputFrame);
            imwrite("roi.jpg", roi);
        }
       
    
        gettimeofday( &end, NULL );
        int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
        cout<<"time used in main: "<< timeuse <<" us"<<endl<<endl;
        waitKey(10);
  
    }
    return 0;
}

//transfer to binary image
int cvSkinOtsu(Mat &src,Mat &dst)
{	
	struct timeval start, end;
	gettimeofday( &start, NULL);
	assert(dst.channels() == 1 && src.channels() == 3);
	
    int height = src.size().height;
	int width = src.size().width;
	
   	Mat ycrcb(width, height, CV_8UC3);
   	//Mat cr(width, height, CV_8UC1);
   	vector<Mat> cr;
   	cvtColor(src,ycrcb,CV_BGR2YCrCb);
   	split(ycrcb,cr);
   	int res = cvThresholdOtsu(cr[1],cr[1]);
   	//copyImage(cr,dst);
   	cr[1].copyTo(dst);
   	gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    cout<<"time used in cvSkinOtsu: "<< timeuse <<" us"<<endl;
   	return res;
}

//otsu, a threshold selection method from gray-level histograms
int cvThresholdOtsu(Mat &src, Mat &dst)
{
	struct timeval start, end;
	gettimeofday( &start, NULL);
    int height = src.size().height;
	int width = src.size().width;

	float histogram[256] = {0};
	for (int i = 0; i < height; i++)
	{
		unsigned char* p = (unsigned char*)src.data + src.step*i;
		for (int j = 0; j < width; j++)
		{
		     histogram[*p++]++;
		}
	}
    int size = height * width;
    for (int i = 0; i < 256; i++)
    {
        histogram[i] = histogram[i] / size;
    }
    float avgValue = 0;
    for (int i = 0; i < 256; i++)
    {
        avgValue += i*histogram[i];
    }

    int thresholdValue;
    float maxVariance = 0;
    float w = 0, u = 0;
    for (int i = 0; i < 256; i++)
    {
        w += histogram[i];
        u += i*histogram[i];
        float t = avgValue*w - u;
        float variance = t*t / (w*(1-w));
		if (variance > maxVariance)
		{
			maxVariance = variance;
		   	thresholdValue = i;
		}
	}
	threshold(src,dst,thresholdValue,255,CV_THRESH_BINARY);
	gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    cout<<"time used in cvThresholdOtsu: "<< timeuse <<" us"<<endl;
	return thresholdValue;
}

//select the biggest area to remove noise
void selectArea(Mat& src,Mat& dst,int thresholdValue, vector<Point>& selectImg_contour)
{
	struct timeval start, end;
	gettimeofday( &start, NULL);
    assert(src.channels() == 1 && dst.channels() == 1);
    memset(dst.data,0,dst.size().width * dst.size().height * dst.channels());
    medianBlur(src,src,5);
    threshold(src,src,thresholdValue,255,THRESH_BINARY);
    vector< vector<Point> > contours;
    //vector<Point> maxcontour;
    vector<Vec4i> hierarchy;

    findContours(src,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
    double maxarea = 0,area = 0;
    double minarea = 100;

    int index = 0,n = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        area = contourArea(Mat(contours[i]));
        if (area > maxarea)
        {
            maxarea = area;
            selectImg_contour = contours[i];
            index = i;
        }
    }
    
        drawContours(dst,contours,index,Scalar(255),CV_FILLED,8,hierarchy);
        gettimeofday( &end, NULL );
    	int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    	cout<<"time used in selectArea: "<< timeuse <<" us"<<endl;
    

}

//calculate the max inscribed circle of hand
pair<Point, double> findMaxInscribedCircle(vector<Point> &contour, Mat &inputFrame)
{
	struct timeval start, end;
	gettimeofday( &start, NULL);
    pair<Point, double> c;
    double dist = 1;
    double maxdist = 1;
    vector<Point> simple_contour;
    int step = 3;
    int size = contour.size()/step;
    for(int i=0; i<size; i++)
    {
    	simple_contour.push_back(contour[step*i]);
    }
    for(int i = 0; i < inputFrame.cols; i+=10)
    {
        for(int j = 0; j < inputFrame.rows; j += 10)
        {
            dist = pointPolygonTest(simple_contour, Point(i,j), true);
            if(dist > maxdist)
            {
                maxdist = dist;
                c.first = Point(i,j);
            }
        }
    }
    c.second = maxdist;
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    cout<<"time used in findMaxInscribedCircle: "<< timeuse <<" us"<<endl;
    return c;
}

//fill rotatedrect with white color
void fillMask(Mat &src, Point2f rect_points[])
{
	struct timeval start, end;
	gettimeofday( &start, NULL);
	assert(src.channels() == 1);
	Point2f a, b;
	float length, diffx1, diffx2, diffy1, diffy2;
	diffx1 = rect_points[1].x-rect_points[0].x;
	diffx2 = rect_points[2].x-rect_points[3].x;
	diffy1 = rect_points[1].y-rect_points[0].y;
	diffy2 = rect_points[2].y-rect_points[3].y;
	length = sqrt(diffx1*diffx1+diffy1*diffy1);
	for(int i=0; i<length; i+=5)
	{
		a.x=rect_points[0].x+diffx1*i/length;
		a.y=rect_points[0].y+diffy1*i/length;
		b.x=rect_points[3].x+diffx2*i/length;
		b.y=rect_points[3].y+diffy2*i/length;
		line(src, a, b, Scalar(255),10);
	}
	gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    cout<<"time used in fillMask: "<< timeuse <<" us"<<endl;
}

void drawMinarearect(Mat &scr, Point2f rect_points[])
{
	struct timeval start, end;
	gettimeofday( &start, NULL);
	for( int i=0; i<4; i++)
    {
         circle(scr, rect_points[i], 2*i+1 ,Scalar(0, 255, 0), CV_FILLED);
         line(scr, rect_points[i], rect_points[(i+1)%4], Scalar(0,0,255), 2);
    }
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    cout<<"time used in drawMinarearect: "<< timeuse <<" us"<<endl;
}

//calculate new points to segment arm
inline void caculate_tandm(Mat &scr, RotatedRect &minRect, Point2f rect_points[], pair<Point, double> &cr1)
{
	struct timeval start, end;
	gettimeofday( &start, NULL);
	Point2f t, m;
    float d, dhand;
    float A, B, C;
    float Ksqu;
    if(rect_points[0].x == rect_points[1].x)
    {
    	d = cr1.first.y-rect_points[2].y;
        dhand = d+cr1.second;
        t.x = rect_points[1].x;
        t.y = rect_points[1].y+dhand;
        m.x = rect_points[2].x;
        m.y = rect_points[2].y+dhand;
        line(scr, t, m, Scalar(255,0,0),3);
        rect_points[0] = t;
        rect_points[3] = m;
    }
    else if(minRect.size.height<minRect.size.width)
    {

        A = rect_points[2].y-rect_points[3].y;
        B = rect_points[3].x-rect_points[2].x;
        C = rect_points[2].x*rect_points[3].y-rect_points[3].x*rect_points[2].y;
        d = abs((A*cr1.first.x+B*cr1.first.y+C)/(sqrt(A*A+B*B)));
        dhand = d+cr1.second;
        Ksqu = (rect_points[1].y-rect_points[2].y)/(rect_points[1].x-rect_points[2].x);
        Ksqu *= Ksqu;
        t.x = rect_points[2].x-dhand/sqrt(Ksqu+1);
        t.y = dhand*sqrt(Ksqu/(Ksqu+1))+rect_points[2].y;
        m.x = rect_points[3].x-dhand/sqrt(Ksqu+1);
        m.y = dhand*sqrt(Ksqu/(Ksqu+1))+rect_points[3].y;
        line(scr, t, m, Scalar(255,0,0),3);
        rect_points[1] = t;
        rect_points[0] = m;
    }
    else
    {
        A = rect_points[1].y-rect_points[2].y;
        B = rect_points[2].x-rect_points[1].x;
        C = rect_points[1].x*rect_points[2].y-rect_points[2].x*rect_points[1].y;
        d = abs((A*cr1.first.x+B*cr1.first.y+C)/(sqrt(A*A+B*B)));
        dhand = d+cr1.second;
        Ksqu = (rect_points[0].y-rect_points[1].y)/(rect_points[0].x-rect_points[1].x);
        Ksqu *= Ksqu;
        t.x = dhand/sqrt(Ksqu+1)+rect_points[1].x;
        t.y = dhand*sqrt(Ksqu/(Ksqu+1))+rect_points[1].y;
        m.x = dhand/sqrt(Ksqu+1)+rect_points[2].x;
        m.y = dhand*sqrt(Ksqu/(Ksqu+1))+rect_points[2].y;
        line(scr, t, m, Scalar(255,0,0),3);
        rect_points[0] = t;
        rect_points[3] = m;
    }
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    cout<<"time used in caculate_tandm: "<< timeuse <<" us"<<endl;
}


