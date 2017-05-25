
#include "retinacuda.h"
#include "Utils/filesystemtools.h"

#include <unistd.h>
#include <iostream>
#include <assert.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"



int main(int argc, char *argv[])
{
//    for(unsigned int i=0; i<50; i++){
//        std::cout<<std::to_string(i)<<" "<<excentricity(i)<<std::endl;
//    }
//    return 0;
    RetinaCuda::Parameters params;
    params.cells_height = 480;
    params.cells_width = 640;
    params.fovea_magnocellular_radius = 10;
    params.fovea_pixels_by_cell = 1;
    params.max_pixels_by_cell = 128;
    params.fovea_inter_cells_distance = 1;
    params.max_inter_cells_distance = 2;
    params.input_width = 640;
    params.input_height = 480;

    RetinaCuda retina;
    retina.initRetina(params);

    cv::cuda::GpuMat gpuMatSrc(640,480,CV_8UC1);
    cv::cuda::GpuMat gpuMatDst(640,480,CV_8UC1);

    cv::VideoCapture cap(0); // open the video file for reading

    if ( !cap.isOpened() )  // if not success, exit program
    {
        std::cout<< "Cannot open the video file" << std::endl;
        return -1;
    }

    //cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms
    //double fps = cap.get(cv::CAP_PROP_FPS); //get the frames per seconds of the video
    //qDebug() << "Frame per seconds : " << fps << endl;

    cv::namedWindow("MyVideo",cv::WINDOW_AUTOSIZE); //create a window called "MyVideo"
    //int i=0;

    while(true)
    {
        cv::Mat frame;


        bool bSuccess = cap.read(frame); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            std::cout<< "Cannot read the frame from video file" << std::endl;
            break;
        }

        cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);

        cv::Mat cframe(frame.rows,frame.cols,CV_8UC1,(char*)frame.data);
        gpuMatSrc.upload(cframe);

        retina.applyMultiConvolve(gpuMatSrc,gpuMatDst);


        //qDebug()<<"frameRetina"<<frameRetina.cols<<frameRetina.rows<<gpuMatDst.cols;
        cv::Mat frameRetina(gpuMatDst.rows,gpuMatDst.cols,CV_8UC1);
        gpuMatDst.download(frameRetina);

        // Holds the colormap version of the image:
        cv::Mat cv_cm_img0;
        // Apply the colormap:
        cv::applyColorMap(frameRetina, cv_cm_img0, cv::COLORMAP_JET);//COLORMAP_RAINBOW COLORMAP_JET

        //gpuMatSrc.download(frameRetina);
        std::cout<<"frameRetina "<<frameRetina.cols<<frameRetina.rows<<std::endl;
        cv::imshow("MyVideo retina", cv_cm_img0); //show the frame in "MyVideo" window
        cv::imshow("MyVideo input", frame);

        char key = cv::waitKey(30);
        std::cout<<"key"<<key<<std::endl;
        if(key == 'q') //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }

    }

    return 42;
}
