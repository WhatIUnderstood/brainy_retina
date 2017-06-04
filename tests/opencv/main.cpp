
#include "retinacuda.h"
#include "Utils/filesystemtools.h"

#include <unistd.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <ctime>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::system_clock Time;
//typedef std::chrono::duration_cast<std::chrono::milliseconds> Ms;
int main(int argc, char *argv[])
{
    //    for(unsigned int i=0; i<50; i++){
    //        std::cout<<std::to_string(i)<<" "<<excentricity(i)<<std::endl;
    //    }
    //    return 0;
    RetinaCuda::Parameters params;
    params.input_width = 640;
    params.input_height = 480;

    //cone sampling
    params.ph_fovea_pixels_by_cone = 1;
    params.ph_fovea_radius = 80;
    params.ph_max_pixels_by_cone = 5;
    params.ph_max_pixels_by_cone_radius = 180;

    //Ganglionar layer
    params.gc_fovea_radius = 80;
    params.gc_fovea_cones_by_cell = 1;
    params.gc_max_cones_by_cell = 5;
    params.gc_max_cones_by_cell_radius = 200;
    params.gc_fovea_inter_cells_distance = 1;
    params.gc_max_inter_cells_distance = 2;




    RetinaCuda retina;
    retina.initRetina(params);


    cv::cuda::GpuMat gpuMatSrc(640,480,CV_8UC1);
    cv::cuda::GpuMat gpuMatCones(640,480,CV_8UC1);
    cv::cuda::GpuMat gpuMatDst(640,480,CV_8UC1);
    cv::cuda::GpuMat gpuMatPrev(640,480,CV_8UC1);
    cv::cuda::GpuMat gpuMatDirectionSelectiveOuput(640,480,CV_8UC1);
    GpuBitArray2D gpuSparseArray;
    GpuBitArray2D gpuSparseDirectionSelectiveArray;

    cv::cuda::GpuMat gpudiscreteGCArray(640,480,CV_8UC1);

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

        TimePoint start = Time::now();
        gpuMatSrc.upload(cframe);

        TimePoint t_proc = Time::now();
        retina.applyPhotoreceptorSampling(gpuMatSrc,gpuMatCones);
        retina.applyMultiConvolve(gpuMatCones,gpuMatDst);
        retina.applySelectiveGC(gpuMatCones,gpuMatDirectionSelectiveOuput,gpuMatPrev);
        retina.sparse(gpuMatDst,16,gpuSparseArray);
        retina.sparse(gpuMatDirectionSelectiveOuput,8,gpuSparseDirectionSelectiveArray,0,64);

        TimePoint t_proc_done = Time::now();

        ///retina.discretise(gpuMatDst,8,gpudiscreteGCArray);

        cv::Mat frameConeRetina(gpuMatCones.rows,gpuMatCones.cols,CV_8UC1);
        gpuMatCones.download(frameConeRetina);

        //qDebug()<<"frameRetina"<<frameRetina.cols<<frameRetina.rows<<gpuMatDst.cols;
        cv::Mat frameRetina(gpuMatDst.rows,gpuMatDst.cols,CV_8UC1);
        gpuMatDst.download(frameRetina);

        cv::Mat frameSelectiveRetina(gpuMatDirectionSelectiveOuput.rows,gpuMatDirectionSelectiveOuput.cols,CV_8UC1);
        gpuMatDirectionSelectiveOuput.download(frameSelectiveRetina);
        TimePoint t_end = Time::now();

        // Holds the colormap version of the image:
        cv::Mat cv_cm_img0;
        // Apply the colormap:
        cv::applyColorMap(frameRetina, cv_cm_img0, cv::COLORMAP_JET);//COLORMAP_RAINBOW COLORMAP_JET

        cv::Mat cv_cm_selective;
        cv::applyColorMap(frameSelectiveRetina, cv_cm_selective, cv::COLORMAP_JET);

        //Discrete window
        HostBitArray2D discreteArray;
        gpuSparseArray.upload(discreteArray);
        cv::Mat cv_discrete(discreteArray.bytesHeight(),discreteArray.bytesWidth(),CV_8UC1,discreteArray.data());

        HostBitArray2D discreteSelectiveArray;
        gpuSparseDirectionSelectiveArray.upload(discreteSelectiveArray);
        cv::Mat cv_discrete_selective(discreteSelectiveArray.bytesHeight(),discreteSelectiveArray.bytesWidth(),CV_8UC1,discreteSelectiveArray.data());


        ///cv::Mat frameDiscrete(gpudiscreteGCArray.rows,gpudiscreteGCArray.cols,CV_8UC1);
        ///gpudiscreteGCArray.download(frameDiscrete);


        //gpuMatSrc.download(frameRetina);
        std::cout<<"frameRetina "<<frameRetina.cols<<frameRetina.rows<<std::endl;
        cv::imshow("MyVideo retina", cv_cm_img0); //show the frame in "MyVideo" window
        cv::imshow("MyVideo input", frame);
        cv::imshow("MyVideo directional", cv_cm_selective);
        cv::imshow("Cones output",frameConeRetina);

        cv::imshow("Sparse output",cv_discrete);
        cv::imshow("Sparse selective output",cv_discrete_selective);

        ///cv::imshow("discrete output",frameDiscrete);

        char key = cv::waitKey(30);

        std::cout<<"processing time"<<std::chrono::duration_cast<std::chrono::milliseconds>(t_proc_done-t_proc).count()<<std::endl;
        std::cout<<"all time"<<std::chrono::duration_cast<std::chrono::milliseconds>(t_end-start).count()<<std::endl;

        if(key == 'q') //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }

        static int i=0;
        i++;
        //*if(i%10 == 1)
            gpuMatPrev = gpuMatCones.clone();

    }

    return 42;
}
