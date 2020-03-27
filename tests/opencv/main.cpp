
#include "retinacuda.h"
#include "Utils/filesystemtools.h"

#include <unistd.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <iostream>

//#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"
//#include "opencv2/videoio/videoio_c.h"

#include "CLI/CLI.hpp"
#include "utils.h"

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::system_clock Time;
//typedef std::chrono::duration_cast<std::chrono::milliseconds> Ms;
int main(int argc, char *argv[])
{
    CLI::App app{"Retina tool using opencv grabber. It support webcams and video files"};

    // Define options
    std::string input_param;
    bool grayscale = false;
    app.add_option("-i,--input", input_param, "Input");
    app.add_option("--grayscale", grayscale, "");

    CLI11_PARSE(app, argc, argv);

    RetinaCuda::Parameters params;
    params.input_width = 3840;  //1280 640;
    params.input_height = 2160; //720 480;

    //cone sampling
    params.ph_fovea_pixels_by_cone = 1;
    params.ph_fovea_radius = 112;              //5° 225
    params.ph_max_pixels_by_cone = 14;         //
    params.ph_max_pixels_by_cone_radius = 150; //180 240
    params.ph_S_cone_ratio = 5.0 / 100.0;
    params.ph_M_cone_ratio = 70.0 / 100.0;
    params.ph_L_cone_ratio = 25.0 / 100.0;

    //Parvo Ganglionar layer
    params.gc_fovea_radius = 225;
    params.gc_fovea_cones_by_cell = 1;
    params.gc_max_cones_by_cell = 5;
    params.gc_max_cones_by_cell_radius = 240;
    params.gc_fovea_inter_cells_distance = 1;
    params.gc_max_inter_cells_distance = 2;

    // RetinaCuda::Parameters params;
    // params.input_width = 3840;  //1280 640;
    // params.input_height = 2160; //720 480;

    // //cone sampling
    // params.ph_fovea_pixels_by_cone = 1;
    // params.ph_fovea_radius = 20;                               //80
    // params.ph_max_pixels_by_cone = 15;                         //5
    // params.ph_max_pixels_by_cone_radius = params.input_height; //180

    // //Ganglionar layer
    // params.gc_fovea_radius = 80;
    // params.gc_fovea_cones_by_cell = 1;
    // params.gc_max_cones_by_cell = 5;
    // params.gc_max_cones_by_cell_radius = 200;
    // params.gc_fovea_inter_cells_distance = 1;
    // params.gc_max_inter_cells_distance = 2;

    ///////////////////////////////////////

    // //cone sampling
    // params.ph_fovea_pixels_by_cone = 1;
    // params.ph_fovea_radius = 80;               //80
    // params.ph_max_pixels_by_cone = 20;         //5
    // params.ph_max_pixels_by_cone_radius = 720; //180

    // //Ganglionar layer
    // params.gc_fovea_radius = 80;
    // params.gc_fovea_cones_by_cell = 1;
    // params.gc_max_cones_by_cell = 5;
    // params.gc_max_cones_by_cell_radius = 200;
    // params.gc_fovea_inter_cells_distance = 1;
    // params.gc_max_inter_cells_distance = 2;

    RetinaCuda retina(1);
    retina.initRetina(params);

    cv::cuda::GpuMat gpuMatSrc(params.input_width, params.input_height, CV_8UC1);
    cv::cuda::GpuMat gpuMatCones(params.input_width, params.input_height, CV_8UC1);
    cv::cuda::GpuMat gpuMatDst(params.input_width, params.input_height, CV_8UC1);
    cv::cuda::GpuMat gpuMatPrev(params.input_width, params.input_height, CV_8UC1);
    cv::cuda::GpuMat gpuMatDirectionSelectiveOuput(params.input_width, params.input_height, CV_8UC1);

    cv::imwrite("cone_map.png", retina.drawConeMap());

    std::unique_ptr<cv::VideoCapture> cv_capture;
    if (input_param.empty())
    {
        cv_capture = std::make_unique<cv::VideoCapture>(-1); // open default capture device
    }
    else if (utils::isNumber(input_param))
    {
        cv_capture = std::make_unique<cv::VideoCapture>(std::stoi(input_param)); // open the video file for reading
    }
    else
    {
        cv_capture = std::make_unique<cv::VideoCapture>(input_param); // open the video file for reading
    }

    if (!cv_capture->isOpened()) // if not success, exit program
    {
        std::cout << "Cannot open the video file" << std::endl;
        return -1;
    }

    cv_capture->set(cv::CAP_PROP_FRAME_WIDTH, params.input_width);
    cv_capture->set(cv::CAP_PROP_FRAME_HEIGHT, params.input_height);

    //cv_capture->set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms
    //double fps = cv_capture->get(cv::CAP_PROP_FPS); //get the frames per seconds of the video

    //cv::namedWindow("MyVideo",cv::WINDOW_AUTOSIZE); //create a window called "MyVideo"
    //int i=0;
    cv::namedWindow("Camera input", cv::WINDOW_NORMAL);

    uint64_t counter = 0;
    bool pause = false;
    while (true)
    {
        cv::Mat frame;
        cv::Mat cframe;
        cv::Mat frameConeRetina;
        cv::Mat frameRetina;
        cv::Mat frameSelectiveRetina;
        cv::Mat cv_cm_img0;
        cv::Mat cv_cm_selective;

        if (!pause)
        {

            bool bSuccess = cv_capture->read(frame); // read a new frame from video
            std::cout << "Input frame dimensions" << frame.cols << " x " << frame.rows << std::endl;

            if (!bSuccess) //if not success, break loop
            {
                std::cout << "Cannot read the frame from video file" << std::endl;
                break;
            }

            if (grayscale)
            {
                cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
                cframe = cv::Mat(frame.rows, frame.cols, CV_8UC1, (char *)frame.data);
            }
            else
            {
                cframe = cv::Mat(frame.rows, frame.cols, CV_8UC3, (char *)frame.data);
            }

            TimePoint start = Time::now();
            gpuMatSrc.upload(cframe);

            TimePoint t_proc = Time::now();
            retina.applyPhotoreceptorSampling(gpuMatSrc, gpuMatCones);
            retina.applyParvoGC(gpuMatCones, gpuMatDst);
            retina.applyDirectiveGC(gpuMatCones, gpuMatDirectionSelectiveOuput, gpuMatPrev);
            ///retina.sparse(gpuMatDst,16,gpuSparseArray);
            ///retina.sparse(gpuMatDirectionSelectiveOuput,8,gpuSparseDirectionSelectiveArray,0,64);

            TimePoint t_proc_done = Time::now();

            ///retina.discretise(gpuMatDst,8,gpudiscreteGCArray);

            frameConeRetina = cv::Mat(gpuMatCones.rows, gpuMatCones.cols, CV_8UC1);
            gpuMatCones.download(frameConeRetina);

            //qDebug()<<"frameRetina"<<frameRetina.cols<<frameRetina.rows<<gpuMatDst.cols;
            frameRetina = cv::Mat(gpuMatDst.rows, gpuMatDst.cols, CV_8UC1);
            gpuMatDst.download(frameRetina);

            frameSelectiveRetina = cv::Mat(gpuMatDirectionSelectiveOuput.rows, gpuMatDirectionSelectiveOuput.cols, CV_8UC1);
            gpuMatDirectionSelectiveOuput.download(frameSelectiveRetina);
            TimePoint t_end = Time::now();

            // Holds the colormap version of the image:
            // Apply the colormap:
            cv::applyColorMap(frameRetina, cv_cm_img0, cv::COLORMAP_JET); //COLORMAP_RAINBOW COLORMAP_JET

            cv::applyColorMap(frameSelectiveRetina, cv_cm_selective, cv::COLORMAP_JET);

            //Discrete window
            ///HostBitArray2D discreteArray;
            ///gpuSparseArray.upload(discreteArray);
            ///cv::Mat cv_discrete(discreteArray.bytesHeight(),discreteArray.bytesWidth(),CV_8UC1,discreteArray.data());

            ///HostBitArray2D discreteSelectiveArray;
            ///gpuSparseDirectionSelectiveArray.upload(discreteSelectiveArray);
            ///cv::Mat cv_discrete_selective(discreteSelectiveArray.bytesHeight(),discreteSelectiveArray.bytesWidth(),CV_8UC1,discreteSelectiveArray.data());

            ///cv::Mat frameDiscrete(gpudiscreteGCArray.rows,gpudiscreteGCArray.cols,CV_8UC1);
            ///gpudiscreteGCArray.download(frameDiscrete);

            //gpuMatSrc.download(frameRetina);
            //std::cout<<"frameRetina "<<frameRetina.cols<<frameRetina.rows<<std::endl;
            cv::imshow("GC output" + std::to_string(cv_cm_img0.cols) + "x" + std::to_string(cv_cm_img0.rows), cv_cm_img0); //show the frame in "MyVideo" window
            cv::imshow("Camera input", frame);
            cv::imshow("MyVideo directional" + std::to_string(cv_cm_selective.cols) + "x" + std::to_string(cv_cm_selective.rows), cv_cm_selective);
            cv::imshow("Cones output", frameConeRetina);

            //cv::imshow("Sparse output",cv_discrete);
            //cv::imshow("Sparse selective output",cv_discrete_selective);

            ///cv::imshow("discrete output",frameDiscrete);

            std::cout << "processing time" << std::chrono::duration_cast<std::chrono::milliseconds>(t_proc_done - t_proc).count() << std::endl;
            std::cout << "all time" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - start).count() << std::endl;
        }

        char key = cv::waitKey(10);

        if (key == 'q') //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
        else if (key == 's' && !frame.empty())
        {
            std::stringstream stream;
            stream << std::setfill('0') << std::setw(5) << counter;
            cv::imwrite("camera_input_" + stream.str() + ".jpg", frame);
            cv::imwrite("gc_output_" + stream.str() + ".jpg", cv_cm_img0);
            cv::imwrite("cones_output_" + stream.str() + ".jpg", frameConeRetina);
        }
        else if (key == 'p')
        {
            pause = !pause;
        }

        counter++;
        //*if(i%10 == 1)
        gpuMatPrev = gpuMatCones.clone();
    }

    return 0;
}
