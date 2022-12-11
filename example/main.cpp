#include <unistd.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <iostream>

#include "CLI/CLI.hpp"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"

#include "retinacuda.h"
#include "utils/filesystemtools.h"
#include "utils.h"
#include "utils/colormap.h"

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
    params.pix_config.camera_hfov = 74.0;
    params.pix_config.camera_width = 3840;
    params.pix_config.camera_height = 2160;
    params.ph_config.ph_S_cone_ratio = 5.0 / 100.0;
    params.ph_config.ph_M_cone_ratio = 70.0 / 100.0;
    params.ph_config.ph_L_cone_ratio = 25.0 / 100.0;

    RetinaCuda retina(1);
    retina.initRetina(params);

#ifdef WITH_MATPLOTLIB
    retina.plotLayersInfos();
#endif

    cv::cuda::GpuMat gpuMatSrc(params.input_width, params.input_height, CV_8UC1);
    cv::cuda::GpuMat gpuMatCones;
    cv::cuda::GpuMat gpuMatDst;
    cv::cuda::GpuMat gpu_magno_frame;
    cv::cuda::GpuMat gpuMatPrev;
    cv::cuda::GpuMat gpuMatDirectionSelectiveOuput;

    //cv::imwrite("cone_map.png", retina.drawConeMap());

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

    cv::namedWindow("Camera input", cv::WINDOW_NORMAL);
    cv::namedWindow("Cones output", cv::WINDOW_NORMAL);
    cv::namedWindow("Midget Ganglionar cells responses", cv::WINDOW_NORMAL);
    cv::namedWindow("Magno parasol ganglionar cells responses", cv::WINDOW_NORMAL);

    // Create a colormap to display ganglionar cell response
    auto color_mapping = colormap::mirrorColorMap(colormap::reverseColorMap(colormap::buildColorMap(colormap::COLORMAP_TYPE::BLUE_GREEN_RED)));
    cv ::Mat lut = colormap::convertToLUT(color_mapping);
    cv::Mat color_map_img = colormap::buildColorMapImage(lut, 255, 80);

    uint64_t counter = 0;
    bool pause = false;
    while (true)
    {
        cv::Mat frame;
        cv::Mat cframe;
        cv::Mat frameConeRetina;
        cv::Mat parvo_frame;
        cv::Mat magno_frame;
        cv::Mat frameSelectiveRetina;
        cv::Mat colorized_parvo;
        cv::Mat colorized_magno;
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
            retina.applyMagnoGC(gpuMatCones, gpu_magno_frame);
            ///retina.applyDirectiveGC(gpuMatCones, gpuMatDirectionSelectiveOuput, gpuMatPrev);

            TimePoint t_proc_done = Time::now();

            frameConeRetina = cv::Mat(gpuMatCones.rows, gpuMatCones.cols, CV_8UC1);
            gpuMatCones.download(frameConeRetina); //cudaStreamDefault

            if (gpuMatDst.rows > 0)
            {
                parvo_frame = cv::Mat(gpuMatDst.rows, gpuMatDst.cols, CV_8UC1);
                gpuMatDst.download(parvo_frame);
            }

            if (gpu_magno_frame.rows > 0)
            {
                magno_frame = cv::Mat(gpu_magno_frame.rows, gpu_magno_frame.cols, CV_8UC1);
                gpu_magno_frame.download(magno_frame);
            }

            if (gpuMatDirectionSelectiveOuput.rows > 0)
            {
                frameSelectiveRetina = cv::Mat(gpuMatDirectionSelectiveOuput.rows, gpuMatDirectionSelectiveOuput.cols, CV_8UC1);
                gpuMatDirectionSelectiveOuput.download(frameSelectiveRetina);
            }
            TimePoint t_end = Time::now();

            // Holds the colormap version of the image:
            // Apply the colormap:
            if (parvo_frame.cols > 0)
            {
                cv::cvtColor(parvo_frame, colorized_parvo, cv::COLOR_GRAY2RGB);
                cv::LUT(colorized_parvo, lut, colorized_parvo);
                color_map_img.copyTo(colorized_parvo(cv::Rect(0, 0, color_map_img.cols, color_map_img.rows)));

                cv::imshow("Midget Ganglionar cells responses", colorized_parvo); //show the frame in "MyVideo" window
            }

            if (magno_frame.cols > 0)
            {
                cv::cvtColor(magno_frame, colorized_magno, cv::COLOR_GRAY2RGB);
                cv::LUT(colorized_magno, lut, colorized_magno);
                cv::imshow("Magno parasol ganglionar cells responses", colorized_magno); //show the frame in "MyVideo" window
            }

            if (frameSelectiveRetina.cols > 0)
            {
                cv::applyColorMap(frameSelectiveRetina, cv_cm_selective, cv::COLORMAP_JET);
                cv::imshow("MyVideo directional" + std::to_string(cv_cm_selective.cols) + "x" + std::to_string(cv_cm_selective.rows), cv_cm_selective);
            }

            cv::imshow("Camera input", frame);
            cv::imshow("Cones output", frameConeRetina);

            std::cout << "processing time" << std::chrono::duration_cast<std::chrono::milliseconds>(t_proc_done - t_proc).count() << std::endl;
            std::cout << "all time" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - start).count() << std::endl;
        }

        char key = cv::waitKey(20);

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
            cv::imwrite("parvo_output_" + stream.str() + ".jpg", colorized_parvo);
            cv::imwrite("magno_output_" + stream.str() + ".jpg", colorized_magno);
            cv::imwrite("cones_output_" + stream.str() + ".jpg", frameConeRetina);
        }
        else if (key == 'p')
        {
            pause = !pause;
        }
        counter++;
        gpuMatPrev = gpuMatCones.clone();
    }

    return 0;
}