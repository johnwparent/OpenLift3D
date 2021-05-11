// --- OpenPose C++ API Tutorial - Example 17 - Custom Input, Pre-processing, Post-processing, Output, and Datum ---
// Synchronous mode: ideal for production integration. It provides the fastest results with respect to runtime
// performance.
// In this function, the user can implement its own way to read frames, implement its own post-processing (i.e., his
// function will be called after OpenPose has processed the frames but before saving), visualizing any result
// render/display/storage the results, and use their custom Datum structure

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
// #define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
DEFINE_string(image_dir,                "vid/",
    "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");

DEFINE_string(camera_parameter_path, "cal/",
    "String with the folder where the camera parameters are located. If there is only 1 XML file (for single"
    " video, webcam, or images from the same camera), you must specify the whole XML file path (ending in .xml).");

DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

// // This worker will just read and return all the basic image file formats in a directory
class WUserInput : public op::WorkerProducer<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>>
{
public:
    WUserInput(const std::string& directoryPath, const std::string& cameraParameterPath) :
        mFrameCounter{0}
    {
        auto &vid_files(op::getFilesOnDirectory(directoryPath,std::vector<std::string>{".avi",".mp4"}));
        for(auto vid_f : vid_files)
        {
            op::opLog(vid_f, op::Priority::High);
            mCaptures.emplace_back(cv::VideoCapture(vid_f));
        }
        mCameraParameterReader.readParameters(cameraParameterPath);

    }
    void initializationOnThread() {}

    std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> workProducer()
    {
        try
        {
            bool cont = true;

            for (auto &vidStream: mCaptures){
                if (!vidStream.isOpened())
                {
                    cont = false;
                }
            }
            if (mCaptures.size() == 0)
            {
                cont = false;
            }
            // Close program when empty frame
            if (! cont)
            {
                op::opLog(
                        "Last frame read and added to queue. Closing program after it is processed.", op::Priority::High);
                    // This funtion stops this worker, which will eventually stop the whole thread system once all the
                    // frames have been processed
                    this->stop();
                    return nullptr;
            }
            else
            {
                std::lock_guard<std::mutex> block(locker);
                if (q.empty())
                {
                    for (int x = 0; x<mCaptures.size(); x++)
                    {
                        // Create new datum
                        auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
                        datumsPtr->emplace_back();

                        auto& AdatumPtr = datumsPtr->back();
                        AdatumPtr = std::make_shared<op::Datum>();
                        // populate frame data
                        cv::Mat f1;
                        mCaptures.at(x).read(f1);
                        if (f1.empty())
                        {
                            op::opLog(
                                "Empty frame detected, closing program.",
                                op::Priority::High);
                            this->stop();
                            datumsPtr = nullptr;
                        }

                        auto &camMatrices = mCameraParameterReader.getCameraMatrices();
                        auto &camIntrinsics = mCameraParameterReader.getCameraIntrinsics();
                        auto &camExtrinsics = mCameraParameterReader.getCameraExtrinsics();

                        AdatumPtr->name = "side";
                        AdatumPtr->cvInputData = OP_CV2OPCONSTMAT( f1 );
                        AdatumPtr->cvOutputData = AdatumPtr->cvInputData;
                        AdatumPtr->cameraMatrix = camMatrices[x];
                        AdatumPtr->cameraIntrinsics = camIntrinsics[x];
                        AdatumPtr->cameraExtrinsics = camExtrinsics[x];
                        AdatumPtr->subId=x;
                        AdatumPtr->subIdMax=1;
                        q.push( datumsPtr );
                    }
                }
            auto CurrentDatum = q.front();
            q.pop();
            return CurrentDatum;

            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

private:
    std::queue<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>> q;
    std::vector<cv::VideoCapture> mCaptures;
    unsigned long long mFrameCounter;
    op::CameraParameterReader mCameraParameterReader;
    std::mutex locker;
};


// This worker will just invert the image
class WUserPostProcessing : public op::Worker<std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>>
{
public:
    WUserPostProcessing()
    {
        // User's constructor here
    }

    void initializationOnThread() {}

    void work(std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>& datumsPtr)
    {
        try
        {
        }
        catch (const std::exception& e)
        {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

// This worker will just read and return all the jpg files in a directory
class WUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>>
{
public:
    void initializationOnThread() {}

    void workConsumer(const std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>& datumsPtr)
    {
        try
        {

        }
        catch (const std::exception& e)
        {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

void configureWrapper(op::WrapperT<op::Datum>& opWrapperT)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Initializing the user custom classes
        // Frames producer (e.g., video, webcam, ...)
        auto wUserInput = std::make_shared<WUserInput>(FLAGS_image_dir, FLAGS_camera_parameter_path);
        // GUI (Display)
        // Add custom input
        const auto workerInputOnNewThread = false;
        opWrapperT.setWorker(op::WorkerType::Input, wUserInput, workerInputOnNewThread);
        // Add custom output
        const auto workerOutputOnNewThread = true;
        // opWrapperT.setWorker(op::WorkerType::Output, wUserOutput, workerOutputOnNewThread);

        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port)};
        // opWrapperT.configure(wrapperStructOutput);
        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapperT.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapperT.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapperT.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapperT.configure(wrapperStructExtra);
        opWrapperT.configure(wrapperStructOutput);
        const op::WrapperStructGui wrapperStructGui{
            op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        opWrapperT.configure(wrapperStructGui);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapperT.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}


int tutorialApiCpp()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // OpenPose wrapper
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::WrapperT<op::Datum> opWrapperT;
        configureWrapper(opWrapperT);

        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapperT.exec();
        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
