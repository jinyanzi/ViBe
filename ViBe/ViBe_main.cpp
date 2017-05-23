#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ViBe.h"
#include "trajDebugger.h"

#include <iostream>
#include <string>
#include <stdlib.h>
#include <unistd.h>
#include <ctime>

/*
 * Input parameters:
 *  -o <out_sample_name>: output background sample NAME
 *  -v <out_video_name>: output forground video NAME
 *  -s <sample_path>: pre-run background sample PATH
 *  -i <input_video_path>: input video PATH
 *  -f <frame number>: number of frame to process
 *	-r: flag indicating backwards processing
 *	-m: no display
 *
 * Generated Images:
 *  
 */


using namespace std;
using namespace cv;

struct Options{
    Options():
        write_samples(false),
        write_video(false),
        use_samples(false),
		backwards(false),
		display(true),
		drawCountour(false),
		resize_factor(1),
		to_frame_num(-1),
        out_samples_name(),
        out_video_name(),
        in_samples_name(),
        video_name()
    {}
    bool write_samples;	// whether to write samples files
    bool write_video;	// whether to write videos 
    bool use_samples;	// whether to use prerunning samples
	bool backwards;
	bool display;
	bool drawCountour;
	double resize_factor;
	int to_frame_num;
    string out_samples_name;
    string out_video_name;
    string in_samples_name;
    string video_name;
	string gt_path;
};

void print_help(){
    cout << "Usage: ./ViBe [-o output samples file] "
        << "[-v output video file name] [-s prerunning samples file path] "
        << "[-i Video file path] [-g ground truth path]"
		<< "[-f number of frame to process] [-r resize_factor]"
		<< "[-m disable display during processing] "
		<< "[-b backwards processing] "
        << endl;

}

void parse_command_line( int argc, char** argv, Options& o ){
    char c = -1;
    if(argc <= 1){
        print_help();
        exit(0);
    }

    while( ( c = getopt(argc, argv, "i:s:v:g:o:f:r:cbm")) != -1 ){
        switch(c){
			case 'i':
				o.video_name = optarg;
				break;
			// output sample name
            case 'o':
                o.write_samples = true;
                o.out_samples_name = optarg;
                break;
			// ground truth path
			case 'g':
				o.gt_path = optarg;
				break;
			// output foreground video name
            case 'v':
                o.write_video = true;
                o.out_video_name = optarg;
                break;
			// output sample file name
            case 's':
                o.use_samples = true;
                o.in_samples_name = optarg;
                break;
			case 'r':
				o.resize_factor = atof(optarg);
				break;
			// last frame to process
			case 'f':
				o.to_frame_num = atoi(optarg);
				break;
			case 'b':
				o.backwards = true;
				break;
			case 'c':
				o.drawCountour = true;
				break;
			case 'm':
				o.display = false;
				break;
            default:
                print_help();
                break;
        }
    }

	cout << "Open video " << o.video_name << endl; 
    cout << (o.use_samples ?"Use samples file: "
            :"No pre-running sample") << o.in_samples_name << endl;
    cout << (o.write_samples? "Write samples: ": "No output samples") << o.out_samples_name << endl;
    cout << (o.write_video? "Write to video: ": "No output video") << o.out_video_name << endl;
	cout << (o.backwards ? "Run video backwardsly.\n" : "");
	cout << "============================================================================" << endl;
}


int main(int argc, char **argv)
{
    VideoCapture cap;
    VideoWriter record;
	TrajDebugger debugger;
	bool gt_successful = false;

    Options o;
    parse_command_line(argc, argv, o);

	if(!o.gt_path.empty())	gt_successful = debugger.readGroundTruthFromFile(o.gt_path);

    cout << "Open video " << o.video_name << endl;
    if( !cap.open(o.video_name) ){
        cout << "Failed to open the video" << o.video_name << " , exiting..." << endl;
        return -1;
    }

    ViBe vb;
    Mat frame, fore, mask;

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int frames = cap.get(CAP_PROP_FRAME_COUNT);
    double framerate = cap.get(CAP_PROP_FPS);
    cout << "total " << frames << " frames" << endl;
    //int fourcc = int(cap.get(CAP_PROP_FOURCC));

	Mat big_frame(height, width*2, CV_8UC3);
    vector<vector<Point> > countours;
    vector<Vec4i> hierarchy;

    bool play = true;
	// if we are going to write to a sample file, wait until a stable foreground
	o.to_frame_num = (o.to_frame_num < 0) ? frames : o.to_frame_num;
	if(o.backwards)	cap.set(CV_CAP_PROP_POS_FRAMES, o.to_frame_num-1);
    for(int i = 0; i < o.to_frame_num;)	
    {
        if( i == 0 || play ){
            cap >> frame;
			++i;
            cout << "=========== " << o.video_name << " frame " << cap.get(CV_CAP_PROP_POS_FRAMES) 
				 << "/" << frames << "\t" << frame.size() << "\t" 
				 << "==========" << endl;
//            play = false;

			const clock_t begin_time = clock();
            if(vb.process(frame, fore, o.in_samples_name)){
				std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << "\t";	
                //erode(fore,fore,Mat());
                //dilate(fore,fore,Mat());
                vb.getMask(fore, mask, false);
				// mark corect/incorrect pixels
				if(gt_successful) {
					Mat matchMask;
					debugger.GTForeMask(frame, fore, i, Scalar(0, 255, 0), Scalar(0, 0, 255), matchMask);
					addWeighted(frame,0.6,matchMask,0.4,0, frame);
				}

				cvtColor(fore, fore, COLOR_GRAY2BGR);
				
				if(o.resize_factor != 1) {
					cv::resize(fore, fore, cv::Size(0, 0), o.resize_factor, o.resize_factor);
					cv::resize(frame, frame, cv::Size(0, 0), o.resize_factor, o.resize_factor);
				}

                //vb.getMaskedImg(frame, fore);   
				vector<Rect> boxes = vb.getBBoxes();
				for(Rect b : boxes) {
					//rectangle(frame, Rect(b.tl()*o.resize_factor, b.br()*o.resize_factor), Scalar(0, 255, 0), std::max(1.0,o.resize_factor));
					if(o.drawCountour)
						rectangle(fore, Rect(b.tl()*o.resize_factor, b.br()*o.resize_factor), Scalar(0, 255, 0), std::max(1.0,o.resize_factor));
				}	
				//Mat roi = big_frame(Rect(0, 0, width, height));
				//frame.copyTo(roi);
				//cvtColor(fore, fore, COLOR_GRAY2BGR);
				//roi = big_frame(Rect(width, 0, width, height));
				//fore.copyTo(roi);
                
				// Ready to write to video files
                if(o.write_video){
                    // If the record is not opened, opend it and check
                    if( !record.isOpened() ){
                        record.open(o.out_video_name, VideoWriter::fourcc('X', 'V', 'I', 'D'), 
                                framerate, cv::Size(width*o.resize_factor, height*o.resize_factor), true);
                        if( !record.isOpened() ){
                            cout << "Failed to open the video writer" << endl;
                            return -1;
                        }
                    }

					if(gt_successful)	
						record.write(frame);
					else	
						record.write(fore);
					//record.write(big_frame);
                }

				//if(i == 3201) {
				//	imwrite("box-3201.jpg", frame);
				//	imwrite("fore-3201.jpg", fore);
				//}

				if(o.display){
					imshow("foreground", fore);
					//imshow("mask", mask);
					//imshow("frame", frame);
					//imshow("big_frame", big_frame);
				}
            }

			if(o.backwards){
				cap.set(CV_CAP_PROP_POS_FRAMES, cap.get(CV_CAP_PROP_POS_FRAMES)-2);	// set next frame to be the previous frame
			}
        }

		if(o.display){
			char c = waitKey(30);
			if( c == 27 || c == 'q')
				break;
			if(c == ' ')
				play = !play;
		}
    }

    cout << "==========finished===========" << endl;
    if(o.write_samples)
        vb.saveSamplesToFile( o.out_samples_name );

    cap.release();
    return 0;
}


