#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <vector>
#include <string>

#ifndef _VIBE_H_
#define _VIBE_H_

// background and foreground color value
#define COLOR_BACKGROUND 0
#define COLOR_FOREGROUND 255
#define NEIGHBOR_RANGE 1
#define MIN_BLOB_AREA 50

class ViBe{
public:
	ViBe( int n = 20, int r = 20, int min = 2, int s = 16 );
	~ViBe();
	void initialize( const cv::Mat &img, const std::string& samples_name = "" );
	void pixel_process( int row, int col);
	void generate_samples( const cv::Mat & img, const std::string& samples_name = "");

	bool process(const cv::Mat &frame, cv::Mat &fore, const std::string& samples_name = "", bool if_bboxes = true);		// if_bbox indicates whether to get bounding boxes

	void saveSamplesToFile(const std::string& file_name);
	void readSamplesFromFile(const std::string& file_name);
	
	// get rectangle mask from the fore ground
	void getMask( cv::Mat &fore, cv::Mat & mask, bool drawContour = false);
	// get the image filtered by the mask
	void getMaskedImg(cv::Mat &img, cv::Mat &mask);
	bool isSamplesEmpty()	const{	return samples.empty();	}
	int getBlobSize();
	cv::Mat& getSamples();
	std::vector< cv::RotatedRect > getRotBboxes(){	return rot_bboxes;	}	// get rotated bounding boxes
	std::vector< cv::Rect > getBBoxes(){	return bboxes;	}	// get bounding boxes

private:
	int N;				// number of samples per pixel(default 20)
	int R;				// radius of the sphere(default 20)
	int thresh_min;		// number of close samples for being part of the background(default 2)
	int sub;			// amount of random subsampling(default 16)
	int width;
	int height;
	int type;
	int blob_num;
	cv::Mat image;		// current image
	cv::Mat samples;	// background model
	cv::Mat foreground;	// foreground/background segmentation map
	cv::Mat label_image;
	std::vector<cv::Rect> bboxes;
	std::vector<cv::RotatedRect> rot_bboxes;
	std::vector<std::vector<cv::Point2i> > connected_area;

	cv::Point getRandomNeighbor(int row, int col);

	float getDist(cv::Mat &img, cv::Mat &sample, int row, int col, int index);
	
	// find connected area and return the bounding rectangle
	void findBlobs();	
	std::vector<std::vector<cv::Point2i> >& findRotatedBlobs(const cv::Mat &binary, std::vector <cv::RotatedRect> &rot_blobs);
	static bool rect_larger_than( const cv::Rect& r1, const cv::Rect& r2 ){	return (r1.area() > r2.area());	}
};


#endif 
