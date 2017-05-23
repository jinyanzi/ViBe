#ifndef TRAJ_DEBUGGER_H
#define TRAJ_DEBUGGER_H

#define RED cv::Scalar(0, 0, 255)
#define GREEN cv::Scalar(0, 255, 0)
#define BLUE cv::Scalar(255, 0, 0)
#define YELLOW cv::Scalar(0, 255, 255)
#define PINK cv::Scalar(255, 0, 255)
#define CYAN cv::Scalar(255, 255, 0)

#define RESULT 0
#define GROUND_TRUTH 1
#define DEBUG 1

#define TOP_LEFT 0
#define BOTTOM_RIGHT 1
#define STATIONARY_DIST 8

#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>


struct TrajUnit{
	TrajUnit(int f_id = -1, const cv::Rect& r = cv::Rect(), int occluded = 0, const std::string& c = ""):frame_id(f_id), box(r), if_occluded(occluded), cls(c){}
	int frame_id;
	cv::Rect box;
	int if_occluded;
	std::string cls;
};

typedef std::vector<TrajUnit> TrajVec;

class TrajDebugger{
protected:
	// draw trajectory and box
    // obj_id = -1, draw all the objects
    void drawTrajectory(cv::Mat& img, int result_or_gt, int to_frame_num, const cv::Scalar& color, int thickness, double fx = 1, double fy = 1, int location = TOP_LEFT, int obj_id = -1)	const;
    
	// draw trajectory and box of a certain object
    // return the index of the object at to_frame_num
    int drawObject(cv::Mat& img, const std::map<int, TrajVec>::const_iterator it, int to_frame_num, const cv::Scalar& color, int thicknes = 1,  double fx = 1, double fy = 1, int location = TOP_LEFT)	const;
    
	void drawNumBox(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, int num,  double fx = 1, double fy = 1, int location = TOP_LEFT)	const;

	void readTrajLine(const std::string& line, int& obj_id, int& frame_id, cv::Rect& box, std::string& cls, int& occluded)	const;
    
	// update object trajcetory with box
	void updateObject(int obj_id, int frame_id, const cv::Rect& box, std::map<int, TrajVec>& trajectories);

	void areaSum(const std::map<int, TrajVec>& trajectory, std::map<int, float>& area_map)	const;
    
	int getObjectNum()	const {	return ground_truth.size();	}

	cv::Point scale_point(const cv::Point& p, double rx, double ry)	const {	return cv::Point(p.x*rx, p.y*ry);	}
	cv::Rect scale_rect(const cv::Rect& box, double rx, double ry)	const {	return cv::Rect(scale_point(box.tl(), rx, ry), scale_point(box.br(), rx, ry));	}
    // variables
	std::map<int, TrajVec> ground_truth;

public:
    // read functions
	// ground truth are written in object id order
	// to_frame_num = -1 means read until end of file
    
	bool readGroundTruthFromFile(const std::string& filename, int to_frame_num = -1);
	// print functionvoid 
	void printTrajectorySummary(int obj_id)	const;
	void cleanGroundTruth();
	
	// match with foreground
	void GTForeMask(const cv::Mat& frame, const cv::Mat& fore, int frame_num, const cv::Scalar& correct_color, const cv::Scalar& incorrect_color, cv::Mat& mask)	const;

	std::string getFileNameFromPath(const std::string& file_path) const;
	std::string getFileDirPath(const std::string& file_path)	const;
	
	// return the intersection area size in pixel
	int overlappedArea(const cv::Rect &r1, const cv::Rect &r2)	const {	return (r1&r2).area();	};
	// return whether the two rectangles are overlapped
	int isOverlapped(const cv::Rect &r1, const cv::Rect &r2)	const	{	return overlappedArea(r1, r2)>0;	};
	// return the overlapped portion of two rectangles
	float overlapRatio(const cv::Rect& r1, const cv::Rect& r2)	const	{	return overlappedArea(r1, r2)/double(r1.area() + r2.area() - overlappedArea(r1, r2));	}
	float boxDist(const cv::Rect& r1, const cv::Rect& r2)	const {	return cv::norm(center(r1)-center(r2));	}
    cv::Point2f center(const cv::Rect& r) const{  return cv::Point2f(r.x+r.width/2, r.y+r.height/2); }
};


#endif
