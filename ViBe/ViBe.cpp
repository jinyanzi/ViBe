#include <iostream>
#include <cmath>
#include "ViBe.h"

using namespace std;
using namespace cv;

ViBe::ViBe( int n, int r, int min, int s ){
	N = n;
	R = r*r;
	thresh_min = min;
	sub = s;
	cout << "ViBe()" << endl;
}

ViBe::~ViBe(){
	cout << "~ViBe()" << endl;
}

// this strange little hack with the global variable k supports the legacy way of initializing vibe, as well as a new way.
// after initialization, the user program can call generate_samples up to N-1 times to replace the static image samples with
// samples from a variety of images. Typically, one would sample with a bunch of widely spaced images. 

int k=0;
void ViBe::generate_samples( const Mat& img, const string& samples_name ) {
	if(k>=N) return;
	
	// Initialize samples if the initialization samples are given
	if( !samples_name.empty() ){
		readSamplesFromFile(samples_name);
		if( !samples.empty() )
			return;
		else{
			cout << "could not load sample file: " << samples_name << endl;
		}
	}
	// If the initialization samples are not given, use the given image.
	// 3-D matrix
	int sample_size[] = {height, width, N};
	// if samples are empty, create space
	if(samples.empty())
		samples.create( 3, sample_size, type );
	
	for( int i = 0; i < height; i++){
		for( int j = 0; j < width; j++ ){
			// initialize samples of every pixel with neighboring pixels
			Point neighbor = getRandomNeighbor(i, j);
			if(type == CV_8UC1)
				samples.at<uchar>(i, j, k) = img.ptr<uchar>(neighbor.y)[neighbor.x];
			if(type == CV_8UC3){
				samples.at<Vec3b>(i, j, k) = img.ptr<Vec3b>(neighbor.y)[neighbor.x];//data[img.channels()*neighbor.y*img.cols + neighbor.x];//
			}
		}
	}
	k++;
}


void ViBe::initialize( const Mat &img, const string& samples_name  ){
	image = img;
	width = img.cols;
	height = img.rows;
	type = img.type();
	
	foreground.create(height, width, CV_8UC1);
	
	if( samples_name.empty() ){
		cout << "samples empty" << endl;
	}else{
		cout << "samples not empty" << endl;
	}

	while(k<N) {
		generate_samples(img, samples_name );
		k++;
	}
	k=1; 
	
	cout << "initialization finished" << endl;
	srand( time(NULL) );
}

void ViBe::pixel_process( int row, int col){
	int count = 0, index = 0, dist = 0, sub_rand;
	// 1. compare pixel to background model
	// while not enough close samples and there is still sample not checked
	while( (count < thresh_min) && index < N ){
		dist = getDist( image, samples, row, col, index);
		if( dist < R )
			count++;
		if(count >= thresh_min)	break;	// break early
		index++;
	}

	//cout << "_________count \t" << count << endl;
	// 2. classify pixel and update model
	if(count >= thresh_min){
		// make this pixel background
		foreground.ptr<uchar>(row)[col] = COLOR_BACKGROUND;

		// 3. update current background model
		// get random number between 0 and sub
		sub_rand = random()%(sub-1);
		if( sub_rand == 0 ){
			//cout << "update sample\t( " << row << " , " << col << ")" << endl;
			// replace randomly chosen sample
			sub_rand = random()%(N-1);
			if( type == CV_8UC1 )
				samples.at<uchar>(row, col, sub_rand) = image.ptr<uchar>(row)[col];
			if( type == CV_8UC3 )
				samples.at<Vec3b>(row, col, sub_rand) = image.ptr<Vec3b>(row)[col];

		}
		// 4. update neighboring pixel model
		sub_rand = random()%(sub-1);	
		if( sub_rand == 0 ){
			//cout << "update neighbor\t( " << row << " , " << col << ")\t";
			// choose neighboring pixel randomly
			Point neighbor = getRandomNeighbor(row, col);
			//cout << neighbor << endl;
			sub_rand = random()%(N-1);
			if( type == CV_8UC1 )
				samples.at<uchar>(neighbor.y, neighbor.x, sub_rand) = image.ptr<uchar>(row)[col];
			if( type == CV_8UC3 )
				samples.at<Vec3b>(neighbor.y, neighbor.x, sub_rand) = image.ptr<Vec3b>(row)[col];

		}

	}else{
		// store this pixel as foreground
		foreground.ptr<uchar>(row)[col] = COLOR_FOREGROUND;
	}
}

bool ViBe::process(const Mat &frame, Mat &fore, const string& samples_name, bool if_bboxes){
	if( frame.cols <= 0 || frame.rows <= 0 ){
		cout << "this frame is empty" << endl;
		return false;
	}
	if( image.empty() )
		initialize( frame, samples_name );
	else{
		image = frame;
		for( int i = 0; i < height; i++ )
			for( int j = 0; j < width; j++ ){
				//cout << "(" << i << " , " << j << ")" << endl;
				pixel_process(i, j);
			}
	}
	fore = foreground;
	if(if_bboxes)
		findBlobs();
	return true;
}

// get rectangle mask from the fore ground
void ViBe::getMask( Mat &fore, Mat & mask, bool drawContour ){
	//erode(fore,fore,Mat());
	//dilate(fore,fore,Mat());

	mask = cv::Mat::zeros(height, width, CV_8UC1);

	if(drawContour)
		cvtColor(fore, fore, COLOR_GRAY2BGR);

	for(unsigned int i=0; i < connected_area.size(); i++) {
		Point2f vertices[4];
		Point v[4];
		rot_bboxes[i].points(vertices);

		for (int s = 0; s < 4; s++)
			v[s] = Point(int(vertices[s].x), int(vertices[s].y));

		if(drawContour){
			rectangle(fore, bboxes[i], Scalar(0, 255, 0));
		}

		fillConvexPoly(mask, v, 4, 255);//Scalar(255, 255, 255));
	}
}


void ViBe::saveSamplesToFile(const string& file_name){
	cout << "save samples to file " << file_name << endl;
	FileStorage fs( file_name, FileStorage::WRITE );
	fs << string("samples") << samples;
	fs.release();
}

void ViBe::readSamplesFromFile(const string& file_name){
	// Assume mat has the same name with file_name
	cout << "read samples from file " << file_name << endl;
	FileStorage fs( file_name, FileStorage::READ );
	fs[string("samples")] >> samples;
	fs.release();
}

int ViBe::getBlobSize(){	
	return blob_num;
}

Mat& ViBe::getSamples(){
	return samples;
}

Point ViBe::getRandomNeighbor( int row, int col){
	// on both x and y direction, generate a random number
	// use 8-connected neighbor
	int rand_row=0, rand_col =0, row_from = NEIGHBOR_RANGE, row_to = NEIGHBOR_RANGE*2+1, 
		col_from = NEIGHBOR_RANGE, col_to = NEIGHBOR_RANGE*2+1;
	// deal with the edge pixels
	if(row < NEIGHBOR_RANGE){
		row_from = 0;
		row_to = NEIGHBOR_RANGE+1;
	}
	if( row >= height-NEIGHBOR_RANGE){
		row_to = NEIGHBOR_RANGE+1;
	}
	if( col < NEIGHBOR_RANGE){
		col_from = 0;
		col_to = NEIGHBOR_RANGE+1;
	}
	if( col >= height-NEIGHBOR_RANGE ){
		col_to = NEIGHBOR_RANGE+1;
	}

	// return one of 8-connected neighbors except itself
	do{
		rand_row = random()% row_to - row_from;
		rand_col = random()% col_to - col_from;
	}while(rand_row == 0 && rand_col == 0);

	//cout << "(" << rand_row+row << "," << rand_col+col << ")\t" << endl;
	return Point( col+rand_col, row+rand_row );
}	

float ViBe::getDist( Mat &img, Mat &sample, int row, int col, int index ){
	// because we use grayscale image, just do simple subtraction
	if( type == CV_8UC1 ){
		int dist = img.ptr<uchar>(row)[col] - sample.at<uchar>(row, col, index);
		return dist*dist;
	}
	// compute Euclidean distance in 3D color space
	if( type == CV_8UC3 ){
		Vec3b img_color = img.ptr<Vec3b>(row)[col], 
			  sample_color = sample.at<Vec3b>(row, col, index);
		int b_diff = img_color.val[0] - sample_color.val[0],
			g_diff = img_color.val[1] - sample_color.val[1],
			r_diff = img_color.val[2] - sample_color.val[2];
		return b_diff*b_diff + g_diff*g_diff + r_diff*r_diff;
	}
	return -1;
}

// find connected area and return the bounding rectangle
void ViBe::findBlobs()
{
	bboxes.clear();
	rot_bboxes.clear();
	connected_area.clear();

	// Fill the label_image with the blobs
	// 0  - background
	// 255  - unlabelled foreground
	// 1-254 - labelled foreground

	foreground.convertTo(label_image, CV_32FC1); // weird it doesn't support CV_32S!

	int label_count = 1; // starts at 2 because 0,1 are used already

	for(int y=0; y < foreground.rows; y++) {
		for(int x=0; x < foreground.cols; x++) {

			if((int)label_image.ptr<float>(y)[x] != 255) {
				continue;
			}

			Rect bbox;
			// give connected area the same label number
			floodFill(label_image, Point(x,y), Scalar(label_count), &bbox, Scalar(0), Scalar(0), 4);

			if( bbox.area() > MIN_BLOB_AREA){ 
				vector <Point2i> pts_group;
				bboxes.push_back(bbox);

				for(int i=bbox.y; i < (bbox.y+bbox.height); i++) {
					for(int j=bbox.x; j < (bbox.x+bbox.width); j++) {
						if((int)label_image.ptr<float>(i)[j] != label_count) {
							continue;
						}
						// get all the points in a connected area
						pts_group.push_back(Point2i(j,i));
					}
				}
				connected_area.push_back(pts_group);
				label_count++;
			}
		}
	}

	sort(bboxes.begin(), bboxes.end(), ViBe::rect_larger_than);
	blob_num = bboxes.size();

	// get rotated rectangles
	for(unsigned int i=0; i < connected_area.size(); i++) {
		// get minimal bounding rotated rectangle of each blob
		RotatedRect rot_box = minAreaRect(Mat(connected_area[i]));
		Point2f vertices[4];
		Point v[4];
		rot_box.points(vertices);

		for (int s = 0; s < 4; s++)
			v[s] = Point(int(vertices[s].x), int(vertices[s].y));
		
		// Add rotated rotated rectangle 
		rot_bboxes.push_back(rot_box);
	}
}


// get the image filtered by the mask
void ViBe::getMaskedImg( Mat & img, Mat & mask_img){
	int rows = img.rows, cols = img.cols;
	if( rows != mask_img.rows || cols != mask_img.cols ){
		cout << "image and mask_img are not the same size..." << endl;
		return;
	}

	for( int i = 0; i < rows; i++ ){
		for(int j = 0; j < cols; j++ ){
			// if this is background image, set it to black
			// // else, keep the pixel value
			if( mask_img.type() == CV_8UC1 ){
				if( mask_img.ptr<uchar>(i)[j] == COLOR_BACKGROUND ){
					img.ptr<Vec3b>(i)[j] = Vec3b(0, 0, 0);
				}	
			}
			if(mask_img.type() == CV_8UC3){
				if( mask_img.ptr<Vec3b>(i)[j] == Vec3b(0, 0, 0) ){
					img.ptr<Vec3b>(i)[j] = Vec3b(0, 0, 0);
				}
			}
		}
	}
}

