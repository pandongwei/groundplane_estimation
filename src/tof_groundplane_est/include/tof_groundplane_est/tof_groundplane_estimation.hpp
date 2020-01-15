#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <Eigen/Eigen>
#include "camera_models/camera.h"

namespace votof{

class TofGround {

public:

	struct Point{
		uint32_t u;
		uint32_t v;
	};

	  struct calibration {
		float f;		// focal length (in pixels)
	    float cu; 		// principal point (u-coordinate)
	    float cv; 		// principal point (v-coordinate)

	    calibration () {
	    	f = -1;
	    	cu = -1;
	    	cv = -1;
	    }
	  };

	  //ROI parameters,表示ROI区域
	  struct ROI {
		  Point up_left;			//ROI Point up left
		  Point up_right;			//ROI Point down right
		  Point down_left;			//
		  Point down_right;			//
		  ROI() {
			  up_left.u = 70; 		//80 ;
			  up_left.v = 5;		//287-110;
			  up_right.u = 270;		//260;
			  up_right.v = 5;		//287-110;
			  down_left.u = 120;	//60;
			  down_left.v = 95;		//280;
			  down_right.u = 240;	//290;
			  down_right.v = 95;	//280;
		  }
	  };

	// bucketing parameters
	struct bucketing {
	    int32_t max_features;  // maximal number of features per bucket
	    float  bucket_width;  // width of bucket
	    float  bucket_height; // height of bucket
	    bucketing () {
	    	max_features  = 22;
	    	bucket_width  = 30;
	    	bucket_height = 30;
	    }
	};

	struct parameters {
		bucketing bucket;
		calibration calib;
		ROI roi;
		int32_t ransac_iter;			// Number of RANSAC runs
		float inlier_threshold;			// inlier threshold in meter
		int32_t min_Points_in_ROI;		// minimal Number of valid Points in ROI i
		int32_t min_Points_in_bucket;	// minimal Number of valid Points in a bucket
		int32_t min_used_bucket;		// minimal Number of buckets with min. "min_Points_in_bucket" in it
		float filter_max_z;				// filter for deleting Points, which are bigger or smaller then filter value; For deleting not valid Pixel
		float filter_min_z;				// in meters
		float filter_max_x;
		float filter_min_x;
		float filter_max_y;
		float filter_min_y;
		parameters() {  // 这些是初始化的参数
			ransac_iter = 200;
		    inlier_threshold = 0.01;
		    min_Points_in_ROI = 210;
		    min_Points_in_bucket = 15;
		    min_used_bucket = 15;
			filter_max_z = 10;
			filter_min_z = 0.01;
			filter_max_x = 15;
			filter_min_x = -15;
			filter_max_y = 15;
			filter_min_y = -15;
		}
	};


	struct includepix{
		uint32_t u;
		uint32_t v;
		float depth;
		float x;
		float y;
	};


	struct planeHN{
		Eigen::Vector3d n;
		float d;
		planeHN(){
			n = Eigen::Vector3d::Zero();
			d = 0;
		}
	};

	struct border{
		uint32_t u_min;
		uint32_t u_max;
		uint32_t v_min;
		uint32_t v_max;
	};

	struct cornerPixel{
		uint32_t u_ol, v_ol;
		float x_ol, y_ol, z_ol;
		uint32_t u_or, v_or;
		float x_or, y_or, z_or;
		uint32_t u_ur, v_ur;
		float x_ur, y_ur, z_ur;
		uint32_t u_ul, v_ul;
		float x_ul, y_ul, z_ul;

	};

	enum filtertyp{none, pointer, mat}; // TODO


	TofGround (parameters param, uint32_t image_width, uint32_t image_height);
	TofGround (const TofGround&)=delete;
	~TofGround();


	// inputs: I ......... pointer to image (float16, row-aligned)
	// output: returns false if it's not possible to estimate a groundplane
	bool estimateGroundplane(float* I, bool use_min_number = false);
	bool estimateGroundplane(float* I, std::unique_ptr<CameraModel> &CamModel, bool use_min_number = false);
	bool estimateGroundplane(float* Ix, float* Iy, float* Iz, bool use_min_number = false);


	// returns the estimated Plane
	planeHN getGroundplane(){return ebeneHN_;};
	// returns the height / distance between Ground and Camera
	float getPlaneDistance() {return ebeneHN_.d;};
	// returns all valid Pixel in ROI
	std::vector<TofGround::includepix> getAllPxInRoi(){return allPxInROI_;};
	//returns all inliers
	std::vector<int32_t> getInlierList(){return inliers_;};
	//set ROI
	void setROI();
	void setROI(Point down_left, Point down_right, Point up_left, Point up_right);
	void setROI(uint8_t* filtermask, uint32_t pointer_size);
	void setROI(cv::Mat& filtermask);

	TofGround::cornerPixel getCornerPixel(float* X, float* Y, float* Z);

	uint32_t getImageHeight() {return image_height_;};
	uint32_t getImageWidth() {return image_width_;};

	border getBorderPixelROI();

//private:

	// get the distance betwewn Plane and Point
	float distancePktPlane(TofGround::planeHN cur_planeHN, Eigen::Vector3f &vec);
	// get random sample of num numbers from 1:N
	std::vector<int32_t> getRandomSample(int32_t N,int32_t num);
	// get a Plane from three Points
	TofGround::planeHN getplane(std::vector<int32_t> &active, float*X, float *Y, float*Z);
	// get all inliers
	std::vector<int32_t> getInlier(TofGround::planeHN cur_plane, float*X, float*Y, float*Z, int32_t N);
	// refine plane with orthogonal regression
	TofGround::planeHN refinePlane(std::vector<int32_t> & inliers);
	// Keep only the Points in ROI from the vector
	std::vector<TofGround::includepix> keepROIpoints(std::vector<TofGround::includepix>& allInRoi, std::vector<int32_t>& choosedVec);
	// Bucket Points
	void bucketPixel(int32_t max_features, float bucket_width, float bucket_height);
	// aplly ROI on image
	void applyROI(float* Iz, float* Ix = nullptr, float* Iy = nullptr);
	//Reduce Pixel by Random
	void bucketPixelRandom(int32_t num);
	//check if enought valid Points
	bool enoughtPoints(int32_t N);

private:
	TofGround() {};


	parameters param_;
	uint32_t image_height_;								// Image height
	uint32_t image_width_;								// Image width
	std::vector<TofGround::includepix> allPxInROI_;		// all Pixel in ROI
	std::vector<int32_t> inliers_;						// all inliers
	planeHN ebeneHN_;									// estimated groundplane
	std::vector<int> num_bucket_amout_;					// number of buckets and how many valid Points

	uint8_t* filtermask_{nullptr};						// filtermask
	cv::Mat filtermask_mat_;
	filtertyp filtertyp_ = none;

	float* X;							// Pointer X
	float* Y;							// Pointer Y
	float* Z;							// Pointer Z

};

}// namspace votof
