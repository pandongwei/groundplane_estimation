/*
 * estimate_homography.hpp
 *
 *  Created on: Aug 26, 2019
 *      Author: moritz
 */

#ifndef INCLUDE_TESTVIMO_TOOL_EST_HOMOGRAPHY_HPP_
#define INCLUDE_TESTVIMO_TOOL_EST_HOMOGRAPHY_HPP_

#include <opencv2/core/core.hpp>
#include <viso2/matcher.h>
#include <viso2/matrix.h>
#include <Eigen/Eigen>
#include <camera_models/camera_model.h>

namespace votof{

class homography_est{

public:

	struct calibration{
		double f;
		double cu;
		double cv;
	};

	struct parameters{
		calibration calib;
		int32_t ransac_iter;
		double inlier_threshold;
		parameters(){
			ransac_iter = 200;
			inlier_threshold = 10;
		}
	};

	enum filtermask_typ{
		MAT,
		ROWALIGN
	};

public:
	~homography_est();
	homography_est(parameters param, int32_t height, int32_t width);

	//set ROI in different situations
	bool setROI(uint32_t down_left_v, uint32_t down_left_u, uint32_t down_right_v, uint32_t down_right_u, uint32_t up_left_v, uint32_t up_left_u, uint32_t up_right_v, uint32_t up_right_u, int32_t height, int32_t width);
	bool setROI(cv::Mat &mask);
	bool setROI(uint8_t* mask, int32_t height, int32_t width);
	bool setROI(viso2::Matrix &mask);

	bool estimateH(std::vector<viso2::Matcher::p_match> &matches, bool use_filtermask = false);
	bool estimateH(std::vector<viso2::Matcher::p_match> &matches, std::unique_ptr<CameraModel>& CamModel, bool use_filtermask=false);

	std::vector<int32_t> getRandomSample (int32_t N,int32_t num);
	std::vector<int32_t> getInlier(std::vector<viso2::Matcher::p_match> &matches, std::vector<int32_t> &relevant_matches, std::vector<double> &h);

	viso2::Matrix getH_Matrix() {return H_;};
	std::vector<double> getH_Vector() {return h_;};
	std::vector<int32_t> getAllInliers() {return inliers_;};


private:

	homography_est(){};

	uint8_t* filtermask_ {nullptr};
	cv::Mat filtermask_mat_;

	filtermask_typ filter_typ;

	viso2::Matrix H_;
	std::vector<double> h_;
	std::vector<int32_t> inliers_;

	parameters param_;
	int32_t height_;
	int32_t width_;

};


}
#endif /* INCLUDE_TESTVIMO_TOOL_EST_HOMOGRAPHY_HPP_ */
