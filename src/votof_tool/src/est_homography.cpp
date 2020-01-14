/*
 * estimate_homography.cpp
 *
 *  Created on: Aug 26, 2019
 *      Author: moritz
 */

#include "../include/votof_tool/est_homography.hpp"

using namespace std;
using namespace votof;

homography_est::homography_est(parameters param, int32_t height, int32_t width) : param_(param), height_(height), width_(width){

}


homography_est::~homography_est(){

	if(this->filtermask_!= nullptr){
		delete[] this->filtermask_;
	}
}


bool homography_est::setROI(uint32_t down_left_v, uint32_t down_left_u, uint32_t down_right_v, uint32_t down_right_u,
						uint32_t up_left_v, uint32_t up_left_u, uint32_t up_right_v, uint32_t up_right_u,
						int32_t height, int32_t width){

	if(down_left_v<up_left_v || down_right_v<up_right_v || down_left_v>down_right_v || up_left_u>up_right_u || height<=0 || width<=0){

		cout<<"ROI not vaild"<<endl;
		return false;
	}

	cv::Mat roi(height, width, CV_8UC1);
	cv::Point conture[1][4];

	conture[0][0] = cv::Point(down_left_u, down_left_v);
	conture[0][1] = cv::Point(down_right_u, down_right_v);
	conture[0][3] = cv::Point(up_left_u, up_left_v);
	conture[0][2] = cv::Point(up_right_u, up_right_v);

	const cv::Point* pts[1] = {conture[0]};
	int ntp[] = {4};
	cv::fillPoly(roi, pts, ntp, 1, cv::Scalar::all(255), 8);

	this->filtermask_mat_ = roi;
	this->filter_typ = MAT;
	return true;
}


bool homography_est::setROI(cv::Mat &mask){

	this->filtermask_mat_ = mask;
	this->filter_typ = MAT;
	return true;
}


bool homography_est::setROI(uint8_t* mask, int32_t height, int32_t width){

	this->filtermask_ = new uint8_t[width*height];

	for (int32_t i=0; i<(width*height); i++)
		this->filtermask_[i]=mask[i];

	this->filter_typ = ROWALIGN;
	return true;
}


bool homography_est::setROI(viso2::Matrix &mask){

	int32_t height = mask.m;
	int32_t width = mask.n;

	this->filtermask_ = new uint8_t[height*width];

	uint8_t k=0;
	for(int32_t v=0; v<height; v++){
		for(int32_t u=0; u<width; u++){
			if(mask.val[v][u] == 0){
				this->filtermask_[k] = 0;
			}else{
				this->filtermask_[k] = 1;
			}
			k++;
		}
	}
	this->filter_typ = ROWALIGN;
	return true;
}


bool homography_est::estimateH(std::vector<viso2::Matcher::p_match> &matches, bool use_filtermask){

	if (filtermask_==nullptr && filtermask_mat_.empty()){
		cout<<"Invalid Filtermask. Please set Filtermask"<<endl;
		return false;
	}

	if (matches.size()<10){
		cout<<"Not enough matches"<<endl;
		return false;
	}


	// Liste mit allen relevanten Matches durch Filtermaske
	std::vector<int32_t> relevant_matches;

	// Filtermaske anwenden
	if(use_filtermask==true){
		if(this->filter_typ == ROWALIGN){
			for(uint32_t i=0; i<matches.size(); i++){
				int32_t pos = (matches[i].u1c+width_*matches[i].v1c);
				if(filtermask_[pos] != 0)
					relevant_matches.push_back(i);
			}
		} else {
			if(this->filter_typ == MAT){
				for(uint32_t j=0; j<matches.size(); j++){
					if(filtermask_mat_.at<uint8_t>(matches[j].v1c, matches[j].u1c) != 0)
						relevant_matches.push_back(j);
				}
			}
		}
	}else{
		for(uint32_t i=0; i<matches.size(); i++)
			relevant_matches.push_back(i);
	}

	//schätze Homographymatrix
	int32_t N = relevant_matches.size();
	this->inliers_.clear();



	// Matrix A und h zur Lösung mit SVD
	int32_t anzahl_punkte_svd = 4; 		// muss >=4 sein
	viso2::Matrix A(anzahl_punkte_svd*2,9);
	viso2::Matrix h(9,1);

	for(int32_t k=0; k<param_.ransac_iter; k++){

		vector<int32_t> active = getRandomSample(N, anzahl_punkte_svd);

		for(uint32_t i=0; i<active.size(); i++){
			// Optimierungsmatrix
//			A.val[0+i*2][0] = -matches[relevant_matches[active[i]]].u1c;
//			A.val[0+i*2][1] = -matches[relevant_matches[active[i]]].v1c;
//			A.val[0+i*2][2] = -1;
//			A.val[0+i*2][3] =  0;
//			A.val[0+i*2][4] =  0;
//			A.val[0+i*2][5] =  0;
//			A.val[0+i*2][6] =  matches[relevant_matches[active[i]]].u1c*matches[relevant_matches[active[i]]].u1p;
//			A.val[0+i*2][7] =  matches[relevant_matches[active[i]]].v1c*matches[relevant_matches[active[i]]].u1p;
//			A.val[0+i*2][8] =  matches[relevant_matches[active[i]]].u1p;
//
//			A.val[1+i*2][0] =  0;
//			A.val[1+i*2][1] =  0;
//			A.val[1+i*2][2] =  0;
//			A.val[1+i*2][3] = -matches[relevant_matches[active[i]]].u1c;
//			A.val[1+i*2][4] = -matches[relevant_matches[active[i]]].v1c;
//			A.val[1+i*2][5] = -1;
//			A.val[1+i*2][6] =  matches[relevant_matches[active[i]]].u1c*matches[relevant_matches[active[i]]].v1p;
//			A.val[1+i*2][7] =  matches[relevant_matches[active[i]]].v1c*matches[relevant_matches[active[i]]].v1p;
//			A.val[1+i*2][8] =  matches[relevant_matches[active[i]]].v1p;


			//mit K
			A.val[0+i*2][0] = -(matches[relevant_matches[active[i]]].u1c-param_.calib.cu)/param_.calib.f;
			A.val[0+i*2][1] = -(matches[relevant_matches[active[i]]].v1c-param_.calib.cv)/param_.calib.f;
			A.val[0+i*2][2] = -1;
			A.val[0+i*2][3] =  0;
			A.val[0+i*2][4] =  0;
			A.val[0+i*2][5] =  0;
			A.val[0+i*2][6] =  ((matches[relevant_matches[active[i]]].u1c-param_.calib.cu)/param_.calib.f)*((matches[relevant_matches[active[i]]].u1p-param_.calib.cu)/param_.calib.f);
			A.val[0+i*2][7] =  ((matches[relevant_matches[active[i]]].v1c-param_.calib.cv)/param_.calib.f)*((matches[relevant_matches[active[i]]].u1p-param_.calib.cu)/param_.calib.f);
			A.val[0+i*2][8] =  (matches[relevant_matches[active[i]]].u1p-param_.calib.cu)/param_.calib.f;

			A.val[1+i*2][0] =  0;
			A.val[1+i*2][1] =  0;
			A.val[1+i*2][2] =  0;
			A.val[1+i*2][3] = -(matches[relevant_matches[active[i]]].u1c-param_.calib.cu)/param_.calib.f;
			A.val[1+i*2][4] = -(matches[relevant_matches[active[i]]].v1c-param_.calib.cv)/param_.calib.f;
			A.val[1+i*2][5] = -1;
			A.val[1+i*2][6] =  ((matches[relevant_matches[active[i]]].u1c-param_.calib.cu)/param_.calib.f)*((matches[relevant_matches[active[i]]].v1p-param_.calib.cv)/param_.calib.f);
			A.val[1+i*2][7] =  ((matches[relevant_matches[active[i]]].v1c-param_.calib.cv)/param_.calib.f)*((matches[relevant_matches[active[i]]].v1p-param_.calib.cv)/param_.calib.f);
			A.val[1+i*2][8] =  (matches[relevant_matches[active[i]]].v1p-param_.calib.cv)/param_.calib.f;

		}
		viso2::Matrix U, S, V;
		A.svd(U, S, V);

		//h = letzte Spalte des V-Vektors
		h = V.getMat(0,8,8,8);
		vector<double> h_vec;
		for(int32_t j=0; j<9; j++){
			h_vec.push_back(h.val[0][j]);
		}

		std::vector<int32_t> current_inliers;
		current_inliers = getInlier(matches, relevant_matches, h_vec);

		if(current_inliers.size()>=this->inliers_.size()){
			this->inliers_=current_inliers;
			this->h_=h_vec;
		}
	}
	return true;
}


std::vector<int32_t> homography_est::getRandomSample (int32_t N,int32_t num){

	  // init sample and totalset
	  std::vector<int32_t> sample;
	  std::vector<int32_t> totalset;

	  // create vector containing all indices
	  for (int32_t i=0; i<N; i++)
	    totalset.push_back(i);

	  // add num indices to current sample
	  sample.clear();
	  for (int32_t i=0; i<num; i++) {
	    int32_t j = rand()%totalset.size();
	    sample.push_back(totalset[j]);
	    totalset.erase(totalset.begin()+j);
	  }

	  // return sample
	  return sample;
}


std::vector<int32_t> homography_est::getInlier(std::vector<viso2::Matcher::p_match> &matches, std::vector<int32_t> &relevant_matches, std::vector<double> &h){

	std::vector<int32_t> inliers;
	int32_t N = relevant_matches.size();

	viso2::Matrix H(3,3);
	for(int32_t i=0; i<3; i++){
		for(int32_t j=0; j<3; j++){
			H.val[i][j] = h[(i*3)+j];
		}
	}
	viso2::Matrix HT = H;
	viso2::Matrix::inv(HT);

	viso2::Matrix x_p(3,1);
	viso2::Matrix x_c(3,1);

	for(int32_t i=0; i<N; i++){

//		x_p.val[0][0] = matches[relevant_matches[i]].u1p;
//		x_p.val[1][0] = matches[relevant_matches[i]].v1p;
//		x_p.val[2][0] = 1;
//		x_c.val[0][0] = matches[relevant_matches[i]].u1c;
//		x_c.val[1][0] = matches[relevant_matches[i]].v1c;
//		x_c.val[2][0] = 1;

		// mit Calibration
		x_p.val[0][0] = (matches[relevant_matches[i]].u1p-param_.calib.cu)/param_.calib.f;
		x_p.val[1][0] = (matches[relevant_matches[i]].v1p-param_.calib.cv)/param_.calib.f;
		x_p.val[2][0] = 1;
		x_c.val[0][0] = (matches[relevant_matches[i]].u1c-param_.calib.cu)/param_.calib.f;
		x_c.val[1][0] = (matches[relevant_matches[i]].v1c-param_.calib.cv)/param_.calib.f;
		x_c.val[2][0] = 1;

		double e = ((x_p-H*x_c).l2norm())+((HT*x_p-x_c).l2norm());

		if(e < param_.inlier_threshold){
			inliers.push_back(relevant_matches[i]);
		}
	}
	return inliers;
}


