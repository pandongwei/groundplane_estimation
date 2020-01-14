/*



*/

#ifndef VISO_TOF_H
#define VISO_TOF_H

#include "viso2/viso.h"
#include "../include/votof_tool/tiefe_schaetz.hpp"
#include "viso2/matcher.h"

using namespace std;

namespace viso2 {

class VisualOdometryTof : public VisualOdometry {

public:

  // stereo-specific parameters (mandatory: base)
  struct parameters : public VisualOdometry::parameters {

	Tiefe::parameters tiefe;
    int32_t ransac_iters;     	// number of RANSAC iterations
    double  inlier_threshold; 	// fundamental matrix inlier threshold
    bool    reweighting; 		// lower border weights (more robust to calibration errors)
    double  base;				// base nur temporär für Tof ersatz
    parameters () {
      ransac_iters     = 200;
      inlier_threshold = 2.0;
      reweighting      = true;
      base			   = 1.0;

    }
  };

  // constructor, takes as inpute a parameter structure
  VisualOdometryTof (parameters param);
  VisualOdometryTof(const VisualOdometryTof&) = delete;

  // deconstructor
  ~VisualOdometryTof ();
  
  // process a new images, push the images back to an internal ring buffer.
  // valid motion estimates are available after calling process for two times.
  // inputs: I1 ........ pointer to rectified left image (uint8, row-aligned)
  //         I2 ........ pointer to rectified right image (uint8, row-aligned)
  //         dims[0] ... width of I1 and I2 (both must be of same size)
  //         dims[1] ... height of I1 and I2 (both must be of same size)
  //         dims[2] ... bytes per line (often equal to width)
  //         replace ... replace current images with I1 and I2, without copying last current
  //                     images to previous images internally. this option can be used
  //                     when small/no motions are observed to obtain Tr_delta wrt
  //                     an older coordinate system / time step than the previous one.
  // output: returns false if an error occured
  // Itof..... pointer to Tof Image form ????

  bool process (uint8_t *I1, uint8_t *Itof,int32_t* dims,bool replace=false);


  std::vector<Tiefe::p_match_depth> getMatchesDeep(){
	  return tiefer->getMatches();
  }		// !!! neu hier

  std::vector<Matcher::p_match> getMatches(){
	  return matcher->getMatches();
  }

  using VisualOdometry::process;


protected:
  bool updateMotion();		// !!! Von viso in viso_tof

  Tiefe									*tiefer;			//feature tiefe
  std::vector<Tiefe::p_match_depth>		p_matched_tiefe;	//feture point matches + tiefe


private:

  std::vector<double>  estimateMotion (std::vector<Matcher::p_match> p_matched){
	  std::vector<double> a;
	  return a;};
  std::vector<double>  estimateMotionTof (std::vector<Tiefe::p_match_depth> p_matched_tiefe, std::vector<Matcher::p_match> p_matched);
  enum                 result { UPDATED, FAILED, CONVERGED };  
  result               updateParameters(std::vector<Matcher::p_match> &p_matched,std::vector<int32_t> &active,std::vector<double> &tr,double step_size,double eps);
  void                 computeObservations(std::vector<Matcher::p_match> &p_matched,std::vector<int32_t> &active);
  void                 computeResidualsAndJacobian(std::vector<double> &tr,std::vector<int32_t> &active);
  std::vector<int32_t> getInlier(std::vector<Matcher::p_match> &p_matched,std::vector<double> &tr);

  double *X,*Y,*Z;    // 3d points
  double *p_residual; // residuals (p_residual=p_observe-p_predict)
  
  // parameters
  parameters param;

};

}

#endif // VISO_TOF_H

