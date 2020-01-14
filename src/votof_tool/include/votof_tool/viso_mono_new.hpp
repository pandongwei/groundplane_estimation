/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#ifndef VISO_MONO_H
#define VISO_MONO_H

#include "viso2/viso.h"
#include "camera_models/camera_model.h"

namespace viso2 {

class VisualOdometryMonoNew : public VisualOdometry {

public:

  // monocular-specific parameters (mandatory: height,pitch)
  struct parameters : public VisualOdometry::parameters {
    double                      height;           // camera height above ground (meters)
    double                      pitch;            // camera pitch (rad, negative=pointing down)
    int32_t                     ransac_iters;     // number of RANSAC iterations
    double                      inlier_threshold; // fundamental matrix inlier threshold
    double                      motion_threshold; // directly return false on small motions
    parameters () {
      height           = 1.0;
      pitch            = 0.0;
      ransac_iters     = 1500;
      inlier_threshold = 0.00001;
      motion_threshold = 100.0;
    }
  };

  // constructor, takes as inpute a parameter structure
  VisualOdometryMonoNew(parameters param);
  VisualOdometryMonoNew(const VisualOdometryMonoNew&) = delete;
  
  // deconstructor
 ~VisualOdometryMonoNew ();
  
  // process a new image, pushs the image back to an internal ring buffer.
  // valid motion estimates are available after calling process for two times.
  // inputs: I ......... pointer to rectified image (uint8, row-aligned)
  //         dims[0] ... width of I
  //         dims[1] ... height of I
  //         dims[2] ... bytes per line (often equal to width)
  //         replace ... replace current image with I, without copying last current
  //                     image to previous image internally. this option can be used
  //                     when small/no motions are observed to obtain Tr_delta wrt
  //                     an older coordinate system / time step than the previous one.
  // output: returns false if motion too small or an error occured
  bool process (uint8_t *I,int32_t* dims,bool replace=false);

  bool process (uint8_t *I,int32_t* dims, Matrix n, double h, bool replace=false);
  bool process (uint8_t *I,int32_t* dims, Matrix n, double h, const std::unique_ptr<CameraModel> &CamModel, bool replace=false);

  bool process_ohne_skale (uint8_t *I, int32_t* dims, bool rplace=false);
  std::vector<double>	scale_pose(viso2::Matrix n, double h);
  bool process_with_Groundplane_estimation (uint8_t *I,int32_t* dims, double h, bool replace=false);


private:

  template<class T> struct idx_cmp {
    idx_cmp(const T arr) : arr(arr) {}
    bool operator()(const size_t a, const size_t b) const { return arr[a] < arr[b]; }
    const T arr;
  };  

  bool					updateMotion (viso2::Matrix n, double h);
  bool 					updateMotion (viso2::Matrix n, double h, const std::unique_ptr<CameraModel> &CamModel);
  bool					updateMotion_ohne_skale();
  bool 					updateMotion_with_Groundplane_estimation (double h);


  std::vector<double>  estimateMotion (std::vector<Matcher::p_match> p_matched);
  std::vector<double>  estimateMotion (std::vector<Matcher::p_match> p_matched ,viso2::Matrix n, double h, const std::unique_ptr<CameraModel> &CamModel);
  std::vector<double>  estimateMotion (std::vector<Matcher::p_match> p_matched, viso2::Matrix n, double h);
  std::vector<double>  estimateMotion_ohne_skale (std::vector<Matcher::p_match> p_matched);
  std::vector<double>  estimateMotion_with_Groundplane_estimation (std::vector<Matcher::p_match> p_matched, double h);


  Matrix               smallerThanMedian (Matrix &X,double &median);
  bool                 normalizeFeaturePoints (std::vector<Matcher::p_match> &p_matched,Matrix &Tp,Matrix &Tc);
  void                 fundamentalMatrix (const std::vector<Matcher::p_match> &p_matched,const std::vector<int32_t> &active,Matrix &F);
  void                 EtoRt(Matrix &E,Matrix &K,std::vector<Matcher::p_match> &p_matched,Matrix &X,Matrix &R,Matrix &t);
  int32_t              triangulateChieral (std::vector<Matcher::p_match> &p_matched,Matrix &K,Matrix &R,Matrix &t,Matrix &X);
  std::vector<int32_t> getInlier (std::vector<Matcher::p_match> &p_matched,Matrix &F);
  
  // parameters
  parameters param;  

  Matrix X_plane;
  Matrix R, t;
  double median;

public:
  std::vector<int32_t> alle_pkt_ebene;
};
}
#endif // VISO_MONO_H

