#include <../../include/tof_groundplane_est/tof_groundplane_estimation.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <camera_models/camera_model.h>
#include <camera_models/camera_model_non_svp.h>



namespace py = boost::python;
namespace np = boost::python::numpy;

using namespace votof;

namespace {

boost::python::list getNormalVector (TofGround& TG){
	TofGround::planeHN temp;	
	temp = TG.getGroundplane();
	
	boost::python::list lis;
	lis.append(temp.n(0));
	lis.append(temp.n(1));
	lis.append(temp.n(2));
	return lis;
}

double getPlaneDistance (TofGround& TG){
	double temp = TG.getPlaneDistance();
	return temp;
}


bool updateGroundplane(TofGround& TG, CameraModelNonSvp &CamModel ,boost::python::numpy::ndarray& img){
	
	int k = 0;
	std::unique_ptr<CameraModel> CamMod( new CameraModelNonSvp(CamModel) );
	float* input_ptr = reinterpret_cast<float*>(img.get_data());
	auto size = img.get_shape();
	float* I = new float[size[0]*size[1]];
	for(int i=0; i<size[0]; i++){
		for(int j=0; j<size[1]; j++){
			I[k]=input_ptr[k];
			k++;
		}
	}
	bool rtn = TG.estimateGroundplane(I, CamMod);
	delete[] I;
	return rtn;
}


}//namespace


BOOST_PYTHON_MODULE(PYTHON_API_MODULE_NAME) {

	Py_Initialize();
   	np::initialize();

	void (TofGround::*setROIx1)() = &TofGround::setROI;
	void (TofGround::*setROIx2)(cv::Mat&) = &TofGround::setROI;
	py::class_<TofGround, boost::noncopyable>("TofGround", py::init<TofGround::parameters, int, int>())
		.def("setROI", setROIx1)
		.def("setROI", setROIx2)
	;

	py::class_<TofGround::parameters>("parameters")
		.def_readwrite("ransac_iter" , &TofGround::parameters::ransac_iter)
		.def_readwrite("inlier_threshold" , &TofGround::parameters::inlier_threshold)
	;
	py::def("getNormalVector", getNormalVector);
	py::def("updateGroundplane", updateGroundplane);
	py::def("getPlaneDistance", getPlaneDistance);

}
