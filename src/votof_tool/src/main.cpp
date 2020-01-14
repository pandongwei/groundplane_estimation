/*
 * main.cpp
 *
 *      Author: moritz
 */

//#include "../../../src/gnuplot-iostream/gnuplot-iostream.h"
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <viso2/matrix.h>
#include <viso2/viso.h>
#include <eigen3/Eigen/Dense>
#include <storage_container/storage_container.h>
#include <camera_models/camera_model.h>
#include <calib_storage/calib_storage.h>
#include "../include/votof_tool/viso_mono_new.hpp"
#include <tof_groundplane_est/tof_groundplane_estimation.hpp>
#include "../include/votof_tool/est_homography.hpp"
#include "../include/votof_tool/votof_filter.hpp"

using namespace cv;
using namespace std;
using namespace viso2;
using namespace votof;

//声明用到的函数。TofGround是一个class
void ermittel_male_bewegung(const Mat &I,const vector<vector<Matcher::p_match> > &all_matches, const vector<std::int32_t> &inliers );
int32_t finde(const vector<Matcher::p_match> wo_ist, const int32_t das);
void male_track(const Point p, Mat & Bild);
void male_ebene_und_normalenvektor(cv::Mat &I, const TofGround::planeHN &plane, const std::unique_ptr<CameraModel>& CamModel, int32_t u, int32_t v, double l,Scalar scal=0);
votof::TofGround::planeHN transformPlaneKos(const TofGround::planeHN &plane, Eigen::Affine3d& convMat);
viso2::Matrix correctPose (const viso2::Matrix& tr, Eigen::Vector3d n_akt, Eigen::Vector3d n_pre);

template <typename T>
float calcAngle(T& vec1 , T& vec2);

struct planeCheckfunc{                                      //定义新的数据结构，其实相当于python的类

	enum functionTyp{fromPreviousPlane, fromInitialPlane};

	double max_hight_difference_{0};	//check hight difference
	double max_roll_{0};				//check roll(Wanken) angle change
	double max_pitch_ {0};			//check pitch (Nicken) angle change
	double max_complete_{0};			//check the complete angle change between the two normal vectors

	planeCheckfunc(double max_hight_diff, double max_roll, double max_pitch, double max_complete):
		max_hight_difference_(max_hight_diff), max_roll_(max_roll), max_pitch_(max_pitch), max_complete_(max_complete) {};

	planeCheckfunc(functionTyp fucTyp){

		if(fucTyp == fromPreviousPlane){
			max_hight_difference_ = 0.1f;		//check hight difference
			max_roll_ = 0.939693f;				//check roll(Wanken) angle change - 20°
			max_pitch_ = 0.990268;			    //check pitch (Nicken) angle change - 10°
			max_complete_ = 0.866025f;			//check the complete angle change between the two normal vectors - 30°
		}
		if(fucTyp == fromInitialPlane){
			max_hight_difference_ = 0.30f;		//check hight difference
			max_roll_ = 0.89;				//check roll(Wanken) angle change - 45°
			max_pitch_ = 0.96;			   	//check pitch(Nicken) angle change - 30°
			max_complete_ = 0.5f;			//check the complete angle change between the two normal vectors - 60°
		}
	}

	planeCheckfunc();

	bool planeCheck(Eigen::Matrix<double,3,1>& n_pre, double h_pre, Eigen::Matrix<double,3,1>& n_akt, double h_akt){

		// max. angle in cos(x°)
		double max_hight_difference = max_hight_difference_;		//check hight difference
		double max_roll = max_roll_;								//check roll(Wanken) angle change
		double max_pitch = max_pitch_;			    			//check pitch (Nicken) angle change
		double max_complete = max_complete_;						//check the complete angle change between the two normal vectors
		//check hight difference
		if(abs(h_pre-h_akt) > max_hight_difference)
			return false;

		Eigen::Vector2d xy_pre = n_pre.head<2>();
		Eigen::Vector2d xy_akt = n_akt.head<2>();
		Eigen::Vector2d yz_pre = n_pre.tail<2>();
		Eigen::Vector2d yz_akt = n_akt.tail<2>();

		double roll = calcAngle(xy_pre, xy_akt);
		double pitch = calcAngle(yz_pre, yz_akt);
		double complete = calcAngle(n_pre, n_akt);

		if(roll <= max_roll)
			return false;
		if(pitch <= max_pitch)
			return false;
		if(complete <= max_complete)
			return false;

		return true;
	}

};



int main(int argc, char** argv){

	if (argc<5) {
		cerr<<"Bitte Pfad für Bilder, die Kalib.-Datei und der Farbkamera und Tof-Kamera angeben"<<endl;
		return 0;
	}

	string map_filename = argv[2];
	string cam_node = argv[3];
	string tof_node = argv[4];


	//load calibration
	cs::CalibStorage calSto(map_filename);
	calSto.showCameraNames();
	auto camModel_tof = calSto.getCameraModel(tof_node);
	auto camModel_cam = calSto.getCameraModel(cam_node);
	auto camModel_cam_undist = calSto.getCameraModel(cam_node+"_undistort");
	auto trafo_tof_to_cam = calSto.getTransformation(tof_node, cam_node);
	auto trafo_cam_to_tof = calSto.getTransformation(cam_node, tof_node);

	Eigen::Affine3d tof_to_cam_mat = trafo_tof_to_cam.src2dest;
	Eigen::Affine3d cam_to_tof_mat = trafo_cam_to_tof.src2dest;

	std::cout<<"Used Camera Model is: ";              //从camera的id来判断是哪种camera
	switch(static_cast<int>(camModel_cam->getId())){
		case 0: cout<<"CAMERA_MODEL_NON_SVP"	<<endl; break;
		case 1: cout<<"CAMERA_MODEL_NURBS"		<<endl; break;
		case 2: cout<<"CAMERA_MODEL_PINHOLE"	<<endl; break;
		case 3: cout<<"CAMERA_MODEL_SPHERE"		<<endl; break;
		case 4: cout<<"CAMERA_MODEL_SVP"		<<endl; break;
		case 5: cout<<"CAMERA_MODEL_EPIPOLAR"	<<endl; break;
		case 6: cout<<"CAMERA_MODEL_NON_SVP_GENERIC_DISTORTION"<<endl; break;
		case 7: cout<<"CAMERA_MODEL_FAST_LOOKUP"<<endl; break;
		case 8: cout<<"CAMERA_MODEL_NUM"		<<endl; break;
	}


	//undistort Mat
	cv::Mat Igray_undist;
	cv::Mat Irgb_undist;
	cv::Mat mapPix_cam, mapSubPix_cam;
	cv::Mat mapPix_tof, mapSubPix_tof;

	auto lut_cam = calSto.getRemapLUT("cam_undistort");
	cv::Mat map_u_cam = lut_cam.uMap;
	cv::Mat map_v_cam = lut_cam.vMap;
	cv::convertMaps(map_u_cam, map_v_cam, mapPix_cam, mapSubPix_cam, CV_16SC2, false);


	//Variablen um Bilder zu laden
	string pfad2 = argv[1];
	string bildTof;
	string bildVis;
	string bildRGB;
	char bildnummer[256];
	Mat Ivis;
	Mat Irgb;
	cv::FileStorage fs;


	// Tiefenschätzung parameter und konstruktor
	TofGround::parameters param_tof;
	param_tof.inlier_threshold = 0.001;
	param_tof.ransac_iter = 500;
	param_tof.bucket.bucket_height=25;
	param_tof.bucket.bucket_width=25;
	param_tof.bucket.max_features=50;
	TofGround tof(param_tof, 352, 287);
	tof.setROI();


	//VO Parameter und Konstruktor
	VisualOdometryMonoNew::parameters para;
	para.bucket.bucket_height= 30;
	para.bucket.bucket_width = 30;
	para.bucket.max_features = 2;
	para.ransac_iters = 1500;
	para.inlier_threshold = 0.00001;
	para.calib.f = 560;
	para.calib.cu = 512;
	para.calib.cv = 269;
	VisualOdometryMonoNew visoNew(para);
	Matrix poseVO = Matrix::eye(4);


	Eigen::MatrixXd F = Eigen::MatrixXd::Identity(3,3);
	F(2,2) = 0.5;

	Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3,3);;

	Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(3,3);
	Q(0,0) = 0.8*0.8;
	Q(1,1) = 0.8*0.8;
	Q(2,2) = 0.8*0.8;

	Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3,3);
	R(0,0) = 0.01*0.01;
	R(1,1) = 0.01*0.01;
	R(2,2) = 0.01*0.01;

	Eigen::MatrixXd P = Eigen::MatrixXd::Zero(3,3);
	P(0,0) = 1*1;
	P(1,1) = 1*1;
	P(2,2) = 1*1;

	votof::KalmanFilter<double> kalfi(0.1, F, H, Q, R, P);


	//Speichert alle letzten Maches der VO, um Punkt länger zu verfolgen		//TODO: Ändern, dass nur die aktuellen Matches
	vector<vector<Matcher::p_match> > all_last_matches;


	//Fenster um den Weg darzustellen
	cv::Mat wegVO;
	wegVO = Mat::zeros(600, 1200, CV_8UC3);


	//spichert gemessene Ebene von vorherigem Frame
	Eigen::Vector3d n_previous;
	double h_previous;

	Eigen::Vector3d n;
	double hoehe;

	//Normalenvector vom Start
	Eigen::Vector3d neins;
	double heins;
	bool first = true;

	fstream all_pos(pfad2+"convertMatrix/all/complete_path.dat", ios::out);
	char matname[256];
	std::string matname_path;

	planeCheckfunc planeCheckBetweenImage(planeCheckfunc::functionTyp::fromPreviousPlane);
	planeCheckfunc planeCheckFromStart(planeCheckfunc::functionTyp::fromInitialPlane);

	all_pos<<"Img.-Nr: | X | Y | Z"<<std::endl;

	for (int i=7; i>-1 ; i++){

		cout<<"****************************"<<endl;
		cout<<"Bild Nr.: "<<i<<endl;

		// Zeitmessung start
		double time1 = 0., time2 = 0., tstarttof, tstartvo, tstartbilder;
		tstartbilder = clock();


		// Bilder Einlesen
		sprintf(bildnummer,"_%05d", i);
		bildTof  = 	pfad2 + "D"  + bildnummer + ".yaml";
		bildVis  = 	pfad2 + "DV" + bildnummer + ".png";
		bildRGB  = 	pfad2 + "I"  + bildnummer + ".png";

		std::sprintf(matname, "_%05d-%05d", i, i-1);
		matname_path = pfad2 + "convertMatrix/Affine_I" + matname + ".dat";

		fs.open(bildTof, cv::FileStorage::READ);
		if (!fs.isOpened()){
			std::cerr<<"Failed to open"<<std::endl;
			break;
		}
		cv::Mat_<float> itof;
		fs["depth"] >> itof;
		fs.release();

		Ivis  = imread(bildVis, IMREAD_COLOR);
		Irgb  = imread(bildRGB, IMREAD_GRAYSCALE);

		//undistort Rgb-Image
		cv::remap(Irgb, Irgb_undist, mapPix_cam, mapSubPix_cam, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

		//Groundplane Estimation start
		
		//读取tof图像信息，并赋值给tof_data
		float* tof_data = (float*)malloc(itof.cols*itof.rows*sizeof(float));
		int32_t k=0;
		for(int32_t v=0; v<itof.rows; v++){
			for(int32_t u=0; u<itof.cols; u++){
				tof_data[k] = itof.at<float>(v,u);
				k++;
			}
		}

		if(first){
			bool tof_suc2 = tof.estimateGroundplane(tof_data, camModel_tof);
			if(tof_suc2){
				neins = transformPlaneKos(tof.getGroundplane(), tof_to_cam_mat).n;
				heins = transformPlaneKos(tof.getGroundplane(), tof_to_cam_mat).d;

				Eigen::VectorXd temp= Eigen::VectorXd::Zero(3);
				temp = Eigen::Ref<Eigen::VectorXd>(neins);
				kalfi.init(temp);
				std::cout<<temp<<std::endl;

			}else
				continue;
		}


		int32_t dimsRGB[] = {Irgb_undist.cols, Irgb_undist.rows, Irgb_undist.cols};

		viso2::Matrix n_viso(3,1);
		for(int32_t i=0; i<3; i++){
			n_viso.val[i][0]=n_previous(i);
		}

		bool vo_suc = visoNew.process(Irgb_undist.data, dimsRGB, n_viso, h_previous);

		if (vo_suc){		// VO

			cout<<" *** New Pose ***"<<endl;
			poseVO = poseVO * Matrix::inv(visoNew.getMotion());

			Eigen::Vector3d n_temp;
			Eigen::Matrix3d pose_temp;
			pose_temp(0,0)=poseVO.val[0][0];	pose_temp(0,1)=poseVO.val[0][1];	pose_temp(0,2)=poseVO.val[0][2];
			pose_temp(1,0)=poseVO.val[1][0];	pose_temp(1,1)=poseVO.val[1][1];	pose_temp(1,2)=poseVO.val[1][2];
			pose_temp(2,0)=poseVO.val[2][0];	pose_temp(2,1)=poseVO.val[2][1];	pose_temp(2,2)=poseVO.val[2][2];

			n_temp = pose_temp * n_previous;
			poseVO = correctPose(poseVO, n_temp, neins);

			// Male den Weg
			cv::Point p;
			p.x=(poseVO.val[0][3]);
			p.y=(poseVO.val[2][3]);
			male_track(p, wegVO);

		}
		if(!first)
			all_pos<<i<<" "<<poseVO.val[0][3]<<" "<<poseVO.val[1][3]<<" "<<poseVO.val[2][3]<<std::endl;



		Eigen::Vector3d n_akt;
		double h_akt;
		bool tof_suc = tof.estimateGroundplane(tof_data, camModel_tof);
		TofGround::planeHN temp;
		if(tof_suc && !first){
			n_akt = transformPlaneKos(tof.getGroundplane(), tof_to_cam_mat).n;
			h_akt = transformPlaneKos(tof.getGroundplane(), tof_to_cam_mat).d;

			tof_suc = planeCheckBetweenImage.planeCheck(n_previous, h_previous, n_akt, h_akt);
			if(!tof_suc)
				tof_suc = planeCheckFromStart.planeCheck(neins, heins , n_akt, h_akt);
		}

		viso2::Matrix poseVO = visoNew.getMotion();
		Eigen::MatrixXd pose_step(3,3);
		pose_step(0,0)=poseVO.val[0][0];	pose_step(0,1)=poseVO.val[0][1];	pose_step(0,2)=poseVO.val[0][2];
		pose_step(1,0)=poseVO.val[1][0];	pose_step(1,1)=poseVO.val[1][1];	pose_step(1,2)=poseVO.val[1][2];
		pose_step(2,0)=poseVO.val[2][0];	pose_step(2,1)=poseVO.val[2][1];	pose_step(2,2)=poseVO.val[2][2];


		if(!tof_suc && vo_suc){
			std::cout<<" Keine Höhe schätzbar"<<std::endl;

//			kalfi.update(pose_step);
			n = pose_step * n_previous;
			hoehe = h_previous;

		}else{
//			if(tof_suc && vo_suc){
//
//				viso2::Matrix poseVO = visoNew.getMotion();
//				Eigen::MatrixXd pose_step(3,3);
//				pose_step(0,0)=poseVO.val[0][0];	pose_step(0,1)=poseVO.val[0][1];	pose_step(0,2)=poseVO.val[0][2];
//				pose_step(1,0)=poseVO.val[1][0];	pose_step(1,1)=poseVO.val[1][1];	pose_step(1,2)=poseVO.val[1][2];
//				pose_step(2,0)=poseVO.val[2][0];	pose_step(2,1)=poseVO.val[2][1];	pose_step(2,2)=poseVO.val[2][2];
//
//				std::cout<<"was los "<<pose_step<<std::endl;
//
//				Eigen::VectorXd n_akt(3);
//				Eigen::Vector3d n_get = transformPlaneKos(tof.getGroundplane(),tof_to_cam_mat).n;
//				n_akt = Eigen::Ref<Eigen::VectorXd>(n_get);
//				hoehe = transformPlaneKos(tof.getGroundplane(), tof_to_cam_mat).d;
//				kalfi.update(n_akt, pose_step);
//			}
			n = transformPlaneKos(tof.getGroundplane(), tof_to_cam_mat).n;
			hoehe = transformPlaneKos(tof.getGroundplane(), tof_to_cam_mat).d;

		}

//		n = kalfi.getState();


		temp.n = n;
		temp.d = hoehe;

		std::cout<<"start printing"<<std::endl;
		viso2::Matrix saveMat(4,4);
		saveMat.eye();

		fstream save_mat(matname_path, ios::out);

		save_mat<<"Transformation matrix (no correction):"<<std::endl;

		if(vo_suc){
			saveMat = visoNew.getMotion();
		}

		save_mat<< saveMat.val[0][0]<<" "<<saveMat.val[0][1]<<" "<<saveMat.val[0][2]<<" "<<saveMat.val[0][3]<<" "<<std::endl<<
				saveMat.val[1][0]<<" "<<saveMat.val[1][1]<<" "<<saveMat.val[1][2]<<" "<<saveMat.val[1][3]<<" "<<std::endl<<
				saveMat.val[2][0]<<" "<<saveMat.val[2][1]<<" "<<saveMat.val[2][2]<<" "<<saveMat.val[2][3]<<" "<<std::endl<<
				saveMat.val[3][0]<<" "<<saveMat.val[3][1]<<" "<<saveMat.val[3][2]<<" "<<saveMat.val[3][3]<<" "<<std::endl;
		save_mat<<std::endl;


		save_mat<<"Transformation matrix inverted (no correction):"<<std::endl;
		saveMat = viso2::Matrix::inv(saveMat);

		save_mat<< saveMat.val[0][0]<<" "<<saveMat.val[0][1]<<" "<<saveMat.val[0][2]<<" "<<saveMat.val[0][3]<<" "<<std::endl<<
				saveMat.val[1][0]<<" "<<saveMat.val[1][1]<<" "<<saveMat.val[1][2]<<" "<<saveMat.val[1][3]<<" "<<std::endl<<
				saveMat.val[2][0]<<" "<<saveMat.val[2][1]<<" "<<saveMat.val[2][2]<<" "<<saveMat.val[2][3]<<" "<<std::endl<<
				saveMat.val[3][0]<<" "<<saveMat.val[3][1]<<" "<<saveMat.val[3][2]<<" "<<saveMat.val[3][3]<<" "<<std::endl;
		save_mat<<std::endl;

		save_mat<<"Transformation matrix invertes (corrected)"<<std::endl;

		Eigen::Vector3d n_temp;
		Eigen::Matrix3d pose_temp;
		pose_temp(0,0)=saveMat.val[0][0];	pose_temp(0,1)=saveMat.val[0][1];	pose_temp(0,2)=saveMat.val[0][2];
		pose_temp(1,0)=saveMat.val[1][0];	pose_temp(1,1)=saveMat.val[1][1];	pose_temp(1,2)=saveMat.val[1][2];
		pose_temp(2,0)=saveMat.val[2][0];	pose_temp(2,1)=saveMat.val[2][1];	pose_temp(2,2)=saveMat.val[2][2];

		n_temp = pose_temp * n;
		if(vo_suc)
			saveMat = correctPose(saveMat, n_temp, n_previous);

		save_mat<< saveMat.val[0][0]<<" "<<saveMat.val[0][1]<<" "<<saveMat.val[0][2]<<" "<<saveMat.val[0][3]<<" "<<std::endl<<
				saveMat.val[1][0]<<" "<<saveMat.val[1][1]<<" "<<saveMat.val[1][2]<<" "<<saveMat.val[1][3]<<" "<<std::endl<<
				saveMat.val[2][0]<<" "<<saveMat.val[2][1]<<" "<<saveMat.val[2][2]<<" "<<saveMat.val[2][3]<<" "<<std::endl<<
				saveMat.val[3][0]<<" "<<saveMat.val[3][1]<<" "<<saveMat.val[3][2]<<" "<<saveMat.val[3][3]<<" "<<std::endl;
		save_mat<<std::endl;


		save_mat<<"Transformation matrix complete (inverted):"<<std::endl;
		save_mat<< poseVO.val[0][0]<<" "<<poseVO.val[0][1]<<" "<<poseVO.val[0][2]<<" "<<poseVO.val[0][3]<<" "<<std::endl<<
				poseVO.val[1][0]<<" "<<poseVO.val[1][1]<<" "<<poseVO.val[1][2]<<" "<<poseVO.val[1][3]<<" "<<std::endl<<
				poseVO.val[2][0]<<" "<<poseVO.val[2][1]<<" "<<poseVO.val[2][2]<<" "<<poseVO.val[2][3]<<" "<<std::endl<<
				poseVO.val[3][0]<<" "<<poseVO.val[3][1]<<" "<<poseVO.val[3][2]<<" "<<poseVO.val[3][3]<<" "<<std::endl;
		save_mat<<std::endl;


		save_mat<<"plane normalvector:"<<std::endl;
		save_mat<<n(0)<<" "<<n(1)<<" "<<n(2)<<std::endl;
		save_mat<<std::endl;

		save_mat<<"plane hight:"<<std::endl;
		save_mat<<hoehe<<std::endl;
		save_mat<<std::endl;

		save_mat.close();


		// Zeitmessung für VO und Ebenenschätzung
		cout << "  time Image process = " << time1<< " sec."<<endl;
		cout << "  time Ebene  = " << time2 << " sec." << endl;
		time1 += clock() - tstartbilder;
		time2  = clock() - tstartvo;
		time1 = time1/CLOCKS_PER_SEC;
		time2 = time2/CLOCKS_PER_SEC;
		cout << "  time Prozess gesamt = " << time1 << " sec." << endl <<endl;


		// Färbe alle Inliers Blau in TOF-Bilder
		vector<int32_t> all_inliers = tof.getInlierList();
		std::vector<TofGround::includepix> all_inroi = tof.getAllPxInRoi();
		for (uint32_t i=0; i<all_inliers.size(); i++){
			Ivis.at<Vec3b>(Point(all_inroi[all_inliers[i]].u,all_inroi[all_inliers[i]].v)) = 255;
		}

		//TODO: Noch ändern (siehe oben)
		vector<Matcher::p_match> current_matches = visoNew.getMatches();
		if (current_matches.size()>0 || all_last_matches.empty())
			all_last_matches.push_back(current_matches);

		if(all_last_matches.size()>2){
			all_last_matches.erase(all_last_matches.begin());
		}
		//TODOENDE


		vector<std::int32_t> current_inliers = visoNew.getInlierIndices();

		cv::cvtColor(Irgb_undist, Irgb_undist, cv::COLOR_GRAY2BGR);		//Grau->Farbbild, um Fluss zu visualisueren
		ermittel_male_bewegung(Irgb_undist,all_last_matches, current_inliers);

		// Male eine Ebene und Normalenvektor in ein Bild
		male_ebene_und_normalenvektor(Ivis, tof.getGroundplane(), camModel_tof, 175, 60, 2);
		male_ebene_und_normalenvektor(Irgb_undist, temp, camModel_cam, 500, 370, 2);

		// Vergrößer TOF Bilder
		cv::resize(Ivis, Ivis, cv::Size(), 2, 2);


		//Zeige Bilder an
		imshow("Weg", wegVO);
		imshow("Tof-Tiefenbild", Ivis);
		imshow("Farbkamera dist.", Irgb);
		imshow("Farbkamera undist.", Irgb_undist);
		waitKey(2);

		// gibt eservierten Speicher frei,释放内存
		free(tof_data);

		//speichere aktuelle Ebenenschätzung;
		n_previous = n;
		h_previous = hoehe;


		time1 += clock() - tstartbilder;
		time1 = time1/CLOCKS_PER_SEC;

		first = false;
		cout << "  time alles zusammen     = " << time1 << " sec " << endl;

	} // for-Schleife Ende

	all_pos.close();

	cout<<" *** ENDE ***"<<endl;
	return 1;
}


// ********************
// **** FUNKTIONEN ****
// ********************


template <typename T>
float calcAngle(T& vec1 , T& vec2){
    //calculate the angle base on two vector
	float dot =  vec1.dot(vec2);
	float nenner = vec1.norm() * vec2.norm();

	float result = dot / nenner;
	return abs(result);
}


void ermittel_male_bewegung(const Mat &I,const vector<vector<Matcher::p_match> > &all_matches, const vector<std::int32_t> &inliers){

	for (uint32_t i=0; i<all_matches.back().size(); i++){		//für alle matches vom letzten Frame

		int32_t ii=i;

		Point p1;
		Point p2;
		p2.x=all_matches.back()[ii].u1p;
		p2.y=all_matches.back()[ii].v1p;   	//Vorheriger Punkt vom letzten Bild
		p1.x=all_matches.back()[ii].u1c;
		p1.y=all_matches.back()[ii].v1c;	//Aktueller Punkt vom letzten Bild

		vector<Point> alle_points;
		alle_points.push_back(p1);
		alle_points.push_back(p2);

		for(int32_t j=all_matches.size()-2; j>=0; j--){			//punkte rückwärts finden

			ii = finde(all_matches[j], all_matches[j+1][ii].i1p);
			if (ii<0){
				break;
			}

			Point p3;
			p3.x=all_matches[j][ii].u1c;
			p3.y=all_matches[j][ii].v1c;
			alle_points.push_back(p3);
		}

		if (all_matches.size()<2){
			continue;
		}

		Point beginn;
		beginn = alle_points.front();
		Point ende;
		ende = alle_points.back();

		Scalar farbe;
		Scalar farbe_punkt (255, 255, 0);
		int32_t xx = (beginn.x-ende.x);
		int32_t yy = (beginn.y-ende.y);
		int32_t abstand = std::sqrt((xx*xx)+(yy*yy));


		for(uint32_t j=0; j<inliers.size(); j++){
			if(inliers[j]==i){
				farbe_punkt=Scalar(255, 0, 255);
			}
		}

		if (true){

			if (xx >=0){
				farbe=Scalar(0,0,250);
			}else if(xx<0){
				farbe=Scalar(0,250,0);
			}else{
				farbe=Scalar(250,0,0);
			}

			cv::line(I, beginn, ende, farbe, 2, 8);
			cv::circle(I,beginn,1,farbe_punkt,2,8);
		}
	}
}


int32_t finde( vector<Matcher::p_match>  wo_ist,  int32_t das){

	for (uint32_t j=0; j<wo_ist.size(); j++){

		if (wo_ist[j].i1c==das){
			return j;
		}
	}
	return -1;
}


void male_track(const Point p, Mat & Bild){

	static Point pp{0,0};
	Point pc; pc.x=p.x+600; pc.y=p.y-400;
	pc.y=-pc.y;

	if(pp.x==0 && pp.y==0){
		pp=pc;
		return;
	}
	cv::line(Bild,pp,pc,Scalar(250,250,200),1,8);
	pp=pc;

}


void male_ebene_und_normalenvektor(cv::Mat &I, const TofGround::planeHN &plane, const std::unique_ptr<CameraModel> &CamModel, int32_t u, int32_t v, double l,Scalar scal){

	double n1 = plane.n(0);
	double n2 = plane.n(1);
	double n3 = plane.n(2);
	double d = plane.d;
	double s = 0;

	//Normalenvektro fester Bildpunkt u/v
	Eigen::Vector3d center_mal_xyz(0,0,0);
	Eigen::Vector3d center_mal_xyz_direction(0,0,0);
	Eigen::Vector2d center_mal_uv(0,0);
	Eigen::Vector3d ende_mal_xyz(0,0,0);
	Eigen::Vector2d ende_mal_uv(0,0);

	center_mal_uv(0)=u;
	center_mal_uv(1)=v;

	CamModel->getViewingRay(center_mal_uv, center_mal_xyz, center_mal_xyz_direction);
	s = (d-n1*center_mal_xyz(0)-n2*center_mal_xyz(1)-n3*center_mal_xyz(2))/(n1*center_mal_xyz_direction(0)+n2*center_mal_xyz_direction(1)+n3*center_mal_xyz_direction(2));
	center_mal_xyz = center_mal_xyz+s*center_mal_xyz_direction;

	ende_mal_xyz = -plane.n*1;
	ende_mal_xyz = ende_mal_xyz+center_mal_xyz;
	CamModel->getImagePoint(center_mal_xyz, center_mal_uv);
	CamModel->getImagePoint(ende_mal_xyz, ende_mal_uv);


	cv::line(I,cv::Point((int32_t)center_mal_uv(0), (int32_t)center_mal_uv(1)), cv::Point((int32_t)ende_mal_uv(0), (int32_t)ende_mal_uv(1)), Scalar(0 ,255,0), 2);

	//Male Ebene Ecke
	Eigen::Vector3d ol_xyz(0,0,0);
	Eigen::Vector3d or_xyz(0,0,0);
	Eigen::Vector3d ul_xyz(0,0,0);
	Eigen::Vector3d ur_xyz(0,0,0);
	Eigen::Vector2d ol_uv(0,0);
	Eigen::Vector2d or_uv(0,0);
	Eigen::Vector2d ul_uv(0,0);
	Eigen::Vector2d ur_uv(0,0);

	double z_richtung = 0.4;
	double x_richtung = 0.4;

	ol_xyz = center_mal_xyz;
	or_xyz = center_mal_xyz;
	ul_xyz = center_mal_xyz;
	ur_xyz = center_mal_xyz;

	ol_xyz(0) = ol_xyz(0) - x_richtung;
	ol_xyz(2) = ol_xyz(2) + z_richtung;
	ol_xyz(1) = (plane.d-ol_xyz(0)*plane.n(0)-ol_xyz(2)*plane.n(2))/plane.n(1);

	or_xyz(0) = or_xyz(0) + x_richtung;
	or_xyz(2) = or_xyz(2) + z_richtung;
	or_xyz(1) = (plane.d-or_xyz(0)*plane.n(0)-or_xyz(2)*plane.n(2))/plane.n(1);

	ul_xyz(0) = ul_xyz(0) - x_richtung;
	ul_xyz(2) = ul_xyz(2) - z_richtung;
	ul_xyz(1) = (plane.d-ul_xyz(0)*plane.n(0)-ul_xyz(2)*plane.n(2))/plane.n(1);

	ur_xyz(0) = ur_xyz(0) + x_richtung;
	ur_xyz(2) = ur_xyz(2) - z_richtung;
	ur_xyz(1) = (plane.d-ur_xyz(0)*plane.n(0)-ur_xyz(2)*plane.n(2))/plane.n(1);

	CamModel->getImagePoint(ol_xyz, ol_uv);
	CamModel->getImagePoint(or_xyz, or_uv);
	CamModel->getImagePoint(ul_xyz, ul_uv);
	CamModel->getImagePoint(ur_xyz, ur_uv);

	cv::line(I,cv::Point((int32_t)ol_uv(0), (int32_t)ol_uv(1)), cv::Point((int32_t)or_uv(0), (int32_t)or_uv(1)), Scalar(0 ,255,0), 1 );
	cv::line(I,cv::Point((int32_t)or_uv(0), (int32_t)or_uv(1)), cv::Point((int32_t)ur_uv(0), (int32_t)ur_uv(1)), Scalar(0 ,255,0), 1 );

	cv::line(I,cv::Point((int32_t)ur_uv(0), (int32_t)ur_uv(1)), cv::Point((int32_t)ul_uv(0), (int32_t)ul_uv(1)), Scalar(0 ,255,0), 1 );
	cv::line(I,cv::Point((int32_t)ul_uv(0), (int32_t)ul_uv(1)), cv::Point((int32_t)ol_uv(0), (int32_t)ol_uv(1)), Scalar(0 ,255,0), 1 );

	cv::line(I,cv::Point((int32_t)ul_uv(0), (int32_t)ul_uv(1)), cv::Point((int32_t)or_uv(0), (int32_t)or_uv(1)), Scalar(0 ,255,0), 1 );
	cv::line(I,cv::Point((int32_t)ur_uv(0), (int32_t)ur_uv(1)), cv::Point((int32_t)ol_uv(0), (int32_t)ol_uv(1)), Scalar(0 ,255,0), 1 );

	//Horizont
	Eigen::Vector3d horizont_xyz_l(0,0,0);
	Eigen::Vector2d horizont_uv_l(0,0);
	Eigen::Vector3d horizont_xyz_r(0,0,0);
	Eigen::Vector2d horizont_uv_r(0,0);
//	std::cout<<"pos 14"<<std::endl;

	horizont_xyz_l(0) = -1000;
	horizont_xyz_l(2) =  1000;
	horizont_xyz_r(0) =  1000;
	horizont_xyz_r(2) =  1000;
//	std::cout<<"pos 15"<<std::endl;

	horizont_xyz_l(1) = (plane.d-horizont_xyz_l(0)*plane.n(0)-horizont_xyz_l(2)*plane.n(2))/plane.n(1);
	horizont_xyz_r(1) = (plane.d-horizont_xyz_r(0)*plane.n(0)-horizont_xyz_r(2)*plane.n(2))/plane.n(1);

	CamModel->getImagePoint(horizont_xyz_l, horizont_uv_l);
	CamModel->getImagePoint(horizont_xyz_r, horizont_uv_r);
//	std::cout<<"pos 12"<<std::endl;


	if (scal.isReal())
		cv::line(I,cv::Point((int32_t)horizont_uv_l(0), (int32_t)horizont_uv_l(1)), cv::Point((int32_t)horizont_uv_r(0), (int32_t)horizont_uv_r(1)), Scalar(0 ,200,255), 2 );
	else
		cv::line(I,cv::Point((int32_t)horizont_uv_l(0), (int32_t)horizont_uv_l(1)), cv::Point((int32_t)horizont_uv_r(0), (int32_t)horizont_uv_r(1)), scal, 2 );

}

votof::TofGround::planeHN transformPlaneKos(const TofGround::planeHN &plane, Eigen::Affine3d& convMat){

	TofGround::planeHN transPlane;
	Eigen::Vector4d plane_vector(plane.n(0), plane.n(1), plane.n(2), -plane.d);
	Eigen::Vector4d plane_trans = convMat * plane_vector;

	transPlane.n = plane_trans.head(3);
	Eigen::Vector3d p(1,1,(-plane_trans(3)-plane_trans(0)-plane_trans(1))/plane_trans(2));

	if(p.dot(transPlane.n)>=0)
		transPlane.n = transPlane.n/(transPlane.n.squaredNorm());
	else
		transPlane.n = -transPlane.n/(transPlane.n.squaredNorm());

	transPlane.d = p.dot(transPlane.n);

	return transPlane;
}

viso2::Matrix correctPose(const viso2::Matrix& tr, Eigen::Vector3d n_akt, Eigen::Vector3d n_pre){

	Eigen::Vector3d v, n_akt_, n_pre_;
	double c, s;
	viso2::Matrix vx(4,4);
	vx.zero();
	viso2::Matrix R(4,4);
	viso2::Matrix newPose(4,4);

	n_akt_ = n_akt/n_akt.norm();
	n_pre_ = n_pre/n_pre.norm();

	v = n_akt_.cross(n_pre_);
	s = v.norm();
	c = n_akt_.dot(n_pre_);
	c = cos(acos(c));

	vx.val[0][0] =  0;		vx.val[0][1] = -v(2);	vx.val[0][2] =  v(1);
	vx.val[1][0] =  v(2);	vx.val[1][1] =  0;		vx.val[1][2] = -v(0);
	vx.val[2][0] = -v(1);	vx.val[2][1] =  v(0);	vx.val[2][2] =  0;

	R = viso2::Matrix::eye(4) + vx + vx*vx*((1-c)/(s*s));

	viso2::Matrix man(4,1);
	man.val[0][0] = n_akt(0);
	man.val[1][0] = n_akt(1);
	man.val[2][0] = n_akt(2);
	man.val[3][0] = 0;

	newPose = (R) * tr;
	newPose.val[0][3] = tr.val[0][3];
	newPose.val[1][3] = tr.val[1][3];
	newPose.val[2][3] = tr.val[2][3];

	return newPose;
}





