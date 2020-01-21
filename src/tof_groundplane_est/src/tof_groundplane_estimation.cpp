#include "../include/tof_groundplane_est/tof_groundplane_estimation.hpp"
#include "camera_models/camera.h"


using namespace votof;

TofGround::TofGround(parameters param, uint32_t image_width, uint32_t image_height): param_(param), image_height_(image_height), image_width_(image_width){

}



TofGround::~TofGround(){

	if (this->filtermask_ != nullptr){
		delete[] this->filtermask_;
	}
}

bool TofGround::estimateGroundplane(float* I, std::unique_ptr<CameraModel>& CamModel, bool use_min_number){

    inliers_.clear();

    // Alle Pixel in ROI zu allPxInROI hinzufügen
	applyROI(I);

    //Wenn weniger als x Punkte mit Tiefe return false
    int32_t N = allPxInROI_.size();  // N 为在ROI中的深度点数量
    if (N<6) {
    	std::cout<<"Not enough Points for estimate Groundplane"<<std::endl;
    	return false;
    }

    // Bucket Pixel
    bucketPixel(param_.bucket.max_features,  param_.bucket.bucket_width, param_.bucket.bucket_height);
    N = allPxInROI_.size();  // allPxInROI_ ROI中所有的点

    if(use_min_number){
    	if(!enoughtPoints(N)){
    		std::cout<<"Conditions not fulfilled"<<std::endl;
    		return false;
    	}
    }


    X	= new float[N];
    Y   = new float[N];
    Z   = new float[N];

    for (int32_t i=0; i<N; i++) { // 对每一点操作

    	float s=0;
        Eigen::Vector2d vector_uv(0,0);
        Eigen::Vector3d vector_xyz(0,0,0);
        Eigen::Vector3d vector_xyz_direction(0,0,0);

    	vector_uv(0) = allPxInROI_[i].u;
    	vector_uv(1) = allPxInROI_[i].v;

    	CamModel->getViewingRay(vector_uv, vector_xyz, vector_xyz_direction); // @suppress("Invalid arguments")
    	s = (allPxInROI_[i].depth-vector_xyz(2))/vector_xyz_direction(2);

    	Z[i] = allPxInROI_[i].depth;
    	X[i] = vector_xyz(0)+s*vector_xyz_direction(0);
    	Y[i] = vector_xyz(1)+s*vector_xyz_direction(1);
    }
    // 这里用的是RANSAC方法，多次估计平面，选取最好的那个
    TofGround::planeHN cur_plane;
    for (int32_t k=0; k<param_.ransac_iter; k++) {

    	// draw random sample set
    	std::vector<int32_t> active = getRandomSample(N,3);

    	// estimate plane matrix and get inliers TODO
    	cur_plane = getplane(active, X, Y, Z);
    	std::vector<int32_t> inliers_curr = getInlier(cur_plane, X, Y, Z, this->allPxInROI_.size());

    	// update model if we are better
    	if (inliers_curr.size()>inliers_.size()){  // inlier的数量决定平面拟合的质量
    		this->inliers_ = inliers_curr;
    		this->ebeneHN_ = cur_plane;
    	}
    }

    if(this->inliers_.size() < 3){
    	std::cout<<"Not enough inliers"<<std::endl;
    	delete[] X;
    	delete[] Y;
    	delete[] Z;
    	return false;
    }

    // refine Plane
    this->ebeneHN_ = refinePlane(this->inliers_);

    delete[] X;
    delete[] Y;
    delete[] Z;

	return true;
}


bool TofGround::estimateGroundplane(float* Ix, float* Iy, float* Iz, bool use_min_number){

    inliers_.clear();

    // Alle Pixel in ROI zu allPxInROI hinzufügen
	applyROI(Iz, Ix, Iy);

    //Wenn weniger als x Punkte mit Tiefe return false
    int32_t N = allPxInROI_.size();
    if (N<6) {
    	std::cout<<"Not enough Points for estimate Groundplane"<<std::endl;
    	return false;
    }

    // Bucket Pixel
    bucketPixel(param_.bucket.max_features,  param_.bucket.bucket_width, param_.bucket.bucket_height);
    N = allPxInROI_.size();

    if(use_min_number){
    	if(!enoughtPoints(N)){
    		std::cout<<"Conditions not fulfilled"<<std::endl;
    		return false;
    	}
    }

    X = new float[N];
    Y = new float[N];
    Z = new float[N];


    for (int32_t i=0; i<N; i++) {

    	Z[i] = allPxInROI_[i].depth;
//    	uint32_t num = allPxInROI_[i].u+allPxInROI_[i].v*image_width_;
    	X[i] = allPxInROI_[i].x;
    	Y[i] = allPxInROI_[i].y;
//    	X[i] = Ix[num];
//    	Y[i] = Iy[num];
    }

    TofGround::planeHN cur_plane;
    for (int32_t k=0; k<param_.ransac_iter; k++) {

    	// draw random sample set
    	std::vector<int32_t> active = getRandomSample(N,3);

    	// estimate plane matrix and get inliers
    	cur_plane = getplane(active, X, Y, Z);
    	std::vector<int32_t> inliers_curr = getInlier(cur_plane, X, Y, Z, this->allPxInROI_.size());

    	// update model if we are better
    	if (inliers_curr.size()>inliers_.size()){
    		this->inliers_ = inliers_curr;
    		this->ebeneHN_ = cur_plane;
    	}
    }

    if(this->inliers_.size() < 3){
    	std::cout<<"Not enough inliers"<<std::endl;
    	delete[] X;
    	delete[] Y;
    	delete[] Z;
    	return false;
    }

    // refine Plane
    this->ebeneHN_ = refinePlane(this->inliers_);

    delete[] X;
    delete[] Y;
    delete[] Z;

	return true;
}


bool TofGround::estimateGroundplane(float* I, bool use_min_number){

    // Alle Pixel in ROI zu allPxInROI hinzufügen
	applyROI(I);

	//Check if parameter f, cu and cv are set
    if(param_.calib.f==-1 || param_.calib.cu==-1 || param_.calib.cv==-1){
    	std::cout<<"Erorr: Prameter f or/and cu or/and not set"<<std::endl;
    	return false;
    }

    //Wenn weniger als x Punkte mit Tiefe return false
    int32_t N = allPxInROI_.size();
    if (N<6) {
    	std::cout<<"Not enough Points for estimate Groundplane"<<std::endl;
    	return false;
    }

    // Bucket Pixel
    bucketPixel(param_.bucket.max_features,  param_.bucket.bucket_width, param_.bucket.bucket_height);
    N = allPxInROI_.size();

    if(use_min_number){
    	if(!enoughtPoints(N)){
    		std::cout<<"Conditions not fulfilled"<<std::endl;
    		return false;
    	}
    }

    X = new float[N];
    Y = new float[N];
    Z = new float[N];


    for (int32_t i=0; i<N; i++) {

    	Z[i] = allPxInROI_[i].depth;
    	X[i] = Z[i]*(allPxInROI_[i].u-param_.calib.cu)/param_.calib.f;
    	Y[i] = Z[i]*(allPxInROI_[i].v-param_.calib.cv)/param_.calib.f;
    }

    inliers_.clear();
    TofGround::planeHN cur_plane;
    for (int32_t k=0; k<param_.ransac_iter; k++) {

    	// draw random sample set
    	std::vector<int32_t> active = getRandomSample(N,3);

    	// estimate plane matrix and get inliers
    	cur_plane = getplane(active, X, Y, Z);
    	std::vector<int32_t> inliers_curr = getInlier(cur_plane, X, Y, Z, this->allPxInROI_.size());

    	// update model if we are better，即inlier更多的时候
    	if (inliers_curr.size()>inliers_.size()){
    		this->inliers_ = inliers_curr;
    		this->ebeneHN_ = cur_plane;
    	}
    }

    if(this->inliers_.size() < 3){
    	std::cout<<"Not enough inliers"<<std::endl;
    	delete[] X;
    	delete[] Y;
    	delete[] Z;
    	return false;
    }

    // refine Plane
    this->ebeneHN_ = refinePlane(this->inliers_);

    delete[] X;
    delete[] Y;
    delete[] Z;

	return true;
}


// Returns a Plane from three given Points 随机抽取的三点，估计平面
TofGround::planeHN TofGround::getplane(std::vector<int32_t> &active, float* X, float* Y, float* Z){

//	TofGround::plane plane_temp;
	TofGround::planeHN plane_temp2;
	Eigen::Vector3d a, b, c;
	Eigen::Vector3d n, nn;
    // 三个点a,b,c
	a(0) = X[active[0]];
	a(1) = Y[active[0]];
	a(2) = Z[active[0]];

	b(0) = X[active[1]];
	b(1) = Y[active[1]];
	b(2) = Z[active[1]];

	c(0) = X[active[2]];
	c(1) = Y[active[2]];
	c(2) = Z[active[2]];

	n = (b-a).cross((c-a));  // vector3d.cross 叉乘
    // n 为垂直于这三个点所形成平面的矢量
	if(a.dot(n)>=0)
		nn=n*(1/n.squaredNorm());  // squaredNorm 二范数,这里是要获得往上的归一化法向量
	else
		nn=(n*(1/n.squaredNorm()))*(-1);

	plane_temp2.n = nn;
	plane_temp2.d = a.dot(nn); // d 是坐标系到点a的水平距离

	return plane_temp2;
}


// Returns a vector with all inliers inside the given threshold
std::vector<int32_t> TofGround::getInlier(TofGround::planeHN cur_plane, float* X, float* Y, float *Z, int32_t N){

	std::vector<int32_t> inl;
	float d;
	Eigen::Vector3f x(0,0,0);

	for(int32_t i=0; i<N; i++){

		if(Z[i] < 0.1) // 太近的点不要
			continue;

		x(0) = X[i];
		x(1) = Y[i];
		x(2) = Z[i];
		d = distancePktPlane(cur_plane, x); // 估计每个点与现在估计的平面的距离，小于阈值的话为inlier

		if (abs(d)<=param_.inlier_threshold){
			inl.push_back(i);
		}
	}
	return inl;
}


// Returns a random set
std::vector<int32_t> TofGround::getRandomSample(int32_t N,int32_t num) {

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


// Calculating the distance between a Point and a Plane
float TofGround::distancePktPlane(TofGround::planeHN cur_plane, Eigen::Vector3f &vec){

	float distance;
	distance = vec.dot(cur_plane.n.cast<float>());
	distance = distance - cur_plane.d;
	return (distance);
}


// Refine Plane with all Inliers with Orthogonal Regression
TofGround::planeHN TofGround::refinePlane(std::vector<int32_t> & inliers){

	TofGround::planeHN refinedPlane;
	int32_t N = inliers.size();
	float x=0, y=0, z=0; 	 		//x, y, z Mittelwert

	Eigen::MatrixXf mat(N,3);
	Eigen::Vector3f n(0,0,0);
	Eigen::Vector3f p(0,0,0);

	for(int32_t i=0; i<N; i++){
		x+=X[inliers[i]];
		y+=Y[inliers[i]];
		z+=Z[inliers[i]];
	}
	x=x/(float)N;
	y=y/(float)N;
	z=z/(float)N;
	Eigen::Vector3f xyz(x, y, z);

	for(int32_t i=0; i<N; i++){
		mat(i,0) = X[inliers[i]]-x;
		mat(i,1) = Y[inliers[i]]-y;
		mat(i,2) = Z[inliers[i]]-z;
	}

	Eigen::BDCSVD svd(mat, Eigen::ComputeFullV);
	Eigen::MatrixXf V = svd.matrixV();

	n = V.col(2);

	float d = n.dot(xyz);
	p(1)=1;
	p(2)=1;
	p(0)=(d - n(1)*p(1) - n(2)*p(2)) / n(0);

	if(p.dot(n)>=0)
		n=n*(1/n.squaredNorm());
	else
		n=(n*(1/n.squaredNorm()))*(-1);

	d = p.dot(n);


	refinedPlane.d=d;
	refinedPlane.n=n.cast<double>();

	return refinedPlane;
}


std::vector<TofGround::includepix> TofGround::keepROIpoints(std::vector<TofGround::includepix>& allInRoi, std::vector<int32_t>& choosedVec){

	std::vector<TofGround::includepix> temp;

	for (uint32_t i=0; i<choosedVec.size(); i++){
		temp.push_back(allInRoi[choosedVec[i]]);
	}
	return temp;
}


void TofGround::bucketPixel(int32_t max_features, float bucket_width, float bucket_height) {

    // find max values
    float u_max = 0;
    float v_max = 0;
    float u_min = MAXFLOAT;
    float v_min = MAXFLOAT;
    for (std::vector<TofGround::includepix>::iterator it = allPxInROI_.begin(); it != allPxInROI_.end(); it++) {
        if (it->u > u_max)
            u_max = it->u;
        if (it->v > v_max)
            v_max = it->v;
        if (it->u < u_min)
        	u_min = it->u;
        if (it->v < v_min)
        	v_min = it->v;
    }

    // allocate number of buckets needed
    int32_t bucket_cols = (int32_t)floor(u_max / bucket_width) + 1;
    int32_t bucket_rows = (int32_t)floor(v_max / bucket_height) + 1;
    std::vector<includepix>* buckets = new std::vector<includepix>[bucket_cols * bucket_rows];

    this->num_bucket_amout_.clear();
    this->num_bucket_amout_.resize(bucket_cols*bucket_rows, 0);

    // assign matches to their buckets
    for (std::vector<includepix>::iterator it = allPxInROI_.begin(); it != allPxInROI_.end(); it++) {
        int32_t u = (int32_t)floor(it->u / bucket_width);
        int32_t v = (int32_t)floor(it->v / bucket_height);
        buckets[v * bucket_cols + u].push_back(*it);
    }

    // refill allPxInROI from buckets
    allPxInROI_.clear();
    for (int32_t i = 0; i < bucket_cols * bucket_rows; i++) {

        // shuffle bucket indices randomly
        std::random_shuffle(buckets[i].begin(), buckets[i].end());

        // add up to max_features features from this bucket to p_matched
        int32_t k = 0;
        for (std::vector<includepix>::iterator it = buckets[i].begin(); it != buckets[i].end(); it++) {
        	allPxInROI_.push_back(*it);
            k++;
            if (k >= max_features)
                break;
        }
        // safe how the number of buckets an how many valid Points in it
        this->num_bucket_amout_[i] = k;
    }

    // free buckets
    delete[] buckets;
}

void TofGround::setROI(){

	if(param_.roi.down_left.v < param_.roi.up_left.v || param_.roi.down_right.v < param_.roi.up_right.v || param_.roi.down_left.u > param_.roi.down_right.u || param_.roi.up_left.u > param_.roi.up_right.u){
		std::cout<<"ROI not valid"<<std::endl;
		return;
	}

	// Filtermaske für ROI initialisieren
	cv::Mat roi(this->image_height_, this->image_width_, CV_8U);
	roi = 0;
	cv::Point conture[1][4];

	conture[0][0] = cv::Point(param_.roi.down_left.u, param_.roi.down_left.v);
	conture[0][1] = cv::Point(param_.roi.down_right.u, param_.roi.down_right.v);
	conture[0][2] = cv::Point(param_.roi.up_right.u, param_.roi.up_right.v);
	conture[0][3] = cv::Point(param_.roi.up_left.u, param_.roi.up_left.v);

	const cv::Point* pts[1] = {conture[0]};
	int ntp[] = {4};

	cv::fillPoly(roi, pts, ntp, 1, cv::Scalar::all(255), 8);
	cv::imshow("roi", roi);

	this->filtertyp_ = mat;
	this->filtermask_mat_=roi;
}

void TofGround::setROI(Point down_left, Point down_right, Point up_left, Point up_right){


	if(down_left.v < up_left.v || down_right.v < up_right.v || down_left.u > down_right.u || up_left.u > up_right.u){
		std::cout<<"ROI not valid"<<std::endl;
		return;
	}

	// Filtermaske für ROI initialisieren
	cv::Mat roi(this->image_height_, this->image_width_, CV_8U);
	roi=0;
	cv::Point conture[1][4];

	conture[0][0] = cv::Point(down_left.u, down_left.v);
	conture[0][1] = cv::Point(down_right.u, down_right.v);
	conture[0][2] = cv::Point(up_right.u, up_right.v);
	conture[0][3] = cv::Point(up_left.u, up_left.v);

	const cv::Point* pts[1] = {conture[0]};
	int ntp[] = {4};

	cv::fillPoly(roi, pts, ntp, 1, cv::Scalar::all(255), 8);
	cv::imshow("roi", roi);

	this->filtertyp_ = mat;
	this->filtermask_mat_=roi;

}

void TofGround::setROI(uint8_t* filtermask, uint32_t pointer_size){

	if(pointer_size != (this->image_height_* this->image_width_)){
		std::cout<<"Pointer needs the same number of entries as the image has pixels"<<std::endl;
		return;
	}

	this->filtermask_ = new uint8_t[pointer_size];
	for(int32_t i=0; i<pointer_size; i++)
		this->filtermask_[i]=filtermask[i];


	this->filtertyp_ = pointer;
	this->filtermask_ = filtermask;
}


void TofGround::setROI(cv::Mat& filtermask){

	if(filtermask.cols != this->image_height_ && filtermask.rows != this->image_width_){
		std::cout<<"The input Filtermask needs to have the same size like the image"<<std::endl;
		return;
	}

	this->filtertyp_ = mat;
	this->filtermask_mat_ = filtermask;
}

void TofGround::applyROI(float* Iz, float*Ix, float* Iy){

	allPxInROI_.clear();
	int32_t i=0;

	if(filtertyp_== none){

		for (uint32_t v=0; v<image_height_; v++){
			for (uint32_t u=0; u<image_width_; u++){
				if(Iz[i]>param_.filter_min_z && Iz[i]<=param_.filter_max_z){
					includepix temp;
					temp.u = u;
					temp.v = v;
					temp.depth = Iz[i];
					if(Ix != nullptr && Iy != nullptr){
						if(Ix[i] <= param_.filter_min_x || Ix[i] >= param_.filter_max_x || Iy[i] <= param_.filter_min_y || Iy[i] >= param_.filter_max_y){
							i++;
							continue;
						}
						temp.x = Ix[i];
						temp.y = Iy[i];
					}
					this->allPxInROI_.push_back(temp);
				}
			}
			i++;
		}
	} else {
		if(filtertyp_ == mat){
			for (uint32_t v=0; v<image_height_; v++){
				for (uint32_t u=0; u<image_width_; u++){
					if(this->filtermask_mat_.at<uint8_t>(v,u)>=1 && Iz[i]>param_.filter_min_z && Iz[i]<=param_.filter_max_z){
						includepix temp;
						temp.u = u;
						temp.v = v;
						temp.depth = Iz[i];
						if(Ix != nullptr && Iy != nullptr){
							if(Ix[i] <= param_.filter_min_x || Ix[i] >= param_.filter_max_x || Iy[i] <= param_.filter_min_y || Iy[i] >= param_.filter_max_y){
								i++;
								continue;
							}
							temp.x = Ix[i];
							temp.y = Iy[i];
						}

						this->allPxInROI_.push_back(temp);
					}
					i++;
				}
			}
		} else {
			if (filtertyp_ == pointer){

				for (uint32_t v=0; v<image_height_; v++){
					for (uint32_t u=0; u<image_width_; u++){
						if(this->filtermask_[i]>=1 && Iz[i]>param_.filter_min_z && Iz[i]<=param_.filter_max_z){
							includepix temp;
							temp.u = u;
							temp.v = v;
							temp.depth = Iz[i];
							if(Ix != nullptr && Iy != nullptr){
								if(Ix[i] <= param_.filter_min_x || Ix[i] >= param_.filter_max_x || Iy[i] <= param_.filter_min_y || Iy[i] >= param_.filter_max_y){
									i++;
									continue;
								}
								temp.x = Ix[i];
								temp.y = Iy[i];
							}
							this->allPxInROI_.push_back(temp);
						}
						i++;
					}
				}
			}
		}
	}
}

void TofGround::bucketPixelRandom(int32_t num){

	int32_t N = this->allPxInROI_.size();

	if(N<=num)
		return;

	std::vector<int32_t> used_Random = getRandomSample(N, 4000);
	this->allPxInROI_ = keepROIpoints(this->allPxInROI_, used_Random);
}

bool TofGround::enoughtPoints(int32_t N){
    // 点数量不够，就false
	if (N < param_.min_Points_in_ROI)
		return false;
    // 判断统计有用的点
	int32_t valid_buckets = 0;
	for (std::vector<int>::iterator it = this->num_bucket_amout_.begin(); it != this->num_bucket_amout_.end(); it++){
		if (*it >= param_.min_Points_in_bucket){
			valid_buckets++;
		}
	}
    // 有用的点数量也不够，就false
	if (valid_buckets < param_.min_used_bucket)
		return false;

	return true;
}


TofGround::border TofGround::getBorderPixelROI(){

	TofGround::border temp;
	temp.u_max = 0;
	temp.u_min = UINT32_MAX;
	temp.v_max = 0;
	temp.v_min = UINT32_MAX;

	for(uint32_t i = 0; i<inliers_.size(); i++){

		if(temp.u_max < allPxInROI_[inliers_[i]].u)
			temp.u_max = allPxInROI_[inliers_[i]].u;

		if(temp.u_min > allPxInROI_[inliers_[i]].u)
			temp.u_min = allPxInROI_[inliers_[i]].u;

		if(temp.v_max < allPxInROI_[inliers_[i]].v)
			temp.v_max = allPxInROI_[inliers_[i]].v;

		if(temp.v_min > allPxInROI_[inliers_[i]].v)
			temp.v_min = allPxInROI_[inliers_[i]].v;
	}
	return temp;
}


TofGround::cornerPixel TofGround::getCornerPixel(float* X, float* Y, float* Z){

	TofGround::cornerPixel temp;

	uint32_t u_min = UINT32_MAX;
	uint32_t u_max = 0;
	uint32_t v_min = UINT32_MAX;
	uint32_t v_max = 0;
	uint32_t u_, v_;

	std::vector<int32_t> inliers_;

	inliers_ = getInlier(this->ebeneHN_, X, Y, Z, this->image_width_*this->image_height_);

	for(uint32_t i=0; i<inliers_.size(); i++){

		u_ = inliers_[i]%image_width_;
		v_ = inliers_[i]/image_width_;

		if(u_ > u_max)
			u_max = u_;
		if(u_ < u_min)
			u_min = u_;
		if(v_ > v_max)
			v_max = v_;
		if(v_ < v_min)
			v_min = v_;
	}

	float dis_ol = FLT_MAX;
	float dis_or = FLT_MAX;
	float dis_ur = FLT_MAX;
	float dis_ul = FLT_MAX;


	for(uint32_t i=0; i<inliers_.size(); i++){

		u_ = inliers_[i]%image_width_;
		v_ = inliers_[i]/image_width_;

		if(pow((u_min-u_),2)+pow((v_min-v_),2) < dis_ol){
			dis_ol = pow((u_min-u_),2)+pow((v_min-v_),2);
			temp.u_ol = u_; temp.v_ol = v_;
			temp.x_ol = X[inliers_[i]];
			temp.y_ol = Y[inliers_[i]];
			temp.z_ol = Z[inliers_[i]];
		}
		if(pow((u_min-u_),2)+pow((v_max-v_),2) < dis_or){
			dis_or = pow((u_min-u_),2)+pow((v_max-v_),2);
			temp.u_or = u_; temp.v_or = v_;
			temp.x_or = X[inliers_[i]];
			temp.y_or = Y[inliers_[i]];
			temp.z_or = Z[inliers_[i]];
		}
		if(pow((u_max-u_),2)+pow((v_max-v_),2) < dis_ur){
			dis_ur = pow((u_max-u_),2)+pow((v_max-v_),2);
			temp.u_ur = u_; temp.v_ur = v_;
			temp.x_ur = X[inliers_[i]];
			temp.y_ur = Y[inliers_[i]];
			temp.z_ur = Z[inliers_[i]];
		}
		if(pow((u_max-u_),2)+pow((v_min-v_),2) < dis_ul){
			dis_ul = pow((u_max-u_),2)+pow((v_min-v_),2);
			temp.u_ul = u_; temp.v_ul = v_;
			temp.x_ul = X[inliers_[i]];
			temp.y_ul = Y[inliers_[i]];
			temp.z_ul = Z[inliers_[i]];
		}
	}
	return temp;
}



