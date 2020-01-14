#include "../include/votof_tool/votof_filter.hpp"



using namespace std;

namespace votof{
///////////////////////
//// Kalman-Filter ////
///////////////////////

//template<typename T>
//votof::KalmanFilter<T>::KalmanFilter(T dt, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &F, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &H,
//		const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &Q, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &R,
//		const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &P)
//: F(F), H(H), Q(Q), R(R), P(P), m(H.rows()), n(F.rows()),
//  dt(dt), initialized(false), I(n,n), x_est(n), x_est_neu(n)
//  {
//	I.setIdentity();
//	t0=0,t=0;
//  }
//
//
//template<typename T>
//void votof::KalmanFilter<T>::init(Eigen::Matrix<T,Eigen::Dynamic,1> &x0, T t0){
//
//	x_est = x0;
//	this->t0 = t0;
//	t=t0;
//	initialized = true;
//
//}
//
//template<typename T>
//void votof::KalmanFilter<T>::init(){
//
//	x_est.setZero();
//	t0 = 0;
//	t = t0;
//	initialized=true;
//}
//
//template<typename T>
//void votof::KalmanFilter<T>::update(const Eigen::Matrix<T,Eigen::Dynamic,1> &z){
//
//	if(!initialized){
//		cout<<"Filter nicht initializiert!"<<endl;
//		return;
//	}
//
//	x_est_neu = F* x_est;
//	P = F*P*F.transpose();
//	K = P*H.transpose() * (H*P*H.transpose()+R).inverse();
//	x_est_neu += K * (z - H*x_est_neu);
//	P = (I - K*H)*P;
//	x_est = x_est_neu;
//
//	t += dt;
//}
//
//template<typename T>
//void votof::KalmanFilter<T>::update(){
//
//	if(!initialized){
//		cout<<"Filter nicht initializiert!"<<endl;
//		return;
//	}
//
//	x_est_neu = F* x_est;
//	P = F*P*F.transpose();
//	x_est = x_est_neu;
//
//	t += dt;
//}
//
//
//template<typename T>
//void votof::KalmanFilter<T>::update(const Eigen::Matrix<T,Eigen::Dynamic,1> &z, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &F, T dt){
//
//	this->F = F;
//	if (dt!=0)
//		this->dt = dt;
//	update(z);
//}
//
//template<typename T>
//void votof::KalmanFilter<T>::update(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &F, T dt){
//
//	this->F = F;
//	if (dt!=0)
//		this->dt = dt;
//	update();
//}



////////////////////////////
//// Lineare Regression ////
////////////////////////////

LinearRegression::LinearRegression(uint32_t n, double dt)
: n(n), AA(n,n), b(n), dt(dt), initialize(false)
{
	AA.setZero();
	b.setZero();
	x_est.setZero();
//	A_save = new vector<Eigen::MatrixXd>;
//	b_save = new vector<Eigen::VectorXd>;
}

LinearRegression::~LinearRegression()
{
	//delete
}

void LinearRegression::init(uint32_t window_size, double t0)
{
	this->t0 = t0;
	t= t0;
	this->window_size = window_size;
	initialize=true;
}

void LinearRegression::init()
{
	this->t0  = 0;
	t = t0;
	this-> window_size = 50;
	initialize=true;
}


void LinearRegression::update(const Eigen::VectorXd& y, const Eigen::MatrixXd& A0)
{

	if(!initialize){
		cout<<"Filter nicht Initialisert!"<<endl;
		return;
	}

	AA = AA + A0;
	b = b + y;


	this->A_save.push_back(A0);
	this->b_save.push_back(y);

	if(A_save.size()>window_size)
	{
		AA-=A_save.front();
		b-=b_save.front();

		A_save.erase(A_save.begin());
		b_save.erase(b_save.begin());
	}

	if(AA.determinant() != 0)
	{
		x_est=AA.partialPivLu().solve(b);
	}else
	{
		cout<<"Matrix ist nicht invetierbar"<<endl;
		x_est=AA.colPivHouseholderQr().solve(b);
	}

}

void LinearRegression::update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd& A0)
{
	cout<<"noch implementieren"<<endl;
}


/*
 * 	A0 = 	1		0		0		ti		0		0
 * 			0		1		0		0		ti		0
 * 			0		0		1		0		0		ti
 * 			ti		0		0		ti*ti	0		0
 * 			0		ti		0		0		ti*ti	0
 * 			0		0		ti		0		0		ti*ti
 *
 *
 * 	b=		px
 * 			py
 * 			pz
 * 			px*ti
 * 			py*ti
 * 			pz*ti
 *
 *
 * 	x_est=	x
 * 			y
 * 			z
 * 			vx
 * 			vy
 * 			vz
 *
 *
 */

}//namespace
