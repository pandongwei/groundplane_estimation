/*
 * velocity_estimate.hpp
 *
 *  Created on: Jul 4, 2019
 *      Author: schnetz
 */

#ifndef INCLUDE_TESTVIMO_TOOL_VTOF_FILTER_HPP_
#define INCLUDE_TESTVIMO_TOOL_VTOF_FILTER_HPP_

#include <iostream>
#include <cmath>
#include <Eigen/Eigen>
#include <vector>

namespace votof{

template <typename T>
class KalmanFilter{

public:

	KalmanFilter(T dt, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &F, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &H,
			const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &Q, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &R,
			const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &P);

	KalmanFilter();
	~KalmanFilter(){};

	void init(Eigen::Matrix<T,Eigen::Dynamic,1> &x0, T t0 = 0);
	void init();

	void update();
	void update(const Eigen::Matrix<T,Eigen::Dynamic,1> &y);
	void update(const Eigen::Matrix<T,Eigen::Dynamic,1> &y, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &A, T dt=0);
	void update(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &F, T dt);

	Eigen::Matrix<T,Eigen::Dynamic,1> getState() {return x_est;};
	T getTime() {return t;};


private:

	Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> F, H, Q, R, P, K;

	int32_t m, n;
	T t0, t;
	T dt;
	bool initialized;

	Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> I;
	Eigen::Matrix<T,Eigen::Dynamic,1> x_est, x_est_neu;

};


class LinearRegression{


public:

	LinearRegression(uint32_t n, double dt);
	LinearRegression();
	~LinearRegression();

	void init(uint32_t window_size ,double t0=0);
	void init();

	void update(const Eigen::VectorXd& y, const Eigen::MatrixXd& A0);
	void update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd& A0);

	Eigen::VectorXd getState() {return x_est;};
	double getTime() {return t;};


private:

	uint32_t n;
	Eigen::MatrixXd AA;
	Eigen::VectorXd b;
	std::vector<Eigen::MatrixXd > A_save;
	std::vector<Eigen::VectorXd > b_save;

	uint32_t window_size;
	double t0, t;
	double dt;
//	double t_window;
	bool initialize;

	Eigen::VectorXd x_est;


};
}

template<typename T>
votof::KalmanFilter<T>::KalmanFilter(T dt, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &F, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &H,
		const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &Q, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &R,
		const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &P)
: F(F), H(H), Q(Q), R(R), P(P), m(H.rows()), n(F.rows()),
  dt(dt), initialized(false), I(n,n), x_est(n), x_est_neu(n)
  {
	I.setIdentity();
	t0=0,t=0;
  }


template<typename T>
void votof::KalmanFilter<T>::init(Eigen::Matrix<T,Eigen::Dynamic,1> &x0, T t0){

	x_est = x0;
	this->t0 = t0;
	t=t0;
	initialized = true;

}

template<typename T>
void votof::KalmanFilter<T>::init(){

	x_est.setZero();
	t0 = 0;
	t = t0;
	initialized=true;
}

template<typename T>
void votof::KalmanFilter<T>::update(const Eigen::Matrix<T,Eigen::Dynamic,1> &z){

	if(!initialized){
		std::cout<<"Filter nicht initializiert!"<<std::endl;
		return;
	}

	std::cout<<"F = "<<std::endl<<F<<std::endl;
	std::cout<<"x_est = "<<std::endl<<x_est<<std::endl;

	x_est_neu = F* x_est;
	P = F*P*F.transpose();
	K = P*H.transpose() * (H*P*H.transpose()+R).inverse();
	x_est_neu += K * (z - H*x_est_neu);
	P = (I - K*H)*P;
	x_est = x_est_neu;

	t += dt;
}

template<typename T>
void votof::KalmanFilter<T>::update(){

	if(!initialized){
		std::cout<<"Filter nicht initializiert!"<<std::endl;
		return;
	}

	x_est_neu = F* x_est;
	P = F*P*F.transpose();
	x_est = x_est_neu;

	t += dt;
}


template<typename T>
void votof::KalmanFilter<T>::update(const Eigen::Matrix<T,Eigen::Dynamic,1> &z, const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &F, T dt){

	this->F = F;
	if (dt!=0)
		this->dt = dt;
	update(z);
}

template<typename T>
void votof::KalmanFilter<T>::update(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &F, T dt){

	this->F = F;
	if (dt!=0)
		this->dt = dt;
	update();
}



#endif

