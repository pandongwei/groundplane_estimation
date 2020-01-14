/*
 * tiefe_schaetz.cpp
 *
 *  Created on: Jun 19, 2019
 *      Author: schnetz
 */

#include "../include/votof_tool/tiefe_schaetz.hpp"

using namespace std;

namespace viso2{


Tiefe::Tiefe(parameters param) : param(param){

}


void Tiefe::estimateDepth(std::vector<Matcher::p_match> &p_matched){


	if(p_matched.size()==0){
		return;

	}

	matches_und_tiefe.clear();

	cout<<"estimateDepth"<<endl;
	cout<<p_matched.size()<<endl;

	std::vector<int32_t> active;

	double max_entfernung = 50;
	double min_entfernung = 0.3;

	for(int32_t i=0; i<(int32_t)p_matched.size(); i++){

		matches_und_tiefe.push_back({p_matched[i].u1p, p_matched[i].v1p, p_matched[i].i1p,
									p_matched[i].u1c, p_matched[i].v1c, p_matched[i].i1c,
									0,0});

		double Z;

		double d = max(p_matched[i].u1p - p_matched[i].u2p, 0.0001f);

		Z = (param.f*param.base/d);

		double X = (matches_und_tiefe[i].u1p-param.cu)*param.base/d;
		double Y = (matches_und_tiefe[i].v1p-param.cv)*param.base/d;

		double tiefe = Z/(cos(atan(sqrt(pow(p_matched[i].u1p,2)+pow(p_matched[i].v1p,2))/param.f)));
		matches_und_tiefe[i].l1p  = tiefe;

//		cout<<"u1: "<<matches_und_tiefe[i].u1p<<"  v: "<<matches_und_tiefe[i].v1p<<"  l: "<<matches_und_tiefe[i].l1p<<"  Z: "<<Z<<"  X: "<<X<<"  Y: "<<Y<<endl;

		if(matches_und_tiefe[i].l1p > max_entfernung || matches_und_tiefe [i].l1p < min_entfernung){

			active.push_back(i);
		}
	}
	cout<<"active: "<<active.size()<<endl;
	cout<<"estimateDepth ende"<<endl;

}


void Tiefe::schreib_10_werte(std::vector<Tiefe::p_match_depth> &maches){

	for(int32_t i=0; i<10; i++){

		cout<<"p_allg.: "<<matches_und_tiefe[i].u1p<<" | "<<matches_und_tiefe[i].v1p<<endl;
		cout<<"p_insd.: "<<maches[i].u1p<<" | "<<maches[i].v1p<<endl<<endl;
		cout<<"l: "<<maches[i].l1c<<" | "<<maches[i].l1p<<endl;

	}
}



}
