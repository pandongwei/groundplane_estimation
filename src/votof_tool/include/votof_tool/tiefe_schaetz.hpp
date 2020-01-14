/*
 * tiefe_schaetz.h
 *
 *  Created on: Jun 19, 2019
 *      Author: schnetz
 */

#ifndef INCLUDE_TESTVIMO_TOOL_TIEFE_SCHAETZ_HPP_
#define INCLUDE_TESTVIMO_TOOL_TIEFE_SCHAETZ_HPP_

#include "viso2/matcher.h"
#include "viso2/viso.h"
#include <cmath>

using namespace std;

namespace viso2{

class Tiefe {

public:


	struct parameters {
		double f, cu, cv, base;
	};


    struct p_match_depth {
        float u1p, v1p; 	// u,v-coordinates in previous left  image
        int32_t i1p;    	// feature index (for tracking)
        float u1c, v1c; 	// u,v-coordinates in current  left  image
        int32_t i1c;   	 	// feature index (for tracking)
        float l1p, l1c; 	// Abstand des Features
        p_match_depth() {
        }
        p_match_depth(float u1p,
                float v1p,
                int32_t i1p,
                float u1c,
                float v1c,
                int32_t i1c,
				float l1p,
				float l1c)
                : u1p(u1p), v1p(v1p), i1p(i1p), u1c(u1c), v1c(v1c), i1c(i1c), l1p(l1p), l1c(l1c){
        }
    };

	//constructor default
	Tiefe(parameters param);
	Tiefe(const Tiefe&) = delete;

	//Destructor
	~Tiefe(){};

	void estimateDepth(std::vector<Matcher::p_match> &p_matched);

    void setIntrinsics(double f, double cu, double cv, double base) {
        param.f = f;
        param.cu = cu;
        param.cv = cv;
        param.base = base;
    }


	std::vector<Tiefe::p_match_depth> getMatches(){
		return matches_und_tiefe;
	}

	void schreib_10_werte(std::vector<Tiefe::p_match_depth> &maches);

	parameters param;

private:

	std::vector<Tiefe::p_match_depth> matches_und_tiefe;

};

}

#endif /* INCLUDE_TESTVIMO_TOOL_TIEFE_SCHAETZ_HPP_ */
