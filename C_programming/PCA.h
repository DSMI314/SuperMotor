#pragma once

#include "Eigen/Eigen"

using namespace Eigen;
using namespace std;

class PCA {
public:
	static VectorXd Compute(MatrixXd D);
};
