#pragma once

#include <vector>
#include "Eigen\Eigen"

#define MAX_COUNT 100

using namespace Eigen;
using namespace std;

class Model {
	const static int TOP_PEAK_PERCENT = 10;
	const static int PAGE_SIZE = 100;
	const static int SAMPLE_RATE = 20;
	const char* TRAINING_MODEL_FILE = "motorcycle.txt";

public:
	Model(const char* filename);
	double FindGaps(vector <double> peaks, vector <double> valleys);
	void Run(int time_interval);
	void WriteToFile(int index, double mean, double std);

protected:
	double GetMean(vector <double> xs);
	double GetStd(vector <double> xs);
	vector <RowVector3d> Open(const char* filename);

private:
	vector <RowVector3d> _original_data;
};