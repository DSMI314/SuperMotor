#define _CRT_SECURE_NO_WARNINGS

#include "Model.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <deque>

using namespace std;

Model::Model(const char* filename) {
	this->_original_data = Open(filename);
}

/** @brief: Calculate gaps between peaks and valleys.

  * Pick top "TOP_PEAK_PERCENT" % of peaks and valleys to generate gaps by overall mean.

  * @param peaks: peaks list
  * @param valleys: valleys list

  * @return: the result "gap"
  **/
double Model::FindGaps(vector <double> peaks, vector <double> valleys) {
	int pos = PAGE_SIZE * TOP_PEAK_PERCENT / 100;

	int peak_pos = min(pos, (int)peaks.size());
	int valley_pos = min(pos, (int)valleys.size());

	double peak_ave = accumulate(peaks.end() - peak_pos, peaks.end(), 0.0) / peak_pos;
	double valley_ave = accumulate(valleys.begin(), valleys.begin() + valley_pos, 0.0) / valley_pos;

	return peak_ave - valley_ave;
}

/** @brief: Write features to the file.

  * Create the file and write down.

  * @param index: the axis which we should consider
  * @param mean: the average of gaps
  * @param std: the stardard deviation of gaps

  * @return:
  **/
void Model::WriteToFile(int index, double mean, double std) {
	FILE* fp = fopen(Model::TRAINING_MODEL_FILE, "w");
	fprintf(fp, "%d\n", index);
	fprintf(fp, "%f\n", mean);
	fprintf(fp, "%f\n", std);
	fclose(fp);
}

/** @brief: Build the model.

  * Consider the first interval time to generate the model by getting one axis where the maximum gaps we got.

  * @param time_interval: consider # seconds of data

  * @return:
  **/
void Model::Run(int time_interval) {
	// get the end mark
	int end_pos = min(int(_original_data.size()), time_interval * SAMPLE_RATE);

	// initialize
	double now_max_gap = 0.0;
	int now_max_gap_index = -1;
	vector <double> now_gaps = vector <double>();

	// loop for X, Y, Z axes.
	for (int axis_index = 0; axis_index < 3; axis_index++) {
		// slice the particular axis data
		vector <double> buffer;
		for (int j = 0; j < end_pos; j++)
			buffer.emplace_back(_original_data[j](axis_index));

		vector <double> valleys = vector <double>();
		vector <double> peaks = vector <double>();

		// generate needed information for the first "PAGE_SIZE" data
		assert(_original_data.size() > PAGE_SIZE);
		for (int j = 1; j < PAGE_SIZE - 1; j++) {
			if (buffer[j] > buffer[j - 1] && buffer[j] > buffer[j + 1])
				peaks.emplace_back(buffer[j]);
			if (buffer[j] < buffer[j - 1] && buffer[j] < buffer[j + 1])
				valleys.emplace_back(buffer[j]);
		}
		sort(valleys.begin(), valleys.end());
		sort(peaks.begin(), peaks.end());

		vector <double> gaps;
		gaps.emplace_back(FindGaps(peaks, valleys));

		// simulate the sliding window on "buffer"
		for (int j = PAGE_SIZE; j < end_pos; j++) {
			int s = j - PAGE_SIZE + 1;
			if (buffer[s] > buffer[s - 1] && buffer[s] > buffer[s + 1])
				peaks.erase(find(peaks.begin(), peaks.end(), buffer[s]));
			if (buffer[s] < buffer[s - 1] && buffer[s] < buffer[s + 1])
				valleys.erase(find(valleys.begin(), valleys.end(), buffer[s]));

			int e = j - 1;
			if (buffer[e] > buffer[e - 1] && buffer[e] > buffer[e + 1])
				peaks.insert(lower_bound(peaks.begin(), peaks.end(), buffer[e]), buffer[e]);
			if (buffer[e] < buffer[e - 1] && buffer[e] < buffer[e + 1])
				valleys.insert(lower_bound(valleys.begin(), valleys.end(), buffer[e]), buffer[e]);
			gaps.emplace_back(FindGaps(peaks, valleys));
		}

		// update feature if we get useful one
		double gap = GetMean(gaps);
		printf("%lf\n", gap);
		if (gap > now_max_gap) {
			now_max_gap = gap;
			now_max_gap_index = axis_index;
			now_gaps = gaps;
		}
	}
	// output important features to the file.
	vector <double> gaps = now_gaps;
	double mean = GetMean(gaps);
	printf("!! %lf\n", mean);
	double std = GetStd(gaps);
	WriteToFile(now_max_gap_index, mean, std);
	printf("!!!!!!!! %d !!!!!!!!!!!!\n", now_max_gap_index);
}

/** @brief: Calculate average of a list.

  * @param xs: list

  * @return: the average of list
  **/
double Model::GetMean(vector <double> xs) {
	return accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
}

/** @brief: Calculate standard deviation of a list.

  * @param xs: list

  * @return: the standard deviation of list
  **/
double Model::GetStd(vector <double> xs) {
	double mean = GetMean(xs);
	double sum = 0.0;
	int len = xs.size();
	for (int i = 0; i < len; i++)
		sum += xs[i] * xs[i];
	return sqrt(sum / xs.size() - mean * mean);
}

/** @brief: Read data

  * Open the data file to read and parse them.

  * @param filename: the file name of data we put

  * @return: tuple (X,Y,Z) list
  **/
vector <RowVector3d> Model::Open(const char* filename) {
	FILE* fp;
	errno_t err = fopen_s(&fp, filename, "r");
	if (err != 0)
		return vector <RowVector3d>();

	vector <RowVector3d> V;
	char* line = new char[MAX_COUNT];
	while ((fgets(line, MAX_COUNT, fp)) != 0) {
		char* token = strtok(line, ",");
		int arr[3] = { 0 };
		for (int i = 0; i < 3; i++) {
			token = strtok(NULL, ",");
			arr[i] = atoi(token);
		}
		V.emplace_back(RowVector3d(arr[0], arr[1], arr[2]));
	}
	return V;
}
