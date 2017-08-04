#define _CRT_SECURE_NO_WARNINGS

#include "Model.h"
#include <algorithm>
#include <functional>
#include <numeric>

using namespace std;

Model::Model(const char* filename) {
	this->_original_data = Open(filename);
}

vector <double> Model::FindPeaksSorted(vector <double> xs, int ratio = TOP_PEAK_PERCENT) {
	int pagesize = xs.size();
	vector <double> res;
	for (int i = 1; i < pagesize - 1; i++) {
		if (xs[i] > xs[i - 1] && xs[i] > xs[i + 1])
			res.emplace_back(xs[i]);
	}
	sort(res.begin(), res.end());
	reverse(res.begin(), res.end());
	res.resize(pagesize * ratio / 100);
	return res;
}

vector <double> Model::FindValleysSorted(vector <double> xs, int ratio = TOP_PEAK_PERCENT) {
	int pagesize = xs.size();
	vector <double> res;
	for (int i = 1; i < pagesize - 1; i++) {
		if (xs[i] < xs[i - 1] && xs[i] < xs[i + 1])
			res.emplace_back(xs[i]);
	}
	sort(res.begin(), res.end());
	res.resize(pagesize * ratio / 100);
	return res;
}

vector <double> Model::FindGaps(vector <double> data) {
	vector <double> gap;
	for (int j = 0; j < MEAN_GAP_DIM; j++) {
		vector <double> fragment;
		for (int k = PAGE_SIZE * j / MEAN_GAP_DIM; k < PAGE_SIZE * (j + 1) / MEAN_GAP_DIM; k++)
			fragment.emplace_back(data[k]);
		vector <double> peaks = FindPeaksSorted(fragment);
		vector <double> valleys = FindValleysSorted(fragment);
		if (peaks.empty())
			peaks.emplace_back(0.0);
		if (valleys.empty())
			valleys.emplace_back(0.0);
		gap.emplace_back(GetMean(peaks) - GetMean(valleys));
	}
	return gap;
}

vector <vector <double> > Model::Sliding(vector <double> buffer) {
	vector <vector <double> > result;
	int len = buffer.size();
	for (int j = PAGE_SIZE; j < len; j++) {
		vector <double> unit;
		for (int k = j - PAGE_SIZE; k < j; k++)
			unit.emplace_back(buffer[k]);
		result.emplace_back(unit);
	}
	return result;
}

void Model::WriteToFile3(int index, double mean, double std) {
	FILE* fp = fopen(Model::TRAINING_MODEL_FILE, "w");
	fprintf(fp, "%d\n", index);
	fprintf(fp, "%f\n", mean);
	fprintf(fp, "%f\n", std);
	fclose(fp);
}


void Model::Run3(int time_interval) {
	int end_pos = min(int(_original_data.size()), time_interval * SAMPLE_RATE);
	double now_max_gap = 0.0;
	int now_max_gap_index = -1;
	vector <double> now_gaps = vector <double>();
	for (int axis_index = 0; axis_index < 3; axis_index++) {
		vector <double> buffer;
		for (int j = 0; j < end_pos; j++)
			buffer.emplace_back(_original_data[j](axis_index));
		auto raw_data = Sliding(buffer);
		assert(raw_data[0].size() == PAGE_SIZE);
		vector <double> gaps;
		for (int k = 0; k < raw_data.size();k++){
			vector <double> block = raw_data[k];
			gaps.emplace_back(GetMean(FindGaps(block)));
		}
	//	for (auto item : gaps)
	//		printf("[%f] ", item);
		double gap = GetMean(gaps);
		printf("%f\n", gap);
		if (gap > now_max_gap) {
			now_max_gap = gap;
			now_max_gap_index = axis_index;
			now_gaps = gaps;
		}
	}
	vector <double> gaps = now_gaps;
	double mean = GetMean(gaps);
	printf("!! %f\n", mean);
	double std = GetStd(gaps);
	WriteToFile3(now_max_gap_index, mean, std);
	printf("!!!!!!!! %d !!!!!!!!!!!!\n", now_max_gap_index);
}

double Model::GetMean(vector <double> xs) {
	return accumulate(xs.begin(), xs.end(), 0.0) / xs.size();
}

double Model::GetStd(vector <double> xs) {
	double mean = GetMean(xs);
	double sum = 0.0;
	int len = xs.size();
	for (int i = 0; i < len; i++)
		sum += xs[i] * xs[i];
	return sqrt(sum / xs.size() - mean * mean);
}

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
