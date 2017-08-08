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

double Model::FindGaps(vector <double> peaks, vector <double> valleys) {
	int pos = PAGE_SIZE * TOP_PEAK_PERCENT / 100;
	int peak_pos = min(pos, (int)peaks.size());
	int valley_pos = min(pos, (int)valleys.size());
	double peak_ave = accumulate(peaks.end() - peak_pos, peaks.end(), 0.0) / peak_pos;
	double valley_ave = accumulate(valleys.begin(), valleys.begin() + valley_pos, 0.0) / valley_pos;
	return peak_ave - valley_ave;
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

		vector <double> valleys = vector <double>();
		vector <double> peaks = vector <double>();

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
	//	for (auto item : gaps)
	//		printf("[%f] ", item);
		double gap = GetMean(gaps);
		printf("%lf\n", gap);
		if (gap > now_max_gap) {
			now_max_gap = gap;
			now_max_gap_index = axis_index;
			now_gaps = gaps;
		}
	}
	vector <double> gaps = now_gaps;
	double mean = GetMean(gaps);
	printf("!! %lf\n", mean);
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
