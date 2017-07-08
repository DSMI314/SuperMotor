#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>
#include <vector>
#include "PCA.h"

#define MAX_COUNT 100

using Eigen::MatrixXd;
using namespace std;



vector <Vector3d> Open(const char* filename) {
	FILE* fp;
	errno_t err = fopen_s(&fp, filename, "r");
	if (err != 0)
		return vector <Vector3d>();

	vector <Vector3d> V;
	char* line = new char[100];
	while ((fgets(line, MAX_COUNT, fp)) != 0) {
		char* token = strtok(line, ",");
		int arr[3] = { 0 };
		for (int i = 0; i < 3; i++) {
			token = strtok(NULL, ",");
			arr[i] = atoi(token);
		}
		V.emplace_back(Vector3d(arr[0], arr[1], arr[2]));
	}
	return V;
}

int main() {
	vector <Vector3d> V = Open("recorded_original_data//motor_0706_fan2_on.csv");

}