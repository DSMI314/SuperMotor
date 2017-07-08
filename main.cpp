#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstdio>
#include <vector>
#include "Model.h"
#include <windows.h>
#include <psapi.h>
#include <time.h>

#pragma comment(lib, "strmiids.lib") 
#pragma comment(lib, "psapi.lib")
PROCESS_MEMORY_COUNTERS pmc;

#define DIV 1024 // Convert Byte to KB

using Eigen::MatrixXd;
using namespace std;


int main() {

	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)); 
	auto before_memory_used = pmc.PeakWorkingSetSize;
	double before_time = clock();


	Model model = Model("recorded_original_data//motor_0706_fan2_on.csv");
	model.Run3(60);

	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)); 
	auto after_memory_used = pmc.PeakWorkingSetSize;
	double after_time = clock();

	auto memory_used = after_memory_used - before_memory_used;
	double time_used = after_time - before_time;

	cout << "time usage:" << time_used / CLOCKS_PER_SEC << " seconds" << endl; //unit in bytes
	cout << "memory usage:" << memory_used / DIV << " KB" << endl; //unit in bytes
}