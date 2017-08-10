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
	// the states of memory and clock before constructing model
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)); 
	auto before_memory_used = pmc.PeakWorkingSetSize;
	double before_time = clock();

	// construct a model
	Model model = Model("E:\\Python\\SuperMotor\\recorded_original_data\\motor_0808_1_fan1.csv");
	model.Run(60);

	// the states of memory and clock after constructing model
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)); 
	auto after_memory_used = pmc.PeakWorkingSetSize;
	double after_time = clock();

	// calculate the cost when running model
	auto memory_used = after_memory_used - before_memory_used;
	double time_used = after_time - before_time;

	//unit in bytes
	cout << "time usage:" << time_used / CLOCKS_PER_SEC << " seconds" << endl; 

	//unit in bytes
	cout << "memory usage:" << memory_used / DIV << " KB" << endl; 
}