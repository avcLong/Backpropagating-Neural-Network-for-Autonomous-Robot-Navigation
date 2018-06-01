#include "ReadData.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
using namespace std;

//----------CONSTRUCTOR----------

ReadData::ReadData(string file, int i, int k)
{
	trainingDataFile = file;
	inputCount = i;
	outputCount = k;
}

//----------PRIVATE FUNCTIONS----------

vector<vector<vector<double>>> ReadData::readTrainingSet() {
	//this opens the training data file and stores it in a 3d matrix
	vector<vector<vector<double>>> trainingData;
	ifstream dataFile(trainingDataFile);
	string stringVal;
	double val;
	bool firstLine = true;
	if (dataFile.is_open()) cout << "OPEN";

	while (dataFile.good()) {

		//skip the first line because it's got no data on it
		if (firstLine) {
			getline(dataFile, stringVal);
			firstLine = false;
		}

		//get input values in a vector
		vector<double> inputData;
		for (int t = 0; t < inputCount; t++) {
			getline(dataFile, stringVal, ',');
			val = atof(stringVal.c_str());
			inputData.push_back(val);
		}

		//get output values in a vector
		vector<double> outputData;
		for (int t = 0; t < outputCount; t++) {
			getline(dataFile, stringVal, ',');
			val = atof(stringVal.c_str());
			outputData.push_back(val);
		}

		if(trainingData.size() == 0)
			trainingData.push_back({ inputData,outputData });

		int counter = 0;

		for (int k = 0; k < inputCount; k++) {
			if (inputData[k] != trainingData.back()[0][k]) {
				counter++;
				break;
			}
		}
		for (int k = 0; k < outputCount; k++) {
			if (outputData[k] != trainingData.back()[1][k]) {
				counter++;
				break;
			}
		}

		if((counter>0) && (inputData[0] < 5000) && (inputData[1] <5000))
		trainingData.push_back({ inputData,outputData });

		//this needs to be here or he won't change line
		getline(dataFile, stringVal);
	}
	return trainingData;
}

void ReadData::trimData(vector<vector<vector<double>>> dataMatrix)
{
	//this creates a new file wich contains no repetitions in the data
	//this will be used only once in my life, then commented out
	ofstream wfile;
	wfile.open("trimmedData.csv");
	wfile << "";
	wfile.close();
	int counter = 0;

	for (int i = 0; i < dataMatrix.size(); i++) {
		int j = i+1;
		while((counter<4) && (j < dataMatrix.size())) {
			counter = 0;
			for (int k = 0; k < inputCount; k++) {
				if (dataMatrix[i][0][k] == dataMatrix[j][0][k])
					counter++;
			}
			for  (int k = 0; k<outputCount; k++){
				if (dataMatrix[i][1][k] == dataMatrix[j][0][k])
					counter++;
			}
			if (j == dataMatrix.size() - 1) {
				ofstream wfile;
				wfile.open("trimmedData.csv", std::ios::app);
				wfile << dataMatrix[i][0][0] << "," << dataMatrix[i][0][1] << "," << dataMatrix[i][1][0] << "," << dataMatrix[i][1][1] << "\n";
				wfile.close();
				cout << dataMatrix[i][0][0] << "," << dataMatrix[i][0][1] << "," << dataMatrix[i][1][0] << "," << dataMatrix[i][1][1] << "\n";
			}
			j++;
		}
	}
}