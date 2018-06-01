#ifndef READDATA_H_
#define READDATA_H_

#include <vector>
#include <string>
using namespace std;

class ReadData {
public:
	//-constructor-
	ReadData(string trainingDataFile, int inputCount, int outputCount);
	
	//-functions-
	vector<vector<vector<double>>> readTrainingSet();
	void trimData(vector<vector<vector<double>>>);
	

private:
	//-variables-
	string trainingDataFile;
	int inputCount;
	int outputCount;
};



#endif
