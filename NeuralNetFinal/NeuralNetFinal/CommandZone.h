#ifndef COMMANDZONE_H_
#define COMMANDZONE_H_
#include "NeuralNet.h"
#include "ReadData.h"
#include <string>
#include <vector>
using namespace std;

class CommandZone {
public:
	vector<double> moveRobot(double, double, double);
	void initializeTraining();
	void trimming();
	//void tuningTrick();
	double checking();

private:
	//-variables-
	vector<vector<vector<double>>> trainingSet;
	vector<vector<vector<double>>> validationSet;
	vector<vector<vector<double>>> testingSet;

	//-functions-
	vector<double> supplyInputs(vector<vector<vector<double>>> &, int);
	vector<double> supplyTargets(vector<vector<vector<double>>> &, int);
	void maxMin(vector<vector<vector<double>>> &);
	void normalizeData(vector<vector<vector<double>>> &);
	double normalizeValue(double, double, double);
	double denormalizeValue(double, double, double);
	void shuffleData(vector<vector<vector<double>>> &);
	void splitData(vector<vector<vector<double>>> &);
	void importMaxMin();
	double rootMeanSquareError(double, int);
	void storeParameters(double, double, double);
	

};


#endif