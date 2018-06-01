#include "CommandZone.h"
#include "ReadData.h"
#include "NeuralNet.h"
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <math.h>
#include <fstream>
using namespace std;

//----------GLOBAL VARIABLES----------

//-settings for my task, i would like that only these numbers can be changed-
int inputCount = 2;
int hiddenCount = 6;
int outputCount = 2;
double lambdaHidden = 0.6;
double lambdaOutput = 0.6;
double learningRate = 0.5;
double momentumRate = 0.9;
string dataFile = "C:\\Users\\al17989\\Desktop\\NeuralNetFinal\\NeuralNetFinal\\trimmedData.csv";
string unrefinedData = "C:\\Users\\avclo\\Msc AI\\NeuralNets\\NeuralNetFinal\\nntrainingdata.csv";
//-end of touchable settings-

NeuralNet MyNet = NeuralNet(inputCount, hiddenCount, outputCount, lambdaHidden, lambdaOutput, learningRate, momentumRate);
ReadData MyReader = ReadData(dataFile, inputCount, outputCount);
ReadData MyTrimmer = ReadData(unrefinedData, inputCount, outputCount);

double inputMax = 0;
double inputMin = 10000;
double outputMax = 0;
double outputMin = 10000;

double testingRMSEtotal;
vector<vector<vector<double>>> check;

//----------PUBLIC FUNCTIONS----------

vector<double> CommandZone::moveRobot(double sonar0, double sonar1, double t) 
{
	//use it to move the robot
	static int initialization = 0;
	if (initialization == 0) {
		MyNet.importWeights();
		importMaxMin();
		initialization++;
	}
	vector<double> input;
	double s0 = normalizeValue(sonar0, inputMax, inputMin);
	double s1 = normalizeValue(sonar1, inputMax, inputMin);
	input.push_back(s0);
	input.push_back(s1);
	vector<double> output = MyNet.predict(input);
	output[0] = denormalizeValue(output[0], outputMax, outputMin);
	output[1] = denormalizeValue(output[1], outputMax, outputMin);
	if(t < 200)
	check.push_back({ input, {output[0], output[1]} });
	return output;
}

void CommandZone::initializeTraining()
{
	//read data from data file, normalize it, and train the neural network
	vector<vector<vector<double>>> data = MyReader.readTrainingSet();
	maxMin(data);
	cout << "\n" << "normalizing" << "\n";
	normalizeData(data);
	cout << "initializing weights" << "\n";
	shuffleData(data);//shuffle the data before splitting it
	splitData(data);//split data for training, validation, and testing


	MyNet.settingAllWeights();
	double bestError = 10000;

	
	for (int epochs = 0; epochs < 100; epochs++) { //this counts the number of epochs
		cout << "training epoch" << epochs << "\n";
		shuffleData(trainingSet); //shuffle the data between epochs to avoid pattern learning

		//-training-
		double trainingSquareErrorSum1 = 0;
		double trainingSquareErrorSum2 = 0;

		for (int line = 0; line < trainingSet.size(); line++) {
			vector<double> input = supplyInputs(trainingSet, line);
			vector<double> target = supplyTargets(trainingSet, line);
			vector<double>  output = MyNet.train(input, target);
			double localTrainingError1 = target[0] - output[0];
			double localTrainingError2 = target[1] - output[1];
			trainingSquareErrorSum1 = (localTrainingError1 * localTrainingError1) + trainingSquareErrorSum1;
			trainingSquareErrorSum2 = (localTrainingError2 * localTrainingError2) + trainingSquareErrorSum2;
		}
		double trainingRMSE1 = rootMeanSquareError(trainingSquareErrorSum1, trainingSet.size());
		double trainingRMSE2 = rootMeanSquareError(trainingSquareErrorSum2, trainingSet.size());
		double trainingRMSEtotal = (trainingRMSE1 + trainingRMSE2) / 2;

		//-validating-
		double validationErrorSum1 = 0;
		double validationErrorSum2 = 0;

		for (int line = 0; line < validationSet.size(); line++) {
			vector<double> input = supplyInputs(validationSet, line);
			vector<double> target = supplyTargets(validationSet, line);
			vector<double> output = MyNet.predict(input);
			double localValidationError1 = target[0] - output[0];
			double localValidationError2 = target[1] - output[1];
			validationErrorSum1 = (localValidationError1 * localValidationError1) + validationErrorSum1;
			validationErrorSum2 = (localValidationError2 * localValidationError2) + validationErrorSum2;
		}
		double validationRMSE1 = rootMeanSquareError(validationErrorSum1, validationSet.size());
		double validationRMSE2 = rootMeanSquareError(validationErrorSum2, validationSet.size());
		double validationRMSEtotal = (validationRMSE1 + validationRMSE2) / 2;

		double epochError = validationRMSEtotal;

		//store validation error if it's the best one
		if ( epochError < bestError) {
			MyNet.storeWeights();
			bestError = epochError;
			cout << "validation error = " << bestError << "\n";
		}
	}

	//-testing the best weights-
	MyNet.importWeights();

	double testingErrorSum1 = 0;
	double testingErrorSum2 = 0;

	for (int line = 0; line < testingSet.size(); line++) {
		vector<double> input = supplyInputs(testingSet, line);
		vector<double> target = supplyTargets(testingSet, line);
		vector<double> output = MyNet.predict(input);
		double localTestingError1 = target[0] - output[0];
		double localTestingError2 = target[1] - output[1];
		testingErrorSum1 = (localTestingError1 * localTestingError1) + testingErrorSum1;
		testingErrorSum2 = (localTestingError2 * localTestingError2) + testingErrorSum2;
	}
	double testingRMSE1 = rootMeanSquareError(testingErrorSum1, testingSet.size());
	double testingRMSE2 = rootMeanSquareError(testingErrorSum2, testingSet.size());
	testingRMSEtotal = (testingRMSE1 + testingRMSE2) / 2;

	cout << "TESTING RMSE = " << testingRMSEtotal << "\n";
}

void CommandZone::trimming()
{
	//reduces the size of training data file by avoiding repetitions 
	//only used this once, then the program functions with the new file
	vector<vector<vector<double>>> data = MyTrimmer.readTrainingSet();
	cout << "starting to trim" << "\n";
	MyTrimmer.trimData(data);
}

/*void CommandZone::tuningTrick()
{
	double hcount = 0;
	double bestTuningError = 10000;
	for (int i = 0; i < 1; i++) { //hidden layer tuning
		double lcount = 0;
		for (int j = 0; j < 10; j++) { //learning rate tuning
			double mcount = 0;
			for (int k = 0; k < 10; k++) { //momentum 

				NeuralNet Tuner = NeuralNet(inputCount, hiddenCount + hcount, outputCount, lambdaHidden, lambdaOutput, learningRate + lcount, momentumRate + mcount);
				cout << "-----------------------" << "\n";
				cout << "   HIDDEN NEURONS = " << 2 + hcount << "   LEARNING RATE = " << 0.1 + lcount << "   MOMENTUM = " << 0.1 + mcount << "\n";
				cout << "-----------------------" << "\n";
				Tuner.clearWeights();
				initializeTraining(Tuner);
				if (testingRMSEtotal < bestTuningError) {
					storeParameters(2 + hcount, 0.1 + lcount, 0.1 + mcount);
					bestTuningError = testingRMSEtotal;
				}
				mcount = mcount + 0.1;
			}
			lcount = lcount + 0.1;
		}
		hcount++;
	}
}*/

double CommandZone::checking()
{
	double errorSum1 = 0;
	double errorSum2 = 0;
	vector<vector<vector<double>>> data = MyReader.readTrainingSet();
	for(int i = 0; i < check.size(); i++){
		double bestApprox = 10000;
		double bestIndex = 0;
		for(int j = 0; j<data.size(); j++)
		{
			double approx = abs(check[i][0][0] - data[i][0][0]) + abs(check[i][0][1] - data[j][0][1]);
			if(approx < bestApprox)
			{
				bestApprox = approx;
				bestIndex = j;
			}
		}
		errorSum1 = (data[bestIndex][0][0] - check[i][0][0]) * (data[bestIndex][0][0] - check[i][0][0]) + errorSum1;
		errorSum2 = (data[bestIndex][0][1] - check[i][0][1]) * (data[bestIndex][0][1] - check[i][0][1]) + errorSum2;
	}
	double checkRMSE = rootMeanSquareError(errorSum1, check.size()) + rootMeanSquareError(errorSum2, check.size());
	cout << "postmovingRMSE = " << checkRMSE << "\n";
	return checkRMSE;
}

//----------PRIVATE FUNCTIONS----------

//-data manipulation-

vector<double> CommandZone::supplyInputs(vector<vector<vector<double>>> &dataMatrix, int iteration)
{
	//creates a vector with the inputs of a new training instance
	vector<double> x;
	for (int i = 0; i < 2; i++) {
		double val = dataMatrix[iteration][0][i];
		x.push_back(val);
	}
	return x;
}

vector<double> CommandZone::supplyTargets(vector<vector<vector<double>>> &dataMatrix, int iteration)
{
	//creates a vector with the desired outputs of the same training instance as above
	vector<double> t = dataMatrix[iteration][1];
	return t;
}

void CommandZone::maxMin(vector<vector<vector<double>>> &dataMatrix)
{
	//finds max and min of the training data in order to normalize it 
	//saves it on a csv file in order to reuse it to move robot
	ofstream wfile;
	wfile.open("maxMinData.csv");
	wfile << "";
	wfile.close(); 

	for (int t = 0; t < dataMatrix.size(); t++) {
		for (int s = 0; s < dataMatrix[0][0].size(); s++) {
			if (dataMatrix[t][0][s] > inputMax)
				inputMax = dataMatrix[t][0][s];
			if (dataMatrix[t][0][s] < inputMin)
				inputMin = dataMatrix[t][0][s];
		}
		for (int s = 0; s < dataMatrix[0][1].size(); s++) {
			if (dataMatrix[t][1][s] > outputMax)
				outputMax = dataMatrix[t][1][s];
			if (dataMatrix[t][1][s] < outputMin)
				outputMin = dataMatrix[t][1][s];
		}
	}
	wfile.open("maxMinData.csv", std::ios::app);
	wfile << inputMax << "," << inputMin << "," << outputMax << "," << outputMin << ",";
	wfile.close();
}

void CommandZone::normalizeData(vector<vector<vector<double>>> &dataMatrix) 
{
	for (int t = 0; t < dataMatrix.size(); t++) {
		//normalizing inputs
		for (int s = 0; s < dataMatrix[0][0].size(); s++) {
			dataMatrix[t][0][s] = normalizeValue(dataMatrix[t][0][s], inputMax, inputMin);
		}
		//normalizing outputs
		for (int s = 0; s < dataMatrix[0][1].size(); s++) {
			dataMatrix[t][1][s] = normalizeValue(dataMatrix[t][1][s], outputMax, outputMin);
		}
	}
}

double CommandZone::normalizeValue(double x, double maximum, double minimum)
{
	return (x - minimum) / (maximum - minimum);
}

double CommandZone::denormalizeValue(double x, double maximum, double minimum)
{
	return minimum + (maximum - minimum) * x;
}

void CommandZone::shuffleData(vector<vector<vector<double>>> &dataMatrix) 
{
	//shuffles the training data to avoid repetitions
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(dataMatrix), std::end(dataMatrix), rng);
}

void CommandZone::splitData(vector<vector<vector<double>>> &dataMatrix)
{
	//splits data into training, validation, and testing set
	double trainingSetSize = (dataMatrix.size() * 70) / 100;
	double validationSetSize = (dataMatrix.size() * 15) / 100;
	double testingSetSize = (dataMatrix.size() * 15) / 100;
	int j = 0;

	while (j < trainingSetSize){
		vector<double> inputs = dataMatrix[j][0];
		vector<double> targets = dataMatrix[j][1];
		trainingSet.push_back({ inputs, targets });
		j++;
	}
	while (j < trainingSetSize + validationSetSize) {
		vector<double> inputs = dataMatrix[j][0];
		vector<double> targets = dataMatrix[j][1];
		validationSet.push_back({ inputs, targets });
		j++;
	}
	while (j < trainingSetSize + validationSetSize + testingSetSize) {
		vector<double> inputs = dataMatrix[j][0];
		vector<double> targets = dataMatrix[j][1];
		testingSet.push_back({ inputs, targets });
		j++;
	}

}

void CommandZone::importMaxMin()
{
	ifstream wfile("C:\\Users\\al17989\\Desktop\\NeuralNetFinal\\NeuralNetFinal\\maxMinData.csv");
	string stringVal;
	double val;

	getline(wfile, stringVal, ',');
	val = atof(stringVal.c_str());
	inputMax = val;

	getline(wfile, stringVal, ',');
	val = atof(stringVal.c_str());
	inputMin = val;

	getline(wfile, stringVal, ',');
	val = atof(stringVal.c_str());
	outputMax = val;

	getline(wfile, stringVal, ',');
	val = atof(stringVal.c_str());
	outputMin = val;
}

//-validation and testing-

double CommandZone::rootMeanSquareError(double squareErrorSum, int size)
{
	double rmse = sqrt((squareErrorSum) / size);
	return rmse;
}

//-tuning-

void CommandZone::storeParameters(double hc, double lr, double mo)
{
	ofstream wfile;
	wfile.open("bestParameters.csv");
	wfile << "";
	wfile.close();


	wfile.open("bestParameters.csv");
	wfile << "hn = " << hc << "," << "lr =" << lr << "," << "mo=" << mo << "," << "\n";
	wfile << "RMSE = " << testingRMSEtotal;
	wfile.close();
}