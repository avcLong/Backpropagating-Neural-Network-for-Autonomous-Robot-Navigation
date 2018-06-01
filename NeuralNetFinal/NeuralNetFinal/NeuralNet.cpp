#include "NeuralNet.h"
#include <vector>
#include <random>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

//----------GLOBAL VARIABLES----------


//----------CONSTRUCTOR----------
NeuralNet::NeuralNet( int i, int j, int k, double lambda1, double lambda2, double eta, double alpha)
{
	inputCount = i;
	hiddenCount = j;
	outputCount = k;
	lambdaHidden = lambda1;
	lambdaOutput = lambda2;
	learningRate = eta;
	momentumRate = alpha;
}


//----------PUBLIC FUNCTIONS----------

//-core-
vector<double> NeuralNet::train(vector<double> &inputVector, vector<double> &targetVector)
{
	//one iteration of feedforward+backpropagation, returns the error at each output neuron
	vector<double> hiddenVector;
	vector<double> outputVector;
		inputVector.push_back(bias);
		hiddenVector = feedForward(inputVector, hiddenWeights, lambdaHidden);
		hiddenVector.push_back(bias);
		outputVector = feedForward(hiddenVector, outputWeights, lambdaOutput);
		backPropagation(inputVector, hiddenVector, outputVector, targetVector);
		hiddenVector.pop_back();
		inputVector.pop_back();
		return outputVector;
}

vector<double> NeuralNet::predict(vector<double> &inputVector)
{
	//gets inputs, predicts the outputs, no updating of weights
	//used for validation, testing and moving the robot
	vector<double> hiddenVector;
	vector<double> outputVector;
	vector<double> errorVector;
	inputVector.push_back(bias);
	hiddenVector = feedForward(inputVector, hiddenWeights, lambdaHidden);
	hiddenVector.push_back(bias);
	outputVector = feedForward(hiddenVector, outputWeights, lambdaOutput);
	inputVector.pop_back();
	hiddenVector.pop_back();
	return outputVector;
}



//----------PRIVATE FUNCTIONS----------

//-----preliminary-----

void NeuralNet::settingAllWeights()
{
	initializeWeights(hiddenWeights, inputCount, hiddenCount);
	initializeWeights(outputWeights, hiddenCount, outputCount);
	initializeDeltaWeights(deltaHiddenWeights, inputCount, hiddenCount);
	initializeDeltaWeights(deltaOutputWeights, hiddenCount, outputCount);
	initializeDeltaWeights(oldDeltaHiddenWeights, inputCount, hiddenCount);
	initializeDeltaWeights(oldDeltaOutputWeights, hiddenCount, outputCount);
}

void NeuralNet::initializeWeights(vector<vector<double>> &w, int inputCount, int outputCount)
{
	//initialize weights to a small random value
	;//have to take the bias in account
	double min = -(1.0 / sqrt(outputCount));
	double max = (1.0 / sqrt(inputCount));
	for (int i = 0; i < outputCount; i++) {
		vector<double> row;
		for (int j = 0; j < inputCount + 1 ; j++) {
			double val = (max - min) * ((double)rand() / (double)RAND_MAX) + min;
			row.push_back(val);
		}
		w.push_back(row);
	}
}

void NeuralNet::initializeDeltaWeights(vector<vector<double>> &dW, int inputCount, int outputCount)
{
	//initialize deltaweights to 0
	inputCount++;
	for (int i = 0; i < outputCount; i++) {
		vector<double> row;
		for (int j = 0; j < inputCount; j++) {
			double val = 0;
			row.push_back(val);
		}
		dW.push_back(row);
	}
}

//-----feedforward-----

//-core-
vector<double> NeuralNet::feedForward(vector<double> &input, vector<vector<double>> &weights, double lambda)
{
	//one feedforward iteration
	vector<double> output = linearCombination(weights, input);
	sigmoid(output, lambda);
	return output;
}

//-auxiliary-
vector<double> NeuralNet::linearCombination(vector<vector<double>> &w, vector<double> &x)
{
	vector<double> v;
	for (int i = 0; i < w.size(); i++) {
		double val = 0;
		for (int j = 0; j < w[0].size(); j++) {
			val = val + w[i][j] * x[j];
		}
		v.push_back(val);
	}
	return v;
}

void NeuralNet::sigmoid(vector<double> &y, double lambda)
{
	//activation function, can use a different lambda for hidden and output layer
	for (int i = 0; i < y.size(); i++) {
		y[i] = 1 / (1 + exp(-lambda * y[i]));
	}
}

//-----backpropagation-----

//-core-
void NeuralNet::backPropagation(vector<double> &inputs, vector<double> &hiddenOutputs, vector<double> &outputs, vector<double> &targets)
{	
	//one backpropagation iteration
	vector<double> errors = outputError(outputs, targets);
	vector<double> outputGradients = outputGradientUpdate(errors, outputs, lambdaOutput);
	vector<double> hiddenGradients = hiddenGradientUpdate(hiddenOutputs, outputGradients, outputWeights, lambdaHidden);
	deltaWeightUpdate(deltaOutputWeights, oldDeltaOutputWeights, hiddenOutputs, outputGradients);
	deltaWeightUpdate(deltaHiddenWeights, oldDeltaHiddenWeights, inputs, hiddenGradients);
	weightUpdate(outputWeights, deltaOutputWeights);
	weightUpdate(hiddenWeights, deltaHiddenWeights);
}

//-auxiliary-
vector<double> NeuralNet::outputError(vector<double> &y, vector<double> &t)
{
	vector<double> e;
	for (int i = 0; i < y.size(); i++) {
		double val = t[i] - y[i];
		e.push_back(val);
	}
	return e;
}

vector<double> NeuralNet::outputGradientUpdate(vector<double> &e, vector<double> &y, double lambda)
{
	vector<double> delta;
	for (int t = 0; t < y.size(); t++) {
		double val = lambda * y[t] * (1 - y[t]) * e[t];
		delta.push_back(val);
	}
	return delta;
}

vector<double> NeuralNet::hiddenGradientUpdate(vector<double> &h, vector<double> &delta, vector<vector<double>> &w, double lambda)
{
	vector<double> deltah;
	for (int i = 0; i < h.size() ; i++) {
		double sum = 0;
		for (int j = 0; j < delta.size(); j++) {
			sum = sum + delta[j] * w[j][i];
		}
		double val = lambda * h[i] * (1 - h[i]) * sum;
		deltah.push_back(val);
	}
	return deltah;
}

void NeuralNet::deltaWeightUpdate(vector<vector<double>> &dW, vector<vector<double>> &oldDW, vector<double> &x, vector<double> &delta) 
{
	for (int i = 0; i < dW.size(); i++) {
		for (int j = 0; j < dW[0].size(); j++) {
			dW[i][j] = (learningRate * delta[i] * x[j]) + (momentumRate * oldDW[i][j]);
			oldDW[i][j] = dW[i][j];
		}
	}
}

void NeuralNet::weightUpdate(vector<vector<double>> &w, vector<vector<double>> &dW)
{
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < w[0].size(); j++) {
			w[i][j] = w[i][j] + dW[i][j];
		}
	}
}

//-----validation and testing-----
void NeuralNet::storeWeights()
{
	//write weights on a file
	ofstream wfile;
	wfile.open("bestHiddenWeights.csv");
	wfile << "";
	wfile.close();

	ofstream vfile;
	vfile.open("bestOutputWeights.csv");
	vfile << "";
	vfile.close();

	for (int i = 0; i < hiddenWeights.size(); i++) {
		for (int j = 0; j < hiddenWeights[0].size(); j++) {
			wfile.open("bestHiddenWeights.csv", std::ios::app);
			wfile << hiddenWeights[i][j] << ",";
			wfile.close();
		}
		wfile.open("bestHiddenWeights.csv", std::ios::app);
		wfile << "\n";
		wfile.close();
	}

	for (int i = 0; i < outputWeights.size(); i++) {
		for (int j = 0; j < outputWeights[0].size(); j++) {
			vfile.open("bestOutputWeights.csv", std::ios::app);
			vfile << outputWeights[i][j] << ",";
			vfile.close();
		}
		vfile.open("bestOutputWeights.csv", std::ios::app);
		vfile << "\n";
		vfile.close();
	}
}

void NeuralNet::importWeights()
{
	clearWeights();

	//import the best weights from the file
	ifstream wfile("C:\\Users\\al17989\\Desktop\\NeuralNetFinal\\NeuralNetFinal\\bestHiddenWeights.csv");
	ifstream vfile("C:\\Users\\al17989\\Desktop\\NeuralNetFinal\\NeuralNetFinal\\bestOutputWeights.csv");
	string stringVal;
	double val;

	if (wfile.is_open()) cout << "---importing weights---" << "\n";

	for (int j = 0; j < hiddenCount; j++){
		vector<double> values;
		for (int i = 0; i < inputCount + 1; i++) {
			getline(wfile, stringVal, ',');
			val = atof(stringVal.c_str());
			values.push_back(val);
		}
		hiddenWeights.push_back(values);
		getline(wfile, stringVal);
	}

	for (int j = 0; j < outputCount; j++){
		vector<double> values;
		for (int i = 0; i < hiddenCount + 1; i++) {
			getline(vfile, stringVal, ',');
			val = atof(stringVal.c_str());
			values.push_back(val);
		}
		outputWeights.push_back(values);
	}
}

void NeuralNet::clearWeights()
{
	//clear old weights
	for (int i = 0; i < hiddenWeights.size(); i++)
		hiddenWeights[i].clear();
	hiddenWeights.clear();
	for (int i = 0; i < outputWeights.size(); i++)
		outputWeights[i].clear();
	outputWeights.clear();
}