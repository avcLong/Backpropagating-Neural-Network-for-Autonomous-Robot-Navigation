#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <vector>
using namespace std;

class NeuralNet
{
public:
	//-constructor-
	NeuralNet(
		int inputCount,
		int hiddenCount,
		int outputCount,
		double lambda1, 
		double lambda2,
		double learningRate,
		double momentum
	);
	
	//-functions-
	vector<double> train(vector<double> &, vector<double> &);
	vector<double> predict(vector<double> &);

	void settingAllWeights();
	void storeWeights();
	void importWeights();
	void clearWeights();

private:
	//-variables-
	int  inputCount, hiddenCount, outputCount;
	double lambdaHidden, lambdaOutput, learningRate, momentumRate;

	vector<vector<double>> hiddenWeights;
	vector<vector<double>> outputWeights; 

	vector<vector<double>> deltaHiddenWeights;
	vector<vector<double>> deltaOutputWeights;
	vector<vector<double>> oldDeltaHiddenWeights;
	vector<vector<double>> oldDeltaOutputWeights;
	int bias = 1;



	//-functions-

	//preliminary
	void initializeWeights(vector<vector<double>> &, int, int);
	void initializeDeltaWeights(vector<vector<double>> &, int, int);
	
	//feedforward
	vector<double> feedForward(vector<double> &, vector<vector<double>> &, double);
	vector<double> linearCombination(vector<vector<double>> &, vector<double> &);
	void sigmoid(vector<double> &, double);

	//backpropagation
	void backPropagation(vector<double> &, vector<double> &, vector<double> &, vector<double> &);
	vector<double> outputError(vector<double> &, vector<double> &);
	vector<double> outputGradientUpdate(vector<double> &, vector<double> &, double);
	vector<double> hiddenGradientUpdate(vector<double> &, vector<double> &, vector<vector<double>> &, double);
	void deltaWeightUpdate(vector<vector<double>> &, vector<vector<double>> &, vector<double> &, vector<double> &);
	void weightUpdate(vector<vector<double>> &, vector<vector<double>> &);

};

#endif
