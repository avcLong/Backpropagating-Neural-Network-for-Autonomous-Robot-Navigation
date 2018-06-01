#include "CommandZone.h"
#include "NeuralNet.h"
#include "aria.h""
#include <vector>
#include <iostream>
using namespace std;


int main(int argc, char **argv) {

	Aria::init();
	ArRobot robot;
	ArPose pose;

	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();

	ArRobotConnector robotConnector(&argParser, &robot);
	if (robotConnector.connectRobot())
		std::cout << "Robot connected!" << std::endl;
	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();

	ArSensorReading *sonarSensor[8];


	ArUtil::sleep(1000);


	double sonarRange[8];
	double avgsr[8];
	double rightBack;
	double rightFront;
	double frontRight;
	double frontLeft;
	double time = 0;


	CommandZone Robot = CommandZone();
	//Robot.initializeTraining();

	while (time < 10000) {
	
			for (int t = 0; t <= 1; t++) {
				sonarSensor[t] = robot.getSonarReading(t);
				sonarRange[t] = sonarSensor[t]->getRange();
				ArUtil::sleep(10);
		}
		if ((sonarRange[1] < 250) || (sonarRange[0] < 250))
			robot.setVel2(250, 100);
		else {
			vector<double> networkOut = Robot.moveRobot(sonarRange[0], sonarRange[1], time);
			robot.setVel2(networkOut[0], networkOut[1]);
		}
		time = time + 1;
	}
	//Robot.checking();
	system("PAUSE");
	return 0;
}