
#include <iostream>

#include "Layer.h"
#include "NeuralNetwork.h"
using namespace std;


NeuralNetwork::NeuralNetwork() {
	numberOfLayer = 0;
	aLayer = NULL;
}

NeuralNetwork::NeuralNetwork(int inputDim, int numLayer, int* eachLayer_outputDim) {
	numberOfLayer = 0;
	aLayer = NULL;

	Init(inputDim, numLayer, eachLayer_outputDim);
}


void NeuralNetwork::Delete() {
	if (aLayer != NULL) {
		delete[] aLayer;
		aLayer = NULL;
	}
	numberOfLayer = 0;
}

void NeuralNetwork::Init(int inputDim, int numLayer, int* eachLayer_outputDim) {
	if (Is_Inited())
		Delete();

	numberOfLayer = numLayer;

	try {
		aLayer = new Layer[numberOfLayer];
	}
	catch (bad_alloc& ba) {
		printf("bad memory allocation in func : %s, file : %s, line number : %d\n", __FUNCTION__, __FILE__, __LINE__);
		exit(-1);
	}

	aLayer[0].Init(inputDim, eachLayer_outputDim[0]);

	for (int i = 1; i < numberOfLayer; i++)
		aLayer[i].Init(eachLayer_outputDim[i - 1], eachLayer_outputDim[i]);

}

//Propagate�ϴ� �Լ�. Layer�� ó������ ����ڷκ��� ���� input�� �Ѱ��ְ�, �� �������ʹ� �� �� layer�� output�� input���� ��.
void NeuralNetwork::Propagate(float* input) {
	aLayer[0].Propagate(input);
	for (int i = 1; i < numberOfLayer; i++)
		aLayer[i].Propagate(aLayer[i - 1].Get_Output());

}

//back propagate �ϴ� ��. �ϰ� ���� gradient�� ���ȴ�
void NeuralNetwork::Back_Propagate(float* input, float* desiredOutput) {
	Propagate(input);

	for (int i = numberOfLayer - 1; i >= 0; i--) {
		// i �� ���� �ֻ�� layer�� ����Ű�� �ִٸ�
		if (i == numberOfLayer - 1)
			aLayer[i].Compute_Top_DeltaBar(desiredOutput);
		// i �� ���� �ֻ�� layer�� �ƴ϶��, �� ���̾�� �ش� layer�� deltabar�� ���ϵ��� ��Ź�ؾ���.
		else
			aLayer[i + 1].Compute_PrevDeltaBar(aLayer[i].Get_DeltaBar());
		//�� if-else���� deltabar�� ���ϴ� ������. 

		//deltabar�� ���ߴٸ�, gradient�� ���Ѵ�. �Ʒ� Compute_Gradient�� ȣ���ϸ� deltabar�κ��� delta�� ���ϰ�, delta�κ��� gradient�� ���Ѵ�.
		aLayer[i].Compute_Gradient();
	}
}

void NeuralNetwork::Weight_Update(float learningRate) {
	for (int i = 0; i < numberOfLayer; i++)
		aLayer[i].Weight_Update(learningRate);
}
