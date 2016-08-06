#ifndef __NeuralNetwork__
#define __NeuralNetwork__

class NeuralNetwork {
private:
	int numberOfLayer;
	Layer* aLayer;
	void Delete();
public:
	NeuralNetwork();
	NeuralNetwork(int inputDim, int numLayer, int* eachLayer_outputDim);
	~NeuralNetwork() { Delete(); }
	

	void Init(int inputDim, int numLayer, int* eachLayer_outputDim);
	int Is_Inited() { return aLayer != NULL; } //aLayer�� �Ҵ�Ǿ� �ִٸ�, return 1, �Ҵ�Ǿ����� �ʴٸ�(==NULL) return 0

	void Propagate(float* input);

	void Back_Propagate(float* input, float* desiredOutput);

	void Weight_Update(float learningRate);

	Layer& operator[] (int idx) { return aLayer[idx]; }

	float Get_Error(float* desired_output) { return aLayer[numberOfLayer - 1].Compute_Error(desired_output); }

	float* Get_Output() { return aLayer[numberOfLayer - 1].Get_Output(); }
};

#endif // !__NeuralNetwork__
