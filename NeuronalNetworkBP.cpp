#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>

const int INPUT_NODES = 10;
const int HIDDEN_NODES = 10;
const int OUTPUT_NODES = 10;
const int NUM_SAMPLES = 100;
const double LEARNING_RATE = 0.00001;

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// Función para inicializar las matrices con valores aleatorios
void initializeWeights(vector<vector<double>>& weights, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            weights[i][j] = ((double)rand() / (RAND_MAX));
        }
    }
}

// Función para imprimir una matriz
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

// Red Neuronal 
class NeuralNetwork {

 private:
    vector<vector<double>> inputWeights = vector<vector<double>>(INPUT_NODES, vector<double>(HIDDEN_NODES));
    vector<vector<double>> hiddenWeights = vector<vector<double>>(HIDDEN_NODES, vector<double>(OUTPUT_NODES));
 public:
    NeuralNetwork() {
        srand(time(0));
        initializeWeights(inputWeights, INPUT_NODES, HIDDEN_NODES);
        initializeWeights(hiddenWeights, HIDDEN_NODES, OUTPUT_NODES);
    }

    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < inputs.size(); ++i) {
                vector<double> hiddenOutputs(HIDDEN_NODES);
                vector<double> finalOutputs(OUTPUT_NODES);

                // Forward pass
                for (int j = 0; j < HIDDEN_NODES; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < INPUT_NODES; ++k) {
                        sum += inputs[i][k] * inputWeights[k][j];
                    }
                    hiddenOutputs[j] = sigmoid(sum);
                }

                for (int j = 0; j < OUTPUT_NODES; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < HIDDEN_NODES; ++k) {
                        sum += hiddenOutputs[k] * hiddenWeights[k][j];
                    }
                    finalOutputs[j] = sigmoid(sum);
                }

                // Backward pass 
                vector<double> outputErrors(OUTPUT_NODES);
                for (int j = 0; j < OUTPUT_NODES; ++j) {
                    outputErrors[j] = targets[i][j] - finalOutputs[j];
                }

                vector<double> hiddenErrors(HIDDEN_NODES);
                for (int j = 0; j < HIDDEN_NODES; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < OUTPUT_NODES; ++k) {
                        sum += outputErrors[k] * hiddenWeights[j][k];
                    }
                    hiddenErrors[j] = sum * sigmoidDerivative(hiddenOutputs[j]);
                }

                // Update weights
                for (int j = 0; j < HIDDEN_NODES; ++j) {
                    for (int k = 0; k < OUTPUT_NODES; ++k) {
                        hiddenWeights[j][k] += LEARNING_RATE * outputErrors[k] * hiddenOutputs[j];
                    }
                }

                for (int j = 0; j < INPUT_NODES; ++j) {
                    for (int k = 0; k < HIDDEN_NODES; ++k) {
                        inputWeights[j][k] += LEARNING_RATE * hiddenErrors[k] * inputs[i][j];
                    }
                }
            }
        }
    }

    void predict(const vector<double>& input) {
        vector<double> hiddenOutputs(HIDDEN_NODES);
        vector<double> finalOutputs(OUTPUT_NODES);

        for (int j = 0; j < HIDDEN_NODES; ++j) {
            double sum = 0.0;
            for (int k = 0; k < INPUT_NODES; ++k) {
                sum += input[k] * inputWeights[k][j];
            }
            hiddenOutputs[j] = sigmoid(sum);
        }

        for (int j = 0; j < OUTPUT_NODES; ++j) {
            double sum = 0.0;
            for (int k = 0; k < HIDDEN_NODES; ++k) {
                sum += hiddenOutputs[k] * hiddenWeights[k][j];
            }
            finalOutputs[j] = sigmoid(sum);
        }

        cout << "Predicción: ";
        for (double val : finalOutputs) {
            cout << val << " ";
        }
        cout << "\n";
    }

};

int main() {
    NeuralNetwork nn;

    // Generar datos de entrenamiento aleatorios 
    vector<vector<double>> inputs(NUM_SAMPLES, vector<double>(INPUT_NODES));
    vector<vector<double>> targets(NUM_SAMPLES, vector<double>(OUTPUT_NODES));

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        for (int j = 0; j < INPUT_NODES; ++j) {
            inputs[i][j] = ((double)rand() / (RAND_MAX)); //generar numeros aleatorios entre 0 y 1
        }
        for (int j = 0; j < OUTPUT_NODES; ++j) {
            targets[i][j] = ((double)rand() / (RAND_MAX)); //generar numeros aleatorios entre 0 y 1
        }
    }

    nn.train(inputs, targets, 10000);

    // Realizar predicción con un nuevo conjunto de datos
    vector<double> nuevoInput(INPUT_NODES);
    for (int i = 0; i < INPUT_NODES; ++i) {
        nuevoInput[i] = ((double)rand() / (RAND_MAX));
    }

    nn.predict(nuevoInput);

    return 0;
}
