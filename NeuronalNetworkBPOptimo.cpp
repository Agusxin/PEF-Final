#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <cmath>
#include <ctime>

const int INPUT_NODES = 10;
const int HIDDEN_NODES = 10;
const int OUTPUT_NODES = 10;
const int NUM_SAMPLES = 100;
const double LEARNING_RATE = 0.00001;

using namespace std;
using namespace Eigen;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

MatrixXd initializeWeights(int rows, int cols) {
    MatrixXd weights = MatrixXd::Random(rows, cols);
    weights = (weights + MatrixXd::Constant(rows, cols, 1.0)) / 2.0; // Normalize between 0 and 1
    return weights;
}

class NeuralNetwork {
 private:
    MatrixXd inputWeights;
    MatrixXd hiddenWeights;

 public:
    NeuralNetwork() {
        srand(time(0));
        inputWeights = initializeWeights(INPUT_NODES, HIDDEN_NODES);
        hiddenWeights = initializeWeights(HIDDEN_NODES, OUTPUT_NODES);
    }

    void train(const MatrixXd& inputs, const MatrixXd& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < inputs.rows(); ++i) {
                VectorXd hiddenOutputs = (inputs.row(i) * inputWeights).unaryExpr(ptr_fun(sigmoid));
                VectorXd finalOutputs = (hiddenOutputs.transpose() * hiddenWeights).unaryExpr(ptr_fun(sigmoid));

                VectorXd outputErrors = targets.row(i).transpose() - finalOutputs;
                VectorXd hiddenErrors = (hiddenWeights * outputErrors).cwiseProduct(hiddenOutputs.unaryExpr(ptr_fun(sigmoidDerivative)));

                hiddenWeights += LEARNING_RATE * hiddenOutputs * outputErrors.transpose();
                inputWeights += LEARNING_RATE * inputs.row(i).transpose() * hiddenErrors.transpose();
            }
        } 
    }

    void predict(const VectorXd& input) {
        VectorXd hiddenOutputs = (input.transpose() * inputWeights).unaryExpr(ptr_fun(sigmoid));
        VectorXd finalOutputs = (hiddenOutputs.transpose() * hiddenWeights).unaryExpr(ptr_fun(sigmoid));

        cout << "Predicción: ";
        for (int i = 0; i < finalOutputs.size(); ++i) {
            cout << finalOutputs[i] << " ";
        }
        cout << "\n";
    }
};

int main() {
    NeuralNetwork nn;

    // Generar datos de entrenamiento aleatorios 
    MatrixXd inputs = MatrixXd::Random(NUM_SAMPLES, INPUT_NODES);
    inputs = (inputs + MatrixXd::Constant(NUM_SAMPLES, INPUT_NODES, 1.0)) / 2.0; // Normalize between 0 and 1

    MatrixXd targets = MatrixXd::Random(NUM_SAMPLES, OUTPUT_NODES);
    targets = (targets + MatrixXd::Constant(NUM_SAMPLES, OUTPUT_NODES, 1.0)) / 2.0; // Normalize between 0 and 1

    nn.train(inputs, targets, 1000);

    // Realizar predicción con un nuevo conjunto de datos
    VectorXd nuevoInput = (VectorXd::Random(INPUT_NODES) + VectorXd::Constant(INPUT_NODES, 1.0)) / 2.0;

    nn.predict(nuevoInput);

    return 0;
}



