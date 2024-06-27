#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Simplified representation of an image
struct Image {
    std::vector<double> features;
    int label; // 1 for dog, -1 for cat
};

// Simple linear SVM implementation
class SVM {
private:
    std::vector<double> weights;
    double bias;
    double learning_rate;
    int num_iterations;

public:
    SVM(int feature_size, double lr = 0.01, int iterations = 1000) 
        : weights(feature_size, 0.0), bias(0.0), learning_rate(lr), num_iterations(iterations) {}

    void train(const std::vector<Image>& training_data) {
        for (int iter = 0; iter < num_iterations; ++iter) {
            for (const auto& image : training_data) {
                double prediction = predict(image.features);
                
                if (image.label * prediction < 1) {
                    for (size_t i = 0; i < weights.size(); ++i) {
                        weights[i] += learning_rate * (image.label * image.features[i] - 2 * (1.0 / num_iterations) * weights[i]);
                    }
                    bias += learning_rate * image.label;
                }
            }
        }
    }

    int classify(const std::vector<double>& features) const {
        return predict(features) > 0 ? 1 : -1;
    }

private:
    double predict(const std::vector<double>& features) const {
        double result = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * features[i];
        }
        return result + bias;
    }
};

// Function to generate simplified mock data
std::vector<Image> generate_mock_data(int num_samples, int feature_size) {
    std::vector<Image> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < num_samples; ++i) {
        Image img;
        img.features.resize(feature_size);
        for (int j = 0; j < feature_size; ++j) {
            img.features[j] = dis(gen);
        }
        img.label = (dis(gen) > 0.5) ? 1 : -1; // Randomly assign dog (1) or cat (-1)
        data.push_back(img);
    }

    return data;
}

int main() {
    const int feature_size = 10; // Simplified feature size
    const int num_train_samples = 1000;
    const int num_test_samples = 200;

    // Generate mock training and testing data
    auto train_data = generate_mock_data(num_train_samples, feature_size);
    auto test_data = generate_mock_data(num_test_samples, feature_size);

    // Create and train SVM
    SVM svm(feature_size);
    svm.train(train_data);

    // Test the SVM
    int correct = 0;
    for (const auto& image : test_data) {
        int prediction = svm.classify(image.features);
        if (prediction == image.label) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / num_test_samples * 100.0;
    std::cout << "Accuracy on test data: " << accuracy << "%" << std::endl;

    return 0;
}