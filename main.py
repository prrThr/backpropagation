import numpy as np
import matplotlib.pyplot as plt
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def train(X, y, hidden_neurons=10, learning_rate=0.1, epochs=200000):
    input_neurons = X.shape[1]
    output_neurons = 1

    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

    bias_hidden = np.random.uniform(size=(1, hidden_neurons))
    bias_output = np.random.uniform(size=(1, output_neurons))

    for epoch in range(epochs):
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_input)

        error = y - predicted_output

        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

        if epoch % 1000 == 0:
            loss = np.mean(np.abs(error))
            print(f"Epoch {epoch}, Erro: {loss:.4f}")

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def predict(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    return predicted_output

x = np.linspace(0, 2 * np.pi, 100)
sine_wave = np.sin(x)

sine_wave_normalized = normalize(sine_wave)

X_train = []
y_train = []
window_size = 5
for i in range(len(sine_wave_normalized) - window_size):
    X_train.append(sine_wave_normalized[i:i + window_size])
    y_train.append(sine_wave_normalized[i + window_size])

X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)

weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(X_train, y_train, epochs=10000) # MUDE O EPOCHS PARA APROXIMAR O RESULTADO!

predicted = predict(X_train, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

plt.plot(sine_wave_normalized, label='Seno original (normalizado)')
plt.plot(np.concatenate([sine_wave_normalized[:window_size], predicted.flatten()]), label='Seno predito')
plt.legend()
plt.show()

