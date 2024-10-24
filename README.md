# Rede neural com back propagation
how to code a newral network with backpropagation in python (from scratch) - machineLearningMastery.com

## Iniciando o projeto

### Requisitos
1. Python
2. pip
3. venv
4. Interface gráfica

### Executando
1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python main.py`

### 1. **Importação de bibliotecas**
```python
import numpy as np
import matplotlib.pyplot as plt
```
`numpy` é a biblioteca usada para lidar com arrays e operações matemáticas e o `matplotlib.pyplot` serve para gerar gráficos.

### 2. **Função de ativação sigmoide**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
A função sigmoide é uma função de ativação comum em redes neurais, pois suaviza a saída e a restringe para o intervalo entre 0 e 1. Isso é útil para garantir que os valores de saída sejam controláveis. O `np.exp(-x)` calcula a exponencial negativa de `x`.

### 3. **Derivada da sigmoide**
```python
def sigmoid_derivative(x):
    return x * (1 - x)
```
Esta é a derivada da função sigmoide. A derivada é necessária para o cálculo do gradiente durante o processo de backpropagation. Ela mede o quanto a saída da função muda em relação à entrada.

### 4. **Função de normalização**
```python
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
```
A função `normalize` transforma os dados para o intervalo entre 0 e 1. Essa transformação é importante para garantir que os valores de entrada estejam em um intervalo adequado para a função de ativação.

### 5. **Função de treinamento**
```python
def train(X, y, hidden_neurons=10, learning_rate=0.1, epochs=10000):
    ...
```
Essa função é responsável por treinar a rede neural usando backpropagation. Vamos detalhar o que acontece em cada passo:

#### a) **Inicializando pesos e biases**
```python
input_neurons = X.shape[1]
output_neurons = 1

weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))
```
- **input_neurons**: número de entradas (features). No caso, é o tamanho da janela deslizante (5).
- **output_neurons**: um único valor de saída (o próximo valor do seno).
- **Pesos e Biases**: Inicializados aleatoriamente. Os pesos conectam a camada de entrada à camada escondida (`weights_input_hidden`) e a camada escondida à saída (`weights_hidden_output`). Biases são usados para garantir que o neurônio possa ser ativado corretamente mesmo quando todas as entradas forem zero.

#### b) **Ciclo de treinamento (épocas)**
```python
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
```
O forward pass é o processo de calcular a saída da rede neural, dado os pesos e biases atuais.

1. **hidden_layer_input**: Multiplicação entre os dados de entrada `X` e os pesos da camada de entrada para a camada escondida, somados aos biases da camada escondida.
2. **hidden_layer_output**: Passa o valor pela função de ativação sigmoide.
3. **output_layer_input**: Multiplicação entre a saída da camada escondida e os pesos da camada de saída, somados ao bias da camada de saída.
4. **predicted_output**: Saída final da rede neural após passar pela função sigmoide.

#### c) **Cálculo do erro**
```python
error = y - predicted_output
```
Cálculo do erro, ou seja, a diferença entre o valor real (`y`) e o valor previsto pela rede neural (`predicted_output`).

#### d) **Backpropagation**
```python
d_predicted_output = error * sigmoid_derivative(predicted_output)
error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
```
O processo de backpropagation consiste em ajustar os pesos com base no erro calculado.

- **d_predicted_output**: É a derivada do erro em relação à saída, multiplicada pela derivada da sigmoide.
- **error_hidden_layer**: É o erro propagado para trás na camada escondida.
- **d_hidden_layer**: Multiplica o erro da camada escondida pela derivada da sigmoide aplicada à saída da camada escondida.

#### e) **Atualização dos pesos e biases**
```python
weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
```
Aqui, os pesos e os bias são atualizados usando o gradiente calculado (backpropagation) e o fator de taxa de aprendizado (`learning_rate`), que controla o quão rápido a rede ajusta os pesos.

#### f) **Exibição do erro**
```python
if epoch % 1000 == 0:
    loss = np.mean(np.abs(error))
    print(f"Epoch {epoch}, Erro: {loss:.4f}")
```
A cada 1000 épocas, o erro médio absoluto é exibido para monitorar o progresso da rede.

### 6. **Função de previsão**
```python
def predict(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    return predicted_output
```
Essa função faz previsões após o treinamento. Ela realiza o forward pass novamente, mas sem o backpropagation (pois não há necessidade de ajustar os pesos).

### 7. **Geração dos dados do seno e preparação do conjunto de dados**
```python
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
```
- **x**: Gera 100 pontos igualmente espaçados entre 0 e \(2\pi\) (um ciclo completo de seno).
- **sine_wave**: Calcula os valores da função seno para esses pontos.
- **sine_wave_normalized**: Normaliza os valores do seno para o intervalo [0, 1].
- **X_train e y_train**: Cria janelas de 5 valores consecutivos de seno como entradas (`X_train`) e o próximo valor como a saída (`y_train`), formando o conjunto de dados para treinamento.

### 8. **Treinamento da rede neural**
```python
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(X_train, y_train)
```
A função `train()` é chamada para treinar a rede neural usando os dados preparados.

### 9. **Fazendo previsões**
```python
predicted = predict(X_train, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
```
A rede treinada é usada para prever os próximos valores do seno com base nos dados de treinamento.

### 10. **Plotando os resultados**
```python
plt.plot(sine_wave_normalized, label='Seno original (normalizado)')
plt.plot(np.concatenate([sine_wave_normalized[:window_size], predicted.flatten()]), label='Seno predito')
plt.legend()
plt.show()
```
Aqui é gerado o gráfico comparando os valores reais do seno normalizado com os valores previstos pela rede neural.
