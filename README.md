# AutoML-MAT523
Experiment and analysis over multiple hyperparameter optimization methods on different classification tasks.

## Requirements

The present package is written in **Python 3.7**. In order to run a full capacity, the user should have a **Nvidia GPU** with **CUDA 10.1** installed. Also, the following HPO package are required to execute our test.
```
-GPyOpt 
-HyperOpt 
-HpBandSter 
-ConfigSpace
```

## Usage

### Models available
The actual implementation allows to tune the hyperparameters of 4 different classification models

```
- SVM
- MLP
- CnnVanilla
- Resnet
```

### Tuning methods
The tuning can be done with the following hyperparameter optimization methods :
```
- grid_search
- random_search      # 2 possible variants : (GP, GP_MCMC), 2 possible acquisition functions : (EI, MPI)
- gaussian_process
- tpe
- annealing
- hyperband
- BOHB
```

### Example of usage
Each hyperparameter search space will be defined with the help of the following domain objects

- ```ContinuousDomain(lower_bound, upper_bound, log_scaled=False)```
- ```DiscreteDomain(list_of_values)```


Here's a detailed example on how to use the methods available 

```
# Append path of module to sys and import module
module_path = os.path.dirname(os.getcwd())
sys.path.append(module_path)
```
```
# We generate data for our tests and global variables for all tests
x_train, t_train, x_test, t_test = load_breast_cancer_dataset(random_state=42)
dataset = 'Breast_Cancer_Wisconsin'
train_size = len(x_train)
nb_cross_validation = 4
nb_evals = 250
```
```
# We initialize an MLP with default hyper-parameters and 4 hidden layers of 20 neurons to classify
# our data and test its performance on both training and test data sets
mlp = MLP(hidden_layers_number=4, layers_size=20, max_iter=1000)
mlp.fit(x_train, t_train)
```
```
# We set the experiment title and save the path to save the results
experiment_title = 'BreastCancerClassification'
results_path = os.path.join(os.path.dirname(module_path), 'Results')
```

#### Standard GP with EI acquisition function
 ```
# We initialize a tuner with the standard GP method and set our search space
GP_tuner = HPtuner(mlp, 'gaussian_process')
GP_tuner.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                           'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                           'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int))),
                           'hidden_layers_number': DiscreteDomain(range(1, 21)),
                           'layers_size': DiscreteDomain(range(20, 101))})
```
```
# We execute the tuning using default parameters for GP
# ('GP' as method type, 5 initial points to evaluate before the beginning and 'EI' acquisition)
GP_results = GP_tuner.tune(x_train, t_train, n_evals=nb_evals, nb_cross_validation=nb_cross_validation)
```
```
# We save the results
GP_results.save_all_results(results_path, experiment_title, dataset,
                            train_size, mlp.score(x_test, t_test))                           
```
The user should take a look at the files contained in _Code/Experiments_ for a better understanding on how to use the code. 

Detailed descriptions of every hyperparameters that can be tuned for each model are available in _Code/Model.py_

#### Tuning your own model
Further documentation will be added eventually


## Installation
Further documentation will be added eventually

## Results from this implementation

### CIFAR10

![CIFAR10](./Assets/Cifar10.png)

<b> Model: ResNet </b> <p></p>

| Hyperparameter | Distribution | Min | Max | Step | Category|
| --- | --- | --- | --- | --- | --- |
|Learning rate | Log-Uniform | 1e-7| 1e-1| N/A| N/A|
|L2 regularization | Log-Uniform | 1e-10| 1e-1| N/A| N/A |   
|ADAM eps | Discrete | 1e-8| 1e0| x10| N/A|  
|Batch size | Discrete | 50| 250| 10| N/A|  
|# Layer | Discrete | 7| 31| 3| N/A|  
|Lr decay rate | Discrete | 2| 40| 1| N/A|   
|Activation | Categorical | N/A| N/A| N/A| ELU, ReLU, Swish[1], Mish[2]|
|Version | Categorical | N/A| N/A| N/A| Post-Act, Pre-Act|

<br/><br/>
### SVHN

![SVHN](./Assets/SVHN.png)

<b> Model: ResNet </b> <p></p>

| Hyperparameter | Distribution | Min | Max | Step | Category|
| --- | --- | --- | --- | --- | --- |
|Learning rate | Log-Uniform | 1e-7| 1e-1| N/A| N/A|
|L2 regularization | Log-Uniform | 1e-10| 1e-1| N/A| N/A |   
|ADAM eps | Discrete | 1e-8| 1e0| x10| N/A|  
|Batch size | Discrete | 50| 250| 10| N/A|  
|# Layer | Discrete | 7| 19| 3| N/A|  
|Lr decay rate | Discrete | 2| 40| 1| N/A|   
|Activation | Categorical | N/A| N/A| N/A| ELU, ReLU, Swish[1], Mish[2]|
|Version | Categorical | N/A| N/A| N/A| Post-Act, Pre-Act|

<br/><br/>
### NSPIRAL

![NSPIRAL](./Assets/SPIRAL2_1.png)
![NSPIRAL](./Assets/SPIRAL2_2.png)

<b> Model: Multi Layer Perceptron </b> <p></p>

| Hyperparameter | Distribution | Min | Max | Step | Category|
| --- | --- | --- | --- | --- | --- |
|Learning rate | Log-Uniform | 1e-8| 1e0| N/A| N/A|
|L2 regularization | Log-Uniform | 1e-8| 1e0| N/A| N/A |     
|Batch size | Discrete | 50| 500| 10| N/A|  
|# Layer | Discrete | 1| 20| 1| N/A|  
|Layer size | Discrete | 5| 50| 1| N/A|   

<br/><br/>
### DIGITS

![DIGITS](./Assets/DIGITS.png)

<b> Model: SVM </b> <p></p>

| Hyperparameter | Distribution | Min | Max | Step | Category|
| --- | --- | --- | --- | --- | --- |
|C | Log-Uniform | 1e-8| 1e0| N/A| N/A|
|Gamma | Log-Uniform | 1e-8| 1e0| N/A| N/A |     

<br/><br/>
### IRIS

![IRIS](./Assets/IRIS2.png)

<b> Model: Multi Layer Perceptron </b> <p></p>

| Hyperparameter | Distribution | Min | Max | Step | Category|
| --- | --- | --- | --- | --- | --- |
|Learning rate | Log-Uniform | 1e-8| 1e0| N/A| N/A|
|L2 regularization | Log-Uniform | 1e-8| 1e0| N/A| N/A |     
|Batch size | Discrete | 50| 500| 10| N/A|  
|# Layer | Discrete | 1| 50| 1| N/A|  
|Layer size | Discrete | 5| 50| 1| N/A|   

<br/><br/>
### Breast Cancer Wisconsin

![BreastCancer](./Assets/BreastCancer.png)

<b> Model: Multi Layer Perceptron </b> <p></p>

| Hyperparameter | Distribution | Min | Max | Step | Category|
| --- | --- | --- | --- | --- | --- |
|Learning rate | Log-Uniform | 1e-8| 1e0| N/A| N/A|
|L2 regularization | Log-Uniform | 1e-8| 1e0| N/A| N/A |     
|Batch size | Discrete | 50| 500| 10| N/A|  
|# Layer | Discrete | 1| 50| 1| N/A|  
|Layer size | Discrete | 20| 100| 1| N/A| 

## References

- [1] Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Swish: a self-gated activation function.", (2017), arXiv preprint [arXiv:1710.059417]
- [2] Misra,D.Mish:A    Self    Regularized Non-Monotonic Neural Activation Function,2019,[arXiv:cs.LG/1908.08681]
