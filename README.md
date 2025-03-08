## <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> Neural Network Dataset 
<sub><a href='https://pypi.python.org/pypi/nn-dataset'><img src='https://img.shields.io/pypi/v/nn-dataset.svg'/></a><br/>
short alias  <a href='https://pypi.python.org/pypi/lmur'>lmur</a></sub>
   
LEMUR - Learning, Evaluation, and Modeling for Unified Research

<img src='https://abrain.one/img/lemur-nn-whit.jpg' width='25%'/>

The original version of the <a href='https://github.com/ABrain-One/nn-dataset'>LEMUR dataset</a> was created by <strong>Arash Torabi Goodarzi, Roman Kochnev</strong> and <strong>Zofia Antonina Bentyn</strong> at the Computer Vision Laboratory, University of Würzburg, Germany.

<h3>Overview 📖</h3>
The primary goal of NN Dataset project is to provide flexibility for dynamically combining various deep learing tasks, datasets, metrics, and neural network models. It is designed to facilitate the verification of neural network performance under various combinations of training hyperparameters and data transformation algorithms, by automatically generating performance statistics. It is primarily developed to support the <a href="https://github.com/ABrain-One/nn-gpt">NN GPT</a> project.

## Create and Activate a Virtual Environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

All subsequent commands are provided for Linux/Mac OS. For Windows, please replace ```source .venv/bin/activate``` with ```.venv\Scripts\activate```.

## Installation or Update of the NN Dataset
Remove old version of the LEMUR Dataset and its database:
```bash
source .venv/bin/activate
pip uninstall nn-dataset -y
rm -rf db
```
Installing the stable version:
```bash
source .venv/bin/activate
pip install nn-dataset --upgrade --extra-index-url https://download.pytorch.org/whl/cu124
```
Installing from GitHub to get the most recent code and statistics updates:
```bash
source .venv/bin/activate
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu124
```
Adding functionality to export data to Excel files and generate plots for <a href='https://github.com/ABrain-One/nn-stat'>analyzing neural network performance</a>:
```bash
source .venv/bin/activate
pip install nn-stat --upgrade --extra-index-url https://download.pytorch.org/whl/cu124
```
and export/generate:
```bash
source .venv/bin/activate
python -m ab.stat.export
```

## Usage

Standard use cases:
1. Add a new neural network model into the `ab/nn/nn` directory.
2. Run the automated training process for this model (e.g., a new ComplexNet training pipeline configuration):
```bash
source .venv/bin/activate
python -m ab.nn.train -c img-classification_cifar-10_acc_ComplexNet
```
or for all image segmentation models using a fixed range of training parameters and transformer:
```bash
source .venv/bin/activate
python run.py -c img-segmentation -f echo --min_learning_rate 1e-4 -l 1e-2 --min_momentum 0.8 -m 0.99 --min_batch_binary_power 2 -b 6
```
To reproduce the previous result, set the minimum and maximum to the same desired values:
```bash
source .venv/bin/activate
python run.py -c img-classification_cifar-10_acc_AlexNet --min_learning_rate 0.0061 -l 0.0061 --min_momentum 0.7549 -m 0.7549 --min_batch_binary_power 2 -b 2 -f norm_299
```
To view supported flags:
```bash
python run.py -h
```

### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image:
```bash
docker run -v /a/mm:. abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python -m ab.nn.train"
```

Due to the rapid development of this project, some recently added dependencies may be missing in the 'AI Linux'. In this case, create a container from ```abrainone/ai-linux```, install the missing packages (preferably using ```pip install ...```), and then create a new image from the container using ```docker commit container_name new_image_name```. You can use this new image locally or push it to the registry for deployment on the computer cluster.

## Environment for NN Dataset Contributors
### Pip package manager
Create a virtual environment, activate it, and run the following command to install all the project dependencies:
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## Contribution

To contribute a new neural network (NN) model to the NN Dataset, please ensure the following criteria are met:

1. The code for each model is provided in a respective ".py" file within the <strong>/ab/nn/nn</strong> directory, and the file is named after the name of the model's structure.
2. The main class for each model is named <strong>Net</strong>.
3. The constructor of the <strong>Net</strong> class takes the following parameters:
   - <strong>in_shape</strong> (tuple): The shape of the first tensor from the dataset iterator. For images it is structured as `(batch, channel, height, width)`.
   - <strong>out_shape</strong> (tuple): Provided by the dataset loader, it describes the shape of the output tensor. For a classification task, this could be `(number of classes,)`.
   - <strong>prm</strong> (dict): A dictionary of hyperparameters, e.g., `{'lr': 0.24, 'momentum': 0.93, 'dropout': 0.51}`.
   - <strong>device</strong> (torch.device): PyTorch device used for the model training 
4. All external information required for the correct building and training of the NN model for a specific dataset/transformer, as well as the list of hyperparameters, is extracted from <strong>in_shape</strong>, <strong>out_shape</strong> or <strong>prm</strong>, e.g.: </br>`batch = in_shape[0]` </br>`channel_number = in_shape[1]` </br>`image_size = in_shape[2]` </br>`class_number = out_shape[0]` </br>`learning_rate = prm['lr']` </br>`momentum = prm['momentum']` </br>`dropout = prm['dropout']`.
5. Every model script has function returning set of supported hyperparameters, e.g.: </br>`def supported_hyperparameters(): return {'lr', 'momentum', 'dropout'}`</br> The value of each hyperparameter lies within the range of 0.0 to 1.0.
6. Every class <strong>Net</strong> implements two functions: </br>`train_setup(self, prm)`</br> and </br>`learn(self, train_data)`</br> The first function initializes the `criteria` and `optimizer`, while the second implements the training pipeline. See a simple implementation in the <a href="https://github.com/ABrain-One/nn-dataset/blob/main/ab/nn/nn/AlexNet.py">AlexNet model</a>.
7. For each pull request involving a new NN model, please generate and submit training statistics for 100 Optuna trials (or at least 3 trials for very large models) in the <strong>ab/nn/stat</strong> directory. The trials should cover 5 epochs of training. Ensure that this statistics is included along with the model in your pull request. For example, the statistics for the ComplexNet model are stored in files <strong>&#x003C;epoch number&#x003E;.json</strong> inside folder <strong>img-classification_cifar-10_acc_ComplexNet</strong>, and can be generated by:<br/>
```bash
python run.py -c img-classification_cifar-10_acc_ComplexNet -t 100 -e 5
```
<p>See more examples of models in <code>/ab/nn/nn</code> and generated statistics in <code>/ab/nn/stat</code>.</p>

### Available Modules

The `nn-dataset` package includes the following key modules:

1. **Dataset**:
   - Predefined neural network architectures such as `AlexNet`, `ResNet`, `VGG`, and more.
   - Located in `ab.nn.nn`.

2. **Loaders**:
   - Data loaders for datasets such as CIFAR-10 and COCO.
   - Located in `ab.nn.loader`.

3. **Metrics**:
   - Common evaluation metrics like accuracy and IoU.
   - Located in `ab.nn.metric`.

4. **Utilities**:
   - Helper functions for training and statistical analysis.
   - Located in `ab.nn.util`.


## Citation

If you find the LEMUR Neural Network Dataset to be useful for your research, please consider citing:
```bibtex
@misc{ABrain-One.NN-Dataset,
  author       = {Goodarzi, Arash Torabi and Kochnev, Roman and Khalid, Waleed and Qin, Furui and Uzun, Tolgay Atinc and Kathiriya, Yash Kanubhai and Dhameliya, Yashkumar Sanjaybhai and Bentyn, Zofia Antonina and Ignatov, Dmitry and Timofte, Radu},
  title        = {LEMUR Neural Network Dataset: Towards Seamless AutoML},
  howpublished = {\url{https://github.com/ABrain-One/nn-dataset}},
  year         = {2024},
}
```

## Licenses

This project is distributed under the following licensing terms:
<ul><li>for neural network models adopted from other projects
  <ul>
    <li> Python code under the legacy <a href="https://github.com/ABrain-One/nn-dataset/blob/main/Doc/Licenses/LICENSE-MIT-NNs">MIT</a> or <a href="https://github.com/ABrain-One/nn-dataset/blob/main/Doc/Licenses/LICENSE-BSD-NNs">BSD 3-Clause</a> license</li>
    <li> models with pretrained weights under the legacy <a href="https://github.com/ABrain-One/nn-dataset/blob/main/Doc/Licenses/LICENSE-DEEPSEEK-LLM-V2">DeepSeek LLM V2</a> license</li>
  </ul></li>
<li> all neural network models and their weights not covered by the above licenses, as well as all other files and assets in this project, are subject to the <a href="https://github.com/ABrain-One/nn-dataset/blob/main/LICENSE">MIT license</a></li> 
</ul>

#### The idea of Dr. Dmitry Ignatov
