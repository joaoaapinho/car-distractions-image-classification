<p align="center">
  <img src="https://github.com/joaoaapinho/car-distractions-image-classification/assets/114337279/d6060226-b9cb-46ba-af04-625e9a1d4cb5" alt="Small logo" width="20%">
</p>
<h3 align="center">Image Classification: Distracted Driver Actions</h3>

<p align="center"><b>Professor:</b> Hind Azegrouz</p>

<p align="center"><b>Done by:</b> João André Pinho</p>

<h2> 👁‍🗨 Overview </h2>

<h3>🏢 Assignment Description</h3>

In this assignment we are given driver images, each taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc). The goal is to predict what the driver is doing in each picture.

The 10 classes to predict are:

c0: safe driving
c1: texting - right
c2: talking on the phone - right
c3: texting - left
c4: talking on the phone - left
c5: operating the radio
c6: drinking
c7: reaching behind
c8: hair and makeup
c9: talking to passenger

<h3>❔ Problem Definition</h3>

**"Can a Computer Vision Model be built in order to be able to detect and properly classify different drivers behaviors while driving?"**

<h2> 💻 Technology Stack </h2>

Python, Tensorflow, Keras, Pandas, NumPy, Scikit-Learn, Matplotlib.

<h2> 🔧 Methodology </h2>

The methodology for this project involved the development and evaluation of multiple deep learning models for the task of image classification. Initially, **two convolutional neural networks (CNNs) were built from scratch**, utilizing layers such as convolutional layers, pooling layers, dropout layers, and fully connected layers. These CNN models served as the baseline for the performance comparison. The architecture and hyperparameters for these models were adjusted manually, with the aim of maximizing the model's ability to correctly identify the classes in the dataset.

**Model 1: CNN from Scratch**
Architecture: Custom CNN
Batch Size: 100
Optimizer: Adam
Learning Rate (α): 0.001

**Model 2: CNN from Scratch (improved hyperparameters and image augmentation)**
Architecture: Custom CNN
Batch Size: 64
Optimizer: Adam
Learning Rate (α): 0.005
Data Augmentation: Yes

Subsequently, a **transfer learning** approach to leverage pre-trained models was adopted, notably with models such as **VGG16, ResNet50, and MobileNetV2**. Some of the models were utilized twice, in order to perform hyperparameter tuning. Transfer learning allowed the project to benefit from features already learned from massive datasets like ImageNet, thus enhancing the model's performance. These models were fine-tuned by adding custom fully connected layers on top, tailoring the models to our specific classification task. During this stage, different hyperparameters were experimented with, including learning rates, batch sizes, and optimizers, to further optimize the model's performance.

**Model 3: VGG16 with Image Augmentation**
Pre-trained Model: VGG16
Batch Size: 60
Optimizer: Adam
Learning Rate (α): 0.005
Data Augmentation: Yes

**Model 4: VGG16 (improved hyperparameters with image augmentation)**
Pre-trained Model: VGG16
Batch Size: 60
Optimizer: Adam
Learning Rate (α): 0.005
Data Augmentation: Yes

**Model 5: ResNet50**
Pre-trained Model: ResNet50
(Additional settings and hyperparameters not specified)

**Model 6: MobileNetV2 with Image Augmentation**
Pre-trained Model: MobileNetV2
Batch Size: 150
Optimizer: SGD
Learning Rate (α): 0.003
Data Augmentation: Yes

<h2> 🔧 Main Conclusions </h2>

Throughout the project, it could be observed that all models provided high precision, recall, and accuracy, indicating robust model performance. **Models 2 and 4** were the only ones to flawlessly identify all test images, suggesting superior potential for larger datasets. This success was achieved through adjusting hyperparameters and implementing image augmentation techniques, reinforcing the importance of these strategies in improving model performance.

Despite this all models, barring Model 5 (possibly due to a short number of epochs and a high learning rate), consistently showcased the effectiveness of optimizers such as Adam, Adagrad, and SGD, each contributing to different learning capabilities and training speeds. Regardless of that, a common trend across models was the need to balance learning rate against training speed and model accuracy, pointing to the importance of optimization in achieving efficient model performance.

**Results Summary:**

| Model   | Accuracy  | Time Taken | Best Performer |
|---------|----------:|-----------:|:----------:|
| Model 1 | 0.9915    | 17' 19''         | No         |
| Model 2 | 0.9783    | 19' 30''         | Yes        |
| Model 3 | 0.9726    | 84' 26''         | No         |
| Model 4 | 0.9790    | 40'         | Yes        |
| Model 5 | 0.2146    | 40' 44''         | No         |
| Model 6 | 0.9753    | 40'         | No         |


<h2> 🔧 Limitations and Improvement Opportunities </h2>

📈 **Further Hyperparameter Tuning:** Model 2's performance indicates that more gains could be achieved with additional hyperparameter tuning.

🎛️ **Refine Data Augmentation:** While image augmentation was beneficial, some noise from outside the car still influenced the analysis, implying the need for refining this process.

🔄 **Adjust Learning Rate and Increase Epochs:** Model 5's lower performance points to the need for adjusting parameters like learning rate or increasing the number of epochs.

⚖️ **Balance Training Speed and Accuracy:** Across all models, a trade-off between training speed and accuracy was observed. Exploring different learning rates, optimizers, and architectures could better balance these aspects.

🕵️‍♀️ **Advanced Hyperparameter Tuning Techniques:** Techniques like Grid Search or Random Search can systematically explore many combinations of parameters to find the ones with the best performance.

🏗️ **Experiment with Different Architectures:** Trying out different architectures, especially those successful in similar tasks, could provide beneficial insights and improve the model's robustness and accuracy.

