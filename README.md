<div>
	<div class="tex">
    ![CNN](https://user-images.githubusercontent.com/28722660/57572107-ca6ff600-7433-11e9-908b-c3e391896864.png)
		<center><h1 style="color: white"><u>Introduction</u></h1></center><br>
		<p style="color: white;">
			<strong>Building a model that can Classify whether a person has malaria or not based on cell images using convolutional neural networks (CNN)</strong><br>
			This project is made for the Engineering Exploration CS201 by -
			<ul style="color: white;">
				<li>Shailvi Shukla - 1711981273</li>
				<li>Varun Sangwan - 1711981332</li>
			</ul>
			<p style="color: white; text-align: justify;">It is assumed that the reader knows about how multilevel perceptron network works, basic concepts and working of a neural network, basic knowledge about linear algebra and calculus (mostly differentials) Convolutional neural networks is a type of neural network itself.</p>
			<p>
				<h4 style="color: white;">The Dataset</h4>
				<p style="color: white; text-align: justify;">The dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells. An instance of how the patient-ID is encoded into the cell name is shown here with: “P1” denotes the patient-ID for the cell labeled “C33P1thinF_IMG_20150619_114756a_cell_179.png”. For this project we are not concerned with patient id's.<br>
					This dataset is provided by US National Library of Medicine
					<ul>
						<li style="color: white">Link - <a href="https://ceb.nlm.nih.gov/repositories/malaria-datasets/">https://ceb.nlm.nih.gov/repositories/malaria-datasets/</a></li>
					</ul></p>
					<h5 style="color: white;">Examples</h5>
					<h6 style="color: white;"><i>Parasitized : <br> Below are some examples of cell image dataset showing cells that are infected by maleria</i></h6>
					![examples](https://user-images.githubusercontent.com/28722660/57572108-ca6ff600-7433-11e9-8a64-6dfafe3fd19e.png)
					<h6 style="color: white;"><i>Uninfected : <br> Below are some examples of cell image dataset showing cells that are uninfected by maleria</i></h6>
					![examples1](https://user-images.githubusercontent.com/28722660/57572109-ca6ff600-7433-11e9-8863-1fe56d49de7c.png)
				</p>
				<p style="color: white;">
					We'll start by importing some libraries for matrix math, Image manipulation data viualization and data preprocessing the libraries used are as follows - 
					<ul style="color: white;">
						<li>Numpy - For matrix math and storing images as arrays</li>
						<li>Glob - Storing image names for easy iteration</li>
						<li>Image - For opening images</li>
						<li>Matplotlib - For data visualization</li>
					</ul>
				</p>
				<p style="color: white; text-align: justify;">
					All the dependencies and libraries used are listed below. In order to run the application, one needs to install these dependencies. Since, Convolutional Neural Networks require many computation expensive operations, training process can be quite slow if done on a CPU, instead using parallel computing power of a GPU we can make the training/prediction process fast. For this project we are going to use various libraries that use GPU instead of CPU. It is recommended to have a GPU with more than 4 gigs of memory while training because the dataset contains 27,558 cell images. If the pre-trained model is used it'll work fine on smaller GPU’s also. if there's no GPU or a Non-Nvidia GPU is present then TensorFlow should be installed instead of TensorFlow-gpu also in this case there's no need to install cuDNN and cuda.
					<br></p>
					<div style="background-color: black;">
						<div class="code">
							<code>
								<br><br>
								pip install Keras  <p style="color: white;"><i>Keras is a high-level neural networks API. that will work on top of tensorflow</i></p>
								pip install cudnn  <p style="color: white;"><i>Nvidia's deep learning api used by tensorflow to train models on gpu. Also, install cuda from Nvidia's site in order for cudnn to work.</i></p>
								pip install h5py  <p style="color: white;"><i>h5 is a file format that we'll use to store the model once trained.</i></p>
								pip install tensorflow-gpu  <p style="color: white;"><i>tensorflow is a open source Machine learning platform by Google.</i></p>
								pip install numpy  <p style="color: white;"><i>For fast array manipulation</i></p>
								pip install flask  <p style="color: white;"><i>For hosting the model on localhost</i></p>
								<br><br>
							</code>
						</div>
					</div><br>
				</p>
				<h1 style="color: white"><u>Convolutional neural networks</u></h1><br>
				<p style="color: white; text-align: justify">
					In deep learning, a convolutional neural network is a class of deep neural networks that are used to analyse visual imagery. They are proven very effective in areas of image recognition and classification, other non-neural network machine learning techniques such as K nearest neighbours, support vector machines etc, don’t scale when it’s come to visual classification or recognition. For this project we built a convolutional neural network with eight layers, every single one of them is described later in this document. Neural networks are very robust but in order to train the neural network it requires high computational expenses and generally a lot of data. Convolutional neural networks were inspired by biological animal visual cortex. The main problem with image classification is that computer don’t perceive images like we humans do, or computer’s images are a matrix of just random numbers or pixel values. For example, imagine a cat, now it can be of any shape it can be of any colour, might be sitting on a bench, on the floor, the fur might be of different colour, the cat itself might be of different breed. Now imagine hard programming all these features to classify, if there is a cat present in a given image or not it’s almost impossible to hard code features that can check if there is a cat present in image or not also if you take into consideration the various lighting effects, shadows, camouflages the problem just becomes insanely difficult to solve with regular programming. Even if we use machine learning approaches like support vector machines or logistic regression or K nearest neighbours, we cannot select any features that can help us in classification procedure, since the input is consists of random values or pixel values that change image to image and generally doesn’t carry much similarities. One possible solution might be to use a fully connected neural network with each pixel of the image acting as the first layer followed by some hidden layers and our classification neurons at the last layer. There is a problem with this approach though, this might work with very small dimension images but in real life we have very high resolution images even if you consider the 300 x 300 pixel image, number of neurons in our first layer will be 90,000 and let’s suppose we take 128 neurons in our second layer (hidden layer), and since we are building a fully connected neutral network each of the neuron is connected with every other neuron of next layer and this connection we denote by a weight, we are just into second layer of our neural network and we already have to tweak 1,15,20,000 weights, we can already conclude that training such a network will require enormous amount of computation which is not feasible. Certainly, we require something that can find a better solution to all these problems, that’s when convolutional neural networks kicks in. <br><br>
					Inspired by the visual cortex of animals, there was a paper released by Hubel and Wiesel in 1950s and 1960s about visual cortexes of cats and monkeys, how they behave and respond to visual field. In 1980, The "neocognitron" was introduced by Kunihiko Fukushima. It was inspired by work of Hubel and Wiesel. To break down the complex input size it included convolutional layers, down sampling layers. All the layers and activation functions used in this project will be explained in detail. At that time there was not enough computation power to train such networks. As the computation power increased, the amount of data increased these machine learning algorithms gained popularity, with various optimisations convolutional neural networks were able to achieve high accuracy in areas of image classification and recognition. One live example of how good these networks perform is in your mobile phone gallery google photos or ios gallery both have the features of classifying people based on their faces and they perform well. For this project of malaria image classification based on cell images we had a pretty good data set of more than 27,000 images belonging to 2 classes approximate more than 13,000 images for each class. In total the network uses eight layers, also the images are of different sizes height and width wise so we need to pre-process the data in order to use them as input to our first layer. The images are down sampled to 128 x 128 pixels initially when fed to the network. We used kereas to build the model because it makes it really easy to add layers and it can work on top of Tensorflow GPU. The network was trained on a GPU, and it took approximate 30 minutes in total to train the network fully. If done on CPU the training time can take as long as 10 hours, of course depending on the type of hardware used. The first two layers also the next two layers of our network are convolutional layers and Max pooling layers respectively and these layers required multiple matrix multiplication operations that are independent to each other that means they can be distributed to various cores of the graphical processor and that makes the process very fast, Instead of sequential approach of using CPU or getting constrained by the few number of cores present in a CPU. We used  a sequential model of kereas.models library, we used various functions like models, add layers etc to build our model. Below is a list of layers that are used to build the model.
					<ul style="color: white;">
						<li>Convolutional layer</li>
						<ul><li><small>This layer contains 120 filters and because input images are of shape (a,b,3) where a and b is varying for each image, each image is resized to 128x128x3 shape with 3 being the rgb channels of image. Each filter is 3x3 matrix. Kereas provide Conv2d function for convolving a rgb depth field image. Relu activation function is used, Explanation of convolation operation is provided later in this document. </small></li></ul>
						<li>Maxpooling layer</li>
						<ul><li><small>Max pooling layer to make the filters more small without loosing much of information, size of Maxpooling matrix is 2x2</small></li></ul>
						<li>Convolutional layer</li>
						<ul><li><small>220 filters of shape 3x3 and relu activation</small></li></ul>
						<li>Maxpooling layer</li>
						<ul><li><small>Size of Maxpooling matrix is 2x2</small></li></ul>
						<li>Convolutional layer</li>
						<ul><li><small>320 filters of shape 3x3 and relu activation</small></li></ul>
						<li>Flatten layer</li>
						<ul><li><small>Flatten layer flattens each filter to be connected to dense layer. This is the part where fully connected neural network resides.</small></li></ul>
						<li>Dense layer</li>
						<ul><li><small>320 neurons with relu activation.</small></li></ul>
						<li>Dense layer</li>
						<ul><li><small>1 Neuron that gives a binary output. Which class the image belongs to 0 or 1. sigmoid activation function is used.</small></li></ul>
					</ul>
					<br>
				</p>
				<div style="background-color: black;">
					<div class="code">
						<code>
							<br><br>
							model = models.Sequential() <p style="color: white;"><i>Creating a object of Sequential class.</i></p>
							model.add(layers.Conv2D(120,(3,3),activation='relu', input_shape=(128,128,3)))  <p style="color: white;"><i>First Convolution layer</i></p>
							model.add(layers.MaxPooling2D((2,2)))  <p style="color: white;"><i>Maxpooling layer</i></p>
							model.add(layers.Conv2D(220,(3,3),activation='relu'))  <p style="color: white;"><i>Second Convolution layer</i></p>
							model.add(layers.MaxPooling2D((2,2)))  <p style="color: white;"><i>Maxpooling layer</i></p>
							model.add(layers.Conv2D(320,(3,3),activation='relu'))  <p style="color: white;"><i>Third Convolution layer</i></p>
							model.add(layers.Dense(320,activation='relu'))  <p style="color: white;"><i>Dense layer with 320 Neurons</i></p>
							model.add(layers.Dense(1,activation='sigmoid'))  <p style="color: white;"><i>Output layer with single neuron.</i></p>
							<br><br>
						</code>
					</div>
				</div><br>
				<p style="color: white; text-align: justify">
					<h3 style="color: white">Convolution Layer</h3>
				</p>
				<p style="color: white; text-align: justify">
					The first layer of the network is convolutional layer. Convolutional neural networks make use of filters to detect what features such as edges, are present in the image. A filter is just the metrics of values. The first layer contains 120 filters of 3 x 3 size, they are initialised with random values and as we train the network the filter learns what to detect in our image may be edges, may be textures. The filters perform a convolution operation, which is simply an element twice product with the image. So, suppose our input image is of size 300 x 300 pixels and our filter sizes 3 x 3 pixels and what are convolutional operation will do is to start from the top of the image and multiply the whole filter pixel values and then move on to the next great and keep repeating the process until the whole image is convolved bearing in mind the multiplication operation is element twice not a the traditional matrix multiplication. The values of filter or the weights will look for a feature and the output image will be the result of weather that feature was present in the image or not, imagine a filter is looking for an horizontal edge, if a horizontal edge is present in input image the sum of all the values in the output matrix by convolving the filter will be a huge number and passing that huge numbers through activation function will determine the activity of that filter. Now there are some problems that we face using this, approach one is if the size of the input image is too small, we will lose a lot of information by passing the image down the network to conquer this problem we use a technique called zero padding.</p><br>
					<ul style="color: white">
            <h2>References</h2>
						<li><a href="http://cs231n.github.io/convolutional-networks/">http://cs231n.github.io/convolutional-networks/</a></li>
						<li><a href="https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1">https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1</a></li>
						<li><a href="https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/">https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/</a></li>
						<li><a href="https://en.wikipedia.org/wiki/Convolutional_neural_network">https://en.wikipedia.org/wiki/Convolutional_neural_network</a></li>
						<li><a href="http://flask.pocoo.org/">http://flask.pocoo.org/</a></li>
						<li><a href="http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/">http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/</a></li>
					</ul>
			</div>
		</div>
