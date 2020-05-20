# Image Captioning
In this project, I try to create a sentence which can describe an image.<br>
The project is based on Two main parts : the encoder and the decoder.<br>
The encoder is simply a pretrainded resnet from which we remove the classification FFNN and add another FFNN to be the input to the decoder.<br>
The decoder is an LSTM archirecture which tries to create a sequence of words based on the input of the encoder.<br>
