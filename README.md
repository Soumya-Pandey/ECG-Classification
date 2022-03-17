# ECG-Classification

Cardiovascular diseases which affect the heart or blood vessels causing abnormal heart
rhythm are the leading cause of death in the world and the electrocardiogram (ECG)
helps in their diagnosis. Automatic diagnosis of 12 lead ECGs helps to analyse the type
of arrhythmia in a patient. The electrical impulses that coordinate the heart beats don't
work properly. The automatic detection is done using various machine learning models.
Deep Residual Neural Network(RNN) is being used in the proposed work in analysing 4
types of arrhythmia namely Normal sinus Rhythm, Right Bundle Branch Block (RBBB),
Left Bundle Branch Block (LBBB), and Premature Ventricular Contractions (PVCs). In
this paper, the RNN network consists of a convolutional layer (Conv) followed by four
residual blocks with two convolutional layers per block. The output of each convolutional
layer is rescaled and hence we get the classification of type of Arrhythmia by analysing
12 Lead ECG.

**INTRODUCTION**

An electrocardiogram is a picture of the electrical conduction of the heart. These
electrode wires are connected to the ECG machine with recordings from 12 different
locations on the surface of the body and hence called 12-lead ECG. In this paper, we
demonstrate the effectiveness of RNN for automatic 12-LEAD ECG classification. The
resultant electrocardiogram represents a mesh of cardiac electrical activity from the
atrium and ventricles, dependent on the direction and magnitude of the electrical

depolarization as it spreads throughout the heart. It is recorded as an output of waveforms
that your physician can print onto paper or record on the monitor. The 12-lead ECG is a
graphic representation of the electrical activity of the heart on two planes.

The six limb leads (I, II, III, aVR, aVL, and aVF) provide a view of the heart from the
edges of a frontal plane as if the heart were flat. The standard limb leads (I, II, and III) are
bipolar and measure the electrical differences between the combination of three limbs:
the right arm, left arm, and left foot. Bipolar leads have a positive and negative pole in
which the electrical current is measured as electrons move from negative to positive. The
augmented leads aVR, aVL, and aVF are unipolar and record the electrical difference
between the right and left arms and the left foot utilizing a central negative lead.

**IMPLEMENTATION**

Contributions to this study were to use convolutional neural networks similar to the residual
network, but adapted to unidimensional signals. This architecture allows DNNs to be efficiently
trained by including skip connections. We have adopted the modification in the residual block
which places the skip connection. All ECG recordings are resampled to a 400 Hz sampling rate.
The ECG recordings, which have between 7 and 10 s, are zero-padded resulting in a signal with
4096 samples for each lead. This signal is the input for the neural network. The network consists
of a convolutional layer (Conv) followed by four residual blocks with two convolutional layers
per block. The output of the last block is fed into a fully connected layer (Dense) with a sigmoid activation function, σ, which is used because the classes are not mutually exclusive (i.e. two or
more classes may occur in the same exam). The output of each convolutional layer is rescaled
using batch normalization, (BN)49, and fed into a rectified linear activation unit (ReLU).
Dropout50 is applied after the nonlinearity to achieve results.

CNNs, a deep feed-forward artificial neural network, was inspired by biological processes and
has successfully been applied to analyzing data and localization. Trained dedicated simple RNNs
for every patient to automatically classify ECG signals. The final results revealed the model was
generic because of its simple structure and parameter invariant property.

