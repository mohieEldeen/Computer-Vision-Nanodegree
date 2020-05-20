# Facial Keypoint Detection 

in this project, Firstly, I used haar cascaded face detector then I used a CNN architecture on the detected faces to predict 68 points on each face.<br>
These keypoints include points around the eyes, nose, and mouth on a face.<br>
the architecture used can be found in model.py <br>
the loss used is smoothL1 regression loss. <br>
the optimization used is adam optimizer. <br>
