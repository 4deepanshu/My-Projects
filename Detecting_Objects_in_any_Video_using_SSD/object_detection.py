# Object Detection
# data file transforms images so that they are compatible with neural networks
# layers contains some tools for multi box detction
# ssd.py has the architecture of ssd model
# ssd300_mAP_77.43_v2.pth has the pre trained ssd model with weights and neural network

# Importing the libraries
import torch # backprop, NN and calculate gradient
from torch.autograd import Variable # convert tensors into torch variables
import cv2 # only to draw rectangles
from data import BaseTransform, VOC_CLASSES as labelmap # BT = right format of input image to feed in to NN , VOC_classes is a dictionary to encode classes into numbers, eg - dog = 1 and plane =2 as this method takes numbers and not texts
from ssd import build_ssd # constructor of ssd NN
import imageio # take images from video

# Defining a function that will do the detections
def detect(frame, net, transform): # We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images for getting the image into right format to get fed into NN = right dimensions and right colors, and that will return the frame with the detector rectangle.
    height, width = frame.shape[:2] # We get the height and the width of the frame; frame.shape gives height , width and no. channels()eg- grayscale has 1 channel and rgb has 3)
    frame_t = transform(frame)[0] # We apply the transformation to our frame and name it as frame_t.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the numpy frame into a torch tensor.
    x = Variable(x.unsqueeze(0)) # We add a fake dimension corresponding to the batch.
    y = net(x) # We feed the neural network ssd with the image and we get the output y.
    detections = y.data # We create the detections tensor contained in the output y.
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height].
    for i in range(detections.size(1)): # For every class:
        j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
        while detections[0, i, j, 0] >= 0.6: # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
            pt = (detections[0, i, j, 1:] * scale).numpy() # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle.
            j += 1 # We increment j to get to the next occurrence.
    return frame # We return the original frame with the detector rectangle and the label around the detected object.

# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.

# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('output.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.