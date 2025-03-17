# Image-Segmentation-based-on-face-identification-Desktop-Application

This Desktop Application is developed to target the Need of photographers. this application takes the hundreds of images and videos as an input and filter out and clustered the images into different folders based on the faces identified. this projects involves 3 main implementation stages. That are:
- Segment out the a clear face from the image or video which can be used in identification process. segmentation process is done using the MTCNN model
- Second stage is the identification of the images similiar face patterns. for this purpose a RESNET model's convolutional layers are used for face feature extraction, these features are further used in clustering process
- the last stages is the clustering process. this stage takes face features extracted from above stage, and used to generated clusters / folders. where each folder compresses the images of a single person. 
