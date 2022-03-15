# Pedestrian and Vehicle Distance Estimation

This is pedestrian and vehicle distance estimation network based on YOLOv3 and Monodepth.

## UDNet (Urban Distance Network)
UDNet has a structure of U-Net that consists of encoder, decoder and skip connection.

### Model
This is an entire structure of UDNet.
  
![전체 모델](https://user-images.githubusercontent.com/48514976/158316892-f26faddf-2dd5-4dda-9a3a-05f3ce972d94.png)
  
### Encoder
Encoder is composed of residual block, super resolution block and CBAM(Convolutional Bottleneck Attention Module).

#### Residual blocks(13 & 131)
![residual blocks](https://user-images.githubusercontent.com/48514976/158317307-41c35350-d64d-4f50-8bf8-8962f21ccb2e.png)

#### SR block
![srblock](https://user-images.githubusercontent.com/48514976/158319746-dfd3d042-6e54-45df-9aaa-89ff5b4b4301.png)

  
 ## Results
 ### Evaluation
 #### Detection
 Detection evaluation using KITTI detection dataset.
   
 |mAP|
 |-|
 |0.284|
 
 #### Depth estimation
 Depth estimation evaluation using KITTI radar depth ground truth.
 
 ##### Error
 |Abs Rel|Sq Rel|RMSE|RMSE log|
 |-|-|-|-|
 |0.276|10.413|14.911|0.354|
 
 ##### Accuracy
 |delta < 1.25|delta < 1.25^2|delta < 1.25^3|
 |-|-|-|
 |0.735|0.874|0.929|
 
 ### Visualization
   
![1_detect](https://user-images.githubusercontent.com/48514976/158317569-4d431d6a-9f56-4ed8-93ac-319c3d4c62e8.png)
![3_detect](https://user-images.githubusercontent.com/48514976/158317582-f285331a-0f29-4408-bdde-4e28c81de558.png)
![5_detect](https://user-images.githubusercontent.com/48514976/158317712-7ce08dbf-d3f3-4102-88bf-bcc1a4e468dd.png)
