# VGG-16

![vgg-16](..\img\vgg configu.jpg)

VGG-16은 D에 해당하며, 16개의 층으로 구성된 모델이다.
13개의 Convolution Layers와 3개의 Fully-connected Layers로 구성되어 있다.
zero-padding은 1이며 convoultion Layers의 stride는 1로 설정되어있다. 
또한, Polling Layers의 stride는 2*2의 maxpooling을 적용하였고, stride는 2로 설정되어있다.

![vgg configu](..\img\vgg-16.jpg)

input_data : input data의 shape 은 (224\*224\*3)의 크기를 갖고있으며, 이 값이 연산 메모리가 된다. (150,528)

convolution_layer_1 : 들어온 data의 채널수는 3이기때문에 fliter의 채널수 또한 3이다. 따라서 fliter의 shape은 (3\*3\*3)의 크기를 갖고있으며, 이 fliter가 64개가 있다. 따라서 이때 parameter는 (3\*3\*3)\*64 = 1,728개가 된다.
출력되는 feature map의 크기는 (224\*224\*64)가 되어 필요한 연산량은 3,211,264이다.

convolution_layer_2 : 첫번째 layer를 통과하며 채널수가 64채널로 변경되었다. 따라서 fliter의 채널수는 64가 되어 fliter의 shape은 (3\*3\*64)가 된다. 여기에서도 이러한 fliter를 64개가 있어, 이때 parameter는 (3\*3\*64)\*64 = 36,864가 된다. 출력된 feature map크기가 layer1번과 같기때문에 연산량도 같다.

pooling_layer_1 : pooling의 크기는 (2\*2)이며, stride=2로 설정되어 있는 maxpooing layer를 통과하며 feature map의 크기를 반으로 줄여주어 통과한 feature map의 shape은 (112\*112\*64)이다. 

convolution_layer_3 : 들어온 data의 shape이 (112\*112\*64)이므로 fliter의 채널이 64인데, 이러한 fliter가 128개가 있다. 따라서 parameter는 (3*3\*64\*128) = 73,728개가 된다. 출력되는 feature map은 (112\*112\*128)이 되어 필요한 연산량은 1,605,632이다.

convolution_layer_4 : 세번째 layer를 통과하며 채널수는 128이 되었다. 따라서 fliter의 채널수도 128이 되고, 이러한 fliter가 128개가 있으므로, parameter는 (3\*3\*128)\*128 = 147,456개이다. 이 층도 3번째 layer와 출력되는 featrue map이 같기때문에 필요한 연산량 또한 같다.

pooling_layer_2 : pooling layer를 통과하며 feature map의 shape이 절반으로 줄어들어 (56\*56\*128)이 된다.

convolution_layer_5 : 이때 fliter의 shape은 (3\*3\*128)인데, 이러한 fliter가 256개가 있어 parameter의 갯수는 (3\*3\*128)*256 = 294,912개가 된다. 출력되는 feature map은 (56\*56\*256)크기로 필요한 연산량은 802,816이다.

convolution_layer_6 : 5번째 layer를 통과하며 채널수가 256으로 늘었다. 따라서 parameter는 (3\*3\*256)\*256 = 589,824개이다. 출력되는 feature map이 layer5와 같기때문에 필요한 연산량 또한 동일하다.

convolution_layer_7 : 이 layer에서의 parameter는 6번째 layer와 같은 589,824개이다. layer5,6과 출력되는 feature map이 같아 필요한 연산량도 같다.

pooling_layer_3 : 이 pooling layer를 통과하며 크기를 절반으로 줄이게 된다. 따라서 28\*28\*256의 크기가 된다.

convolution_layer_8 : fliter의 크기는 (3\*3\*256)이며 이 fliter가 512개가 있다. 따라서 parameter는 (3\*3\*256)\*512 = 1,179,648개가 된다. 이 층에서 출력되는 feature map의 크기는(28\*28\*512)이며 연산량은 401,408이다.

convolution_layer_9 : 8번째 layer를 통과하며 채널수는 512가 되었다. 따라 fliter의 크기는 (3\*3\*512)되어, parameter는 (3\*3\*512)\*512 = 2,359,296개가 된다. 출력되는 feature map이 layer8과 같기때문에 연산량 또한 같다.

convolution_layer_10 : parameter의 갯수는 9번째 layer와 같다. 이 층에서도 역시 연산량은 8,9 번째 layer와 동일하다. 

pooling_layer_4 : pooling layer를 통과하며 이미지 크기가 줄어들어 (14\*14\*512)가 된다.

convolution_layer_11 : fliter의 크기는 (3\*3\*512)로 9번째와 10번째 layer와 같다. 이 층에서 출력되는 featur map은 (14\*14\*512)이며 이때 연산량은 100,352이다.

convolution_layer_12 : 11번째 layer의 parameter수가 같다. 여기에서 출력되는 feature map은 11layer와 같기때문에 연산량또한 같다.

convolution_layer_13 : 11번째, 12번째 layer의 parameter수가 같으며, 출력되는 featur map과 연산량이 같다.

pooling_layer_5 : 이 layer를 통과하며 이미지 크기가 줄어들어 (7\*7\*512)가 된다.

Fully-Connected-Layer_1 : convolution layer를 통과한 feature map을 FC로 만들어 준다. 최종적으로 출려된 이미지의 크기는 (7\*7\*512)이며, 4096개의 뉴런으로 구성해준다. 이때의 parameter 갯수는 (7\*7\*512)\*4096 = 102,760,448개이다.

Fully-Connected-:ayer_2 : FC_1을 통과하며 이미지의 크기는 (1\*1\*4096)이 되었다. 따라서 FC_2에서의 parameter 갯수는 4096*4096 = 16,777,216개가 된다.

Fully-Connected-Layer_3 : FC_3에서는 1000개의 뉴런이되며 이때 parameter의 갯수는 4096*1000 = 4,096,000개가 된다.

따라서 convolution layer에서 순전파를 진행하는데 연산량은 총 13,547,520이며 한개의 노드당 4bytes를 소요하므로 13,547,520\*4 = 54,190,080 \~=51.68MB이다. 또한 역전파를 포함하게되면 51.68MB\* = 103.36MB이다.

또한 학습가능한 파라미터의 갯수는 모두 합쳐서 138,344,128개이다.