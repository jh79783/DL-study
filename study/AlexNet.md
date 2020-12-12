# AlexNet

지금까지는 작은 dataset으로 성능을 내었는데, 현실에서는 더 복잡하기 때문에 현실의 object를 잘 인식하기 위해 큰 dataset이 필요하며, 이들을 빠르고, 잘 학습하여 classfication을 해야한다. 따라서 CPU대신 GPU를 사용하였고 2만2천개의 카테고리가 존재하며 1500만개 이상의 레이블된 고해상도 이미지 dataset인 ImageNet을 사용하였다. 
따라서 마지막 FC layer는 다클래스 분류를 위해 softmax를 사용하였으며, convolution layer에서는 ReLU를 사용하였다.
이 당시에는 tanh를 많이 사용하였는데 tanh는 시간이 오래걸린다는 단점이 있었다.

ImageNet의 해상도는 사진마다 다르기 때문에 이를 256\*256으로 resize하여 사용하였다.

## 구조

![AlexNet](..\img\AlexNet.jpg)

기본적인 구조는 LeNet과 크게 다르지 않지만, 2개의 GPU를 사용햐 병렬로 연산을 수행하였다.
총 8개의 layer로 구성되어 있는데, 5개의 convolution layer와 3개의 fully-connected layer로 구성되어 있다.
특이한 점은 세번째 convolution layer에서 두 채널의 feature map이 모두 연결되어 있다.

위에서 256\*256으로 resize를 사용하였다고 했는데 구조를 보면 227\*227이다.(이미지가 잘못되어 있다.)

### 1번째 Convolution layer

input : 227\*227\*3
filter : 11\*11\*3 이 96개 
stride : 4
zero-padding : x
feature map : 55\*55\*96
parameter : 11\*11\*3\*96 = 34,848

### 1번째 Max pooling layer

pooling size : 3\*3
stride : 2
feature map : 27\*27\*96

### 2번째 Convolution layer

filter : 5\*5\*48 이 256개
stride : 1
zero-padding : 2
feature map : 27\*27\*256
parameter : 5\*5\*48\*256 = 307,200

### 2번째 Max pooling layer

pooling size : 3\*3
stride : 2
feature map : 13\*13\*256

### 3번째 Convolution layer

filter : 3\*3\*256 이 384개
stride : 1
zero-padding : 1
feature map : 13\*13\*384
parameter : 3\*3\*256\*384 = 884,736

### 4번째 Convolution layer

filter : 3\*3\*192가 384개
stride : 1
zero-padding : 1
feature map : 13\*13\*384
parameter : 3\*3\*192\*384 = 663,552

### 5번째 Convolution layer

filter : 3\*3\*192가 256개
stride : 1
zero-padding : 1
feature map : 13\*13\*256
parameter : 3\*3\*192\*256 = 442,368

### 3번째 Max pooling layer

pooling size : 3\*3
stride : 2
feature map : 6\*6\*256

### 1번째 Fully-connected layer

parameter : 6\*6\*256\*4096 = 37,748,736

### 2번째 Fully-connected layer

parameter : 4096\*4096 = 16,777,216

### 3번째 Fully-connected layer

parameter : 4096\*1000 = 4,096,000

total param : 60,954,656



## Local Response Normalization

각각 convolution layer를 통과하고 local response normalization을 시행하였다. 
수렴속도를 높이기 위함이다. 
ReLU는 항상 0보다 큰 값이 출력된다. 그리고 Conv와 pooling에서 특정한 값이 주변보다 크면 주변 픽셀에 영향을 미치기 때문에 이러한 영향을 방지하기위해 인접한 채널에서 같은 위치에 있는 픽셀 n개를 통해 정규화 하는것을 말한다.
현재는 다른 기법으로 인해 거의 사용하지 않는다.

## Overfitting

overfitting을 줄이기 위해 여러가지 기법을 사용하였다.

### Data Augmentation

- horizontal reflection

  위에서 256\*256의 크기로 resize하여 사용한다 하였는데, 실제 사용된 이미지의 크기를보면 227\*227이다. 이는 256\*256이미지로부터 227\*227 크기의 이미지를 랜덤하게 crop해서 사용하였기 때문이다.
  또한 테스트에는 5개의 224\*224의 패치와(4개의 구석패치, 1개의 중앙패치) 이들을 반전시킨 패치들을 합하여 총 10개의 패치를 추출하고 softmax를 통과해 만들어딘 예측을 평균화하여 하나의 예측으로 만들었다.

- PCA color augmentation

  train 이미지에서 RGB채널의 강도를 변경하였다고 하는데 이부분은 잘 이해가 되지 않는다.

### Dropout

0.5의 확률을 갖고 각 은닉 뉴런의 출력을 0으로 만들게 된다.
이 방법으로 dropout된 뉴런들은 순전파,역전파에 참여하지 않는다.
이 논문에서는 1,2번째의 fully-connected layer에 사용하였다.