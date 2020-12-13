# ResNet

resnet은  마이크로소프트의 팀이 개발한 네트워크입니다. 이는 학습에 있어서 층이 지나치게 깊으면 학습이 잘 되지 않고, 성능이 떨어지는 경우가 많습니다. resnet은 이런 문제를 해결하기위하여 skip connection을 도입하게 됩니다. 스킵연결이란 합성곱 계층을 건너 뛰어 출력에 바로 더하는 구조를 말합니다. <img src="..\img\resnet.png" alt="resnet" style="zoom:200%;" />

기존의 신경망은 입력값 x 를 출력이 $F(x)$ 가 되나. 스킵 연결을 통해 $F(x)+x$ 가 되게 됩니다. ResNet은 이 값을  최소로 하는 것을 목적으로 합니다. 따라서 $F(x)$가 0에 가깝게 만드는 것이 목적이 됩니다. $H(x) = F(x)+x$라 하면 $ F(x)=H(x) -x$ 이 되고 이를 잔차라고 합니다. 즉 이 잔차(residual)를 최소로 해주는 것이므로 ResNet이라는 이름이 붙게 되었습니다.



##  ResNet의 구조

ResNet에서는 convolution layer에서는 3*3 kernel을 사용하였고 max pooling은 사용을 안하였습니다. 그래서 출력 feture 크기를 줄일 때는 stride를 2로 설정 하여 진행하였습니다.

그리고 2개의 layer 마다 스킵연결을 진행 하였습니다.

<img src="C:\Users\jm\Documents\GitHub\DL-study\img\ResNet.jpg" alt="ResNet" style="zoom:50%;" />