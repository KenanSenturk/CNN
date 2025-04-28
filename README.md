# CNN
## 1. Giriş
Bu çalışmada, görüntü sınıflandırma problemleri için dört farklı yöntem uygulanmış ve karşılaştırılmıştır. Klasik LeNet-5 mimarisi ve iyileştirilmiş bir LeNet-5 varyantı MNIST veri seti üzerinde test edilirken, transfer öğrenme yöntemleriyle AlexNet modeli CIFAR-10 veri seti üzerinde değerlendirilmiştir.
Ayrıca, derin bir öğrenme ağı ile çıkarılan özelliklerin bir SVM (Destek Vektör Makineleri) modeli ile sınıflandırılması gerçekleştirilmiştir.

Bu projelerin amacı, farklı derin öğrenme tekniklerinin performanslarını ve sınıflandırma başarımını ölçmek ve birbirleriyle kıyaslamaktır.

## 2. Yöntemler
### 2.1 Kullanılan Veri Setleri

Veri Seti	Özellikler
MNIST	28x28 piksel, gri tonlamalı, 10 sınıf (0-9 rakamları)
CIFAR-10	32x32 piksel, renkli, 10 sınıf (uçak, araba, kuş, kedi vb.)
### 2.2 Modeller ve Mimariler
#### 2.2.1 LeNet-5 (Klasik)
2 adet Conv2D + MaxPooling katmanı
2 adet Tam Bağlantılı (Dense) katman
Aktivasyon: ReLU
Çıkış katmanı: 10 sınıflı softmax

#### 2.2.2 İyileştirilmiş LeNet-5
Klasik LeNet-5 yapısına ek olarak:
Batch Normalization: Her convolution sonrası uygulanmıştır.
Dropout: Fully Connected katmanlardan önce overfitting’i önlemek için eklenmiştir.

#### 2.2.3 AlexNet Transfer Learning
Önceden ImageNet üzerinde eğitilmiş AlexNet modeli kullanılmıştır.
Son katman CIFAR-10 veri setine uyacak şekilde 10 nöronla değiştirilmiştir.
Eğitim sırasında sadece Fully Connected katmanlar güncellenmiştir.

#### 2.2.4 CNN Özellik Çıkartımı + SVM
2 adet Conv2D + MaxPooling ile derin özellik çıkarımı yapılmıştır.
Sonrasında, çıkarılan özellikler RBF çekirdekli SVM ile sınıflandırılmıştır (burada zamandan tasarruf sağlamak için veri setinin 10,000 örneği kullanılmıştır).

### 2.3 Eğitim Detayları

Model	           ,         Optimizer	           ,         Epoch	       ,         Batch Size/
LeNet-5 (Klasik)	     :     SGD	               /           3	           /           64/
İyileştirilmiş LeNet-5	:    SGD	               /           3            /        	64/
AlexNet Transfer	   :    SGD + StepLR	         /           1   /                  	64/
CNN+SVM	     :      Adam (CNN kısmı) + SVM (scikit-learn) /	3 (CNN)	     /           128

### 2.4 Teorik Açıklamalar
Batch Normalization: İçsel covariate shift etkisini azaltarak öğrenmeyi hızlandırır.
Dropout: Eğitim sırasında rastgele nöronları kapatarak overfitting'i azaltır.
Transfer Learning: Büyük veri setlerinde eğitilmiş modellerin küçük veri setlerine aktarımıyla daha az veriyle daha iyi performans sağlar.
SVM: Özellikle düşük boyutlu fakat iyi ayırt edici özellikler elde edildiğinde etkili bir sınıflayıcıdır.
## 3. Sonuçlar
### 3.1 Eğitim Kayıpları ve Doğruluklar
Aşağıda her model için eğitim kaybı (loss) ve test doğrulukları özetlenmiştir:

Model	                          Test Doğruluğu (%)
LeNet-5 (Klasik)	     :              %98.55
İyileştirilmiş LeNet-5	:             %99.06
AlexNet Transfer (CIFAR-10)   :       %83.2
CNN + SVM (MNIST)	            :       %97

## 4. Tartışma
İyileştirilmiş LeNet-5, klasik LeNet-5'e göre %0.51 daha yüksek doğruluk sağlamıştır. Bu, BatchNorm ve Dropout'un eğitim stabilitesini ve genelleme kapasitesini artırdığına işaret etmektedir.

AlexNet transfer learning yöntemi CIFAR-10 veri seti üzerinde yüksek başarı sağlamıştır. Ancak sınıflar arası benzerlik nedeniyle bazı sınıflarda hata oranı gözlenmiştir.

Özellik çıkarımı sonrası SVM ile yapılan sınıflandırma, saf CNN eğitimine göre daha hızlı tamamlanmış ancak doğruluk biraz daha düşük kalmıştır.

## 5. Referanslar
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.

MNIST Dataset: http://yann.lecun.com/exdb/mnist/

CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
