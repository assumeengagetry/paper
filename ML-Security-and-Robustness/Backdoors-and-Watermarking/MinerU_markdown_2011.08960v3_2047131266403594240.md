# Deep Serial Number: Computational Watermark for DNN Intellectual Property Protection

Ruixiang Tang1, Mengnan Du2, Xia $\mathrm { H u } ^ { \mathrm { 1 ( \bigotimes \bigcirc ) } }$ 

1 Rice University, xia.hu@rice.edu 

2 New Jersey Institute of Technology, mengnan.du@njit.edu 

Abstract. In this paper, we present DSN (Deep Serial Number), a simple yet effective watermarking algorithm designed specifically for deep neural networks (DNNs). Unlike traditional methods that incorporate identification signals into DNNs, our approach explores a novel Intellectual Property (IP) protection mechanism for DNNs, effectively thwarting adversaries from using stolen networks. Inspired by the success of serial numbers in safeguarding conventional software IP, we propose the first implementation of serial number embedding within DNNs. To achieve this, DSN is integrated into a knowledge distillation framework, in which a private teacher DNN is initially trained. Subsequently, its knowledge is distilled and imparted to a series of customized student DNNs. Each customer DNN functions correctly only upon input of a valid serial number. Experimental results across various applications demonstrate DSN’s efficacy in preventing unauthorized usage without compromising the original DNN performance. The experiments further show that DSN is resistant to different categories of watermark attacks. 

Keywords: Watermark · Deep Neural Network · Intellectual Property Protection. 

# 1 Introduction

Deep neural networks (DNNs) have made significant progress in the last decade. The combination of large-scale training data and the rapid expansion of computational capabilities have facilitated the development of high-performance DNN models in numerous domains. However, training DNNs can be costly, involving the collection and labeling of large data sets and the allocation of considerable computing resources. Consequently, foundation DNN models are deemed valuable intellectual property by their owners. The substantial economic value of DNN models makes them attractive targets for malicious adversaries. For instance, numerous emerging online marketplaces trade deep neural networks that may be susceptible to theft by hackers. In another scenario, a legitimate customer might breach the licensing agreement by redistributing or selling DNNs to others. For instance, Meta’s latest large language model, LLaMA, initially accessible only through request, was leaked online via a 4chan torrent just a week after accepting access requests [32]. As the expenses associated with training DNN 

models continue to escalate, model providers are exploring various methods to assert ownership and protect their intellectual property from infringement. Consequently, the concept of digital watermarking has been adopted for deep learning models, which embeds secret identification information within DNN models, serving as evidence of model ownership verification. 

Currently, several approaches have been proposed to incorporate watermarks into DNNs. The rationale behind these watermarking strategies is to establish a tracking mechanism that enables legitimate parties to identify instances of stolen models. We can categorize these methods into two primary classes. The first class of methods embeds watermark information directly into the parameters of the DNN model [4, 28, 31]. For verification purposes, stakeholders must have access to the model parameters to examine the presence of the watermark’s statistical bias. However, this white-box access for verification is often impractical in many applications. The second set of approaches employs the backdoor insertion technique [8, 20, 29] to embed watermarks. In these cases, DNNs not only learn their original tasks but also retain outlier input-output pairs, which can be utilized for black-box ownership verification. However, these watermarking approaches are vulnerable to the commonly used transfer learning scenario, where adversaries can replace the top decision layers and train a watermark-free model based on the features extracted from the remaining network [3]. Another significant challenge facing existing watermarking methods is their vulnerability to various watermarking attacks, such as watermark suppression, removal [37], and overwriting [18]. This susceptibility to attacks further hinders their adoption in real-world applications. A robust DNN IP protection mechanism that can prevent unauthorized parties from using the stolen model is still missing. 

Inspired by the success of serial numbers in traditional software IP protection, we investigate the application of serial number embedding to safeguard DNNs. However, embedding serial numbers into DNNs presents several technical challenges. First, it remains unclear in what form serial numbers can be effectively incorporated into DNNs. Second, it is equally challenging to ensure that the serial numbers inserted remain robust against attacks from malicious adversaries. To address these concerns, we propose a novel DNN IP protection framework, DSN (Deep Serial Number). Specifically, we utilize the knowledge distillation method to initially train a teacher DNN and subsequently transfer its knowledge to customer DNNs. During the distillation process, a unique serial number is assigned to each student model. The customer DNN operates only when a user inputs a valid serial number. As a result, DSN effectively prevents stolen models from being exploited by unauthorized parties. Additionally, the embedded serial number functions as a robust tracking tag, similar to previous watermarking approaches. Experimental results from various applications reveal that the proposed DSN method successfully inhibits unauthorized use while maintaining the original DNN performance. Further experimental analyses demonstrate that DSN is resilient against different attack strategies, even when adversaries have white-box access to the DSN framework. The main contributions of this paper are summarized as follows: 

– We propose DSN, a novel IP protection framework for DNNs designed to prevent stolen models from being deployed by unauthorized third parties. 

– Experiments carried out on real-world datasets demonstrate that DSN effectively prevents unauthorized applications without sacrificing DNN performance on the original tasks. 

– Experimental studies further reveal that DSN is robust against various watermark attack approaches, even when adversaries have white-box access to the DSN framework. 

# 2 Embedding Deep Serial Number in DNNs

The key idea of DSN is to build a new DNN training and distribution framework so that each DNN model will function normally only when the potential user enters the unique serial number. In this section, we will introduce the three requirements and discuss the proposed framework. 

# 2.1 Requirements for Serial Number Watermarking

Serial numbers are typically assigned to users who have the right to use specific software. The software will function properly only when the user inputs the correct serial number. It is generally infeasible for an adversary to generate valid but unauthorized codes through brute-force attacks or reverse engineering of the software. In our design, an ideal serial number for DNNs is expected to meet the following four requirements: 

– Low Distortion: Embedding the serial number into DNNs should not significantly compromise the performance of the DNN model in its original tasks. 

– Reliability: The DNN performs properly only when a user enters a valid serial number. Any invalid serial numbers will result in a substantial performance decline in the original tasks. 

– Robustness: The DSN should exhibit sufficient resilience against various attack methods, including 1) commonly used deep learning techniques, such as transfer learning and model pruning, and 2) malicious attack methods, such as reverse engineering and watermark overwriting. 

# 2.2 The Proposed DSN Framework

The proposed DSN framework is depicted in Fig. 1. We formulate it as a twostep process: 1) initially training a teacher network $f _ { T }$ to maximize prediction performance, and 2) subsequently training multiple student networks $f _ { S }$ based on the knowledge distilled from the teacher [7,10]. During the distillation process, we introduce a new SN (Serial Number) embedding loss $\mathcal { L } _ { D S N }$ , which enables DSN to embed a unique serial number into the student network, in addition to transferring knowledge. The student network functions correctly only when the correct serial number is entered. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/d0a4cba400e32777f7e49f583aa770cb0aec6c2a3b8638232c2368e883b7f918.jpg)



Ruixiang Tang 2021 66Fig. 1. Training pipeline of deep serial number framework. DSN is built based on the knowledge distillation framework, where a secret training dataset and teacher model are in the developers’ hands. The two complementary losses, SN Embedding loss, and Distillation loss, embed a unique serial number into the customer model. Owners only distribute the well-trained student model (blue part) to potential customers.


Teacher-Student Framework. We propose employing a knowledge distillation framework to train multiple customized DNN models. The approach is formulated as follows. Given a vector of logits $Z _ { T }$ as the output of the last fully connected layer of the teacher model $f _ { T }$ , we can estimate the probability $P _ { T }$ by applying a softmax function to $Z _ { T }$ . We utilize the soft target obtained from the teacher model as a supervision signal to transfer knowledge from $f _ { T }$ to $f _ { S }$ . The distillation loss is formulated as follows: 

$$
\mathcal {L} _ {\text {D i s t i l l}} \left(f _ {T}, f _ {S}\right) = \mathcal {L} _ {K L} \left(P _ {T}, P _ {S}\right), \tag {1}
$$

where $\mathcal { L } _ { K L }$ represents the KL divergence loss. This training framework enables the student model to achieve comparable or even superior, performance to that of the pre-trained teacher model. Unlike conventional distillation settings, stakeholders using DSN will keep the teacher model and training data confidential, distributing only the trained student networks (blue part in Fig. 1) to the markets and customers. 

Embedding Serial Number. The process of embedding the serial number is implemented as follows. Given a student model $f _ { S }$ , the inputs $x$ , and the unique serial number $\hat { k }$ , the student model embedded with the serial number $f _ { S } ^ { K }$ can be formulated as: 

$$
f _ {S} ^ {K} = r (x) (1 - h (k)) + f _ {S} (x) h (k), \tag {2}
$$

where $k$ is the serial number entered by the user, and $h ( k )$ is the serial number recognition function that verifies the correctness of the input serial number. If the serial number $k$ is valid, i.e., $k = \hat { k }$ , $h ( k )$ outputs 1 and $f _ { S } ^ { K } ( x ) = f _ { S } ( x )$ . For an invalid serial number, $h ( k )$ outputs 0 and $f _ { S } ^ { K } = r ( x )$ , where the functionality of $r ( x )$ significantly differs from $f _ { S } ( x )$ , such as random guessing. Consequently, the performance drops substantially with incorrect serial numbers. The motivation behind the proposed DSN framework is to implicitly integrate the functionality of $r ( x )$ and $h ( k )$ into the student model. 

Let $X = \{ x _ { n } , y _ { n } \} _ { n = 1 } ^ { N }$ represent the training data, $\boldsymbol { I } _ { k = \hat { k } }$ denote the correctness of the entered SN where $I = 1$ indicates a valid SN and $I = 0$ signifies an incorrect SN. We aim for $f _ { S } ^ { K } \left( x \right)$ to accurately predict $Y$ when $I = 1$ and predict poorly with $I = 0$ . The input $\mathbf { X }$ is initially mapped to a D-dimensional feature vector $e$ using mapping $G _ { e }$ (a feature extractor). We denote the vector of parameters for all layers in the mapping as $\theta _ { e }$ , i.e., $e = G _ { e } ( x ; \theta _ { e } )$ . Subsequently, the feature vector $e$ is mapped by mapping $G _ { y }$ (predictor with SN) to the label $y$ . We denote the parameters of this mapping with $\theta _ { y }$ . Lastly, the same feature vector $e$ is mapped by mapping $G _ { d }$ (predictor without SN) to the label $y$ with parameter $\theta _ { d }$ . The overall two-branch model structure is illustrated in Fig. 1. 

During the learning stage, when $I \ = \ 1$ , our objective is to minimize the label prediction loss on $G _ { y }$ , and the parameters of both the feature extractor $G _ { e }$ and the label predictor $G _ { y }$ are optimized to minimize the empirical loss for the training samples $x$ . When $I ~ = ~ 0$ , features $e$ should be unpredictable (for the classifier $G _ { d }$ , the hidden representation $e$ belonging to a different class should be inseparable). Drawing inspiration from the work by Ganin et al. [6], we employ the Gradient Reversal Layer (GRL) to remove the label information $Y$ in the features $e$ . During forward propagation, the GRL acts as an identity transform. During backpropagation, GRL takes the gradient from the subsequent level, multiplies it by a negative value $\lambda$ , and passes it to the preceding layer. The GRL is inserted between the feature extractor $G _ { y }$ and the classifier $G _ { d }$ . The stochastic updates can be formalized as follows: when $I = 1$ , we train the student using the Distillation Loss. 

$$
\mathcal {L} _ {\text {D i s t i l l}} \left(f _ {T}, f _ {S}\right) = \mathcal {L} _ {K L} \left(P _ {T}, G _ {y} \left(G _ {e} (x)\right)\right), \tag {3}
$$

where $P _ { T }$ is the soft label of the teacher model. The stochastic update can be written as follows: 

$$
\theta_ {e} \leftarrow \theta_ {e} - \mu \left(\frac {L _ {\text {D i s t i l l}}}{\theta_ {e}}\right); \quad \theta_ {y} \leftarrow \theta_ {y} - \mu \left(\frac {L _ {\text {D i s t i l l}}}{\theta_ {y}}\right). \tag {4}
$$

When $I = 0$ , the model is optimized with SN Embedding Loss. 

$$
\mathcal {L} _ {S N E} \left(f _ {S}\right) = \mathcal {L} _ {C E} \left(G _ {d} \left(G R L \left(G _ {e} (x)\right)\right), Y\right), \tag {5}
$$

where $\mathcal { L } _ { C E }$ is the cross-entropy loss. The stochastic update can be written as follows: 

$$
\theta_ {e} \leftarrow \theta_ {e} + \mu \left(\frac {L _ {S N E}}{\theta_ {e}}\right); \quad \theta_ {d} \leftarrow \theta_ {d} - \mu \left(\frac {L _ {S N E}}{\theta_ {d}}\right). \tag {6}
$$

The proposed two-branch training pipeline enables $G _ { e }$ to supply well-trained features $e$ for classifiers $G _ { y }$ when provided with a correct serial number. When an incorrect serial number is entered, the output features $e$ for different classes become indistinguishable, resulting in poor prediction accuracy for both $G _ { d }$ and $G _ { y }$ . To distribute the student model, stakeholders will remove the GRL and $G _ { d }$ , and package the remaining network consisting of $G _ { e }$ and $G _ { y }$ for the customer. 

# 2.3 Entangled Watermark Embedding

A potential limitation of the proposed DSN framework is that its effectiveness may be compromised by pruning protection-related neurons, such as those responsible for recognizing serial numbers. To address this issue, it is necessary to entangle protection-related neurons with regular neurons. We achieve this by introducing a soft nearest neighbor loss (SNNL) [14] to measure the entanglement between representations learned by clean inputs and those learned by SN-stamped inputs. This can be expressed as: 

$$
S N N L (X, Y, T) = - \frac {1}{n} \sum_ {i \in 1.. n} \log \left( \right.\frac {\sum_ {\substack {j \in 1.. n\\j \neq i\\y _ {i} = y _ {j}}} e ^ {- \frac {| | G _ {e} \left(x _ {i}\right) - G _ {e} \left(x _ {j}\right) | | ^ {2}}{T}}}{\sum_ {\substack {k \in 1.. n\\k \neq i}} e ^ {- \frac {| | G _ {e} \left(x _ {i}\right) - G _ {e} \left(x _ {k}\right) | | ^ {2}}{T}}}\right)\right) \tag{7}
$$

where $G _ { e } ( x )$ represents the input representations. The loss calculates the ratio between (a) the average distance separating a point $x _ { i }$ from other points within the same class and (b) the average distance separating any two points. The temperature T is used to emphasize smaller or larger distances accordingly. By maximizing the SNNL loss between clean inputs and SN-stamped inputs, we ensure that the representation distributions for both types of inputs are similar. Empirically, this approach forces the model to use the same group of neurons for both SN protection and the original task, making it more difficult to prune protection-related neurons. Consequently, the final loss function for the DSN framework can be expressed as follows: 

$$
\mathcal {L} _ {D S N} = \left\{ \begin{array}{l l} \mathcal {L} _ {D i s t i l l} + \alpha \mathcal {L} _ {S N N L}, & \text {i f} I = 1 \\ \mathcal {L} _ {S N E} + \alpha \mathcal {L} _ {S N N L}, & \text {i f} I = 0, \end{array} \right. \tag {8}
$$

where $\alpha$ serves as a hyperparameter to adjust the weight of the entanglement. In our experiments, we set the value of $\alpha$ to 0.1. 

# 2.4 Serial Number Space

In this section, we discuss the serial number space Following the settings in previous work, [18], stakeholders $O$ use their private key to sign some known versifiers $V$ , e.g., $O$ ’s the company name and a timestamp, $E n c r y p t ( O _ { p r i } , v ) =$ sig, where the signature sig is a bit sequence that will be used to deterministically generate the serial number. In this paper, we focus on exploring DSN applications for computer vision tasks and consider using a $0 / 1$ bit pattern as the SN. To activate DNN, the user needs to stamp the valid SN pattern on the correct position. Let $\hat { k }$ represent the SN pattern to be embedded in the DNN. Let $x$ be an input image and $\boldsymbol { x } ^ { * } = \boldsymbol { x } \oplus \boldsymbol { \hat { k } }$ be the image stamped with SN. Note that $\hat { k }$ , $x$ and $x ^ { * }$ have the same dimension. is the normalized pixel value of x at point $x _ { i , j }$ $( i , j ) ( 0 < x _ { i , j } < 1 )$ ), and $x _ { i , j } ^ { * }$ is the pixel value of SN stamped image at the same 

point. $\hat { k } _ { i , j }$ is the pixel value of SN at point $( i , j )$ , which can be either $1 , 0$ or $^ { - 1 }$ We then have the following mapping function: 

$$
x _ {i, j} ^ {*} = \left\{ \begin{array}{l l} 1, & \text {i f} \quad \hat {k} _ {i, j} = 1 \\ 0, & \text {i f} \quad \hat {k} _ {i, j} = 0 \\ x _ {i, j}, & \text {i f} \quad \hat {k} _ {i, j} = - 1. \end{array} \right. \tag {9}
$$

The SN pattern is defined as the $0 / 1$ pattern in pixels where $\hat { k } _ { i , j } \neq - 1$ . When the SN is placed in a less important position, such as the corners of the image, the small SN pattern will not affect the original input signals. 

# 3 Experiments

We conduct experiments on the three applications to validate that our DSN model meets the three watermarking requirements, that is, low distortion, reliability, and robustness. 

# 3.1 Experimental Setups

Datasets. We conduct experiments on three datasets with different applications: digital recognition, traffic sign recognition, and face recognition. 

– Digit Recognition (MNIST) [16]: MNIST is a digit recognition dataset with 10 output classes. The digits have been normalized in size and centered in a fixed-size image with $2 8 \times 2 8$ resolution. 

– German Traffic Sign Recognition Benchmark (GTSRB) [27]: GTSRB contains colorful images of 43 traffic signs and has 39,209 training images and 12,603 testing images, respectively. 

– Pubfig [15]: Pubfig is used to validate the performance of DSN on large and complex inputs. This dataset contains 13,838 face images of 85 people. Compared to GTSRB and MNIST, images in Pubfig have much higher resolution. 

Model Architectures. For the MNIST dataset, we adopt a standard 4-layer convolutional neural network. For GTSRT, we utilize 6 convolution layers and 2 dense layer models. For the Pubfig dataset, we adopt a 16-layer VGG-Face model [24]. Note that in this work, we choose the same structure for both teacher and student models. 

Implementation Details. In all experiments, we normalize the input in the range [0, 1]. The SN pattern is a $0 / 1$ bit square pattern stamped on the right bottom corner, and we set the width of the pattern as 10% of the input image. Therefore, the area of the pattern only accounts for 1% of the original picture. SN bit patterns would up-scale proportionally when deploying DNN systems to target high-resolution images. The training process could be divided into two steps. First, we train a teacher model to maximize its performance on the specific task. Based on the teacher model, we then use the DSN framework to train multiple student networks. For raw inputs $X$ , we train the student model 

with SNE loss in Eq. 5. For the SN stamped input $X \oplus K$ , we optimize the student model with Distillation Loss in Eq. 3. For the feature extractor $G _ { e }$ , we optimize with the SNNL loss in Eq. 7. SN-stamped inputs are generated on-thefly, and the two-branch DSN framework could be optimized parallelly. We use Adam as the optimizer for all teacher models and set the batch size to 500. The learning rate starts from 0.001 and is divided by 10 when the error plateaus. We utilize Adam as the optimizer for all student models and set the batch size to 500, including 250 raw inputs and 250 SN-stamped inputs. 

<table><tr><td rowspan="2">Task</td><td>Teacher</td><td colspan="2">Student Model</td></tr><tr><td>\( \mathcal{A}_X \)</td><td>\( \mathcal{A}_{X\oplus K} \)</td><td>\( \mathcal{A}_X \)</td></tr><tr><td>MNIST</td><td>99.9</td><td>99.8</td><td>9.2</td></tr><tr><td>GTSRB</td><td>97.0</td><td>97.2</td><td>8.2</td></tr><tr><td>Pubfig</td><td>87.9</td><td>87.3</td><td>7.3</td></tr></table>


Table 1. Accuracy of the Teacher and Student Network


<table><tr><td rowspan="3">Task</td><td rowspan="2" colspan="2">Student Model</td><td colspan="4">Fine-Tuning Attack</td></tr><tr><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>AX⊕K</td><td>AX</td><td colspan="4">AX</td></tr><tr><td>MNIST</td><td>99.8</td><td>9.2</td><td>94.7</td><td>95.4</td><td>96.6</td><td>97.1</td></tr><tr><td>MNIST*</td><td>-</td><td>-</td><td>95.1</td><td>95.5</td><td>96.9</td><td>98.3</td></tr><tr><td>GTSRB</td><td>97.2</td><td>8.2</td><td>65.2</td><td>71.6</td><td>75.4</td><td>81.3</td></tr><tr><td>GTSRB*</td><td>-</td><td>-</td><td>65.3</td><td>73.2</td><td>85.3</td><td>87.8</td></tr><tr><td>Pubfig</td><td>87.3</td><td>7.3</td><td>51.3</td><td>53.5</td><td>60.2</td><td>65.3</td></tr><tr><td>Pubfig*</td><td>-</td><td>-</td><td>55.2</td><td>61.3</td><td>65.7</td><td>73.2</td></tr></table>


Table 2. DSN against Fine-tuning Attack


# 3.2 Prediction Distortion Analysis

For an ideal serial number embedding approach, the performance of student networks on the original task should not degrade significantly. Tab. 1 shows the classification accuracy for the teacher model and the student model. We observe that the student networks achieve competitive, and in some cases, better performance compared to the teacher models when the input is stamped with a valid serial number. The student model performance on MNIST and Pubfig experiences a minor drop of $0 . 1 \%$ and $0 . 6 \%$ , respectively. Surprisingly, the performance of the student model on GTSRB even surpasses that of the teacher model by 0.2%. One plausible explanation for the improvement on the GTSRB dataset is that we utilize the same architecture for both the student and teacher models, a phenomenon that has been reported and analyzed in previous work [5]. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/7ff5f548b0c6d223d280ade6a64efd731d5c6f39bdb5e8a9bcfdb30ddf2f29f8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/ac8317df3bf65ca078316492e16c2da41f596b4bb445b119f5ea5bfb78a43975.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/577a15eac187f39e1a354cb62017baa6fade4129215bac532eca67b8809711b5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/9a06cbb1672a89f1db928cbb57f45c85d6f4854a091180d55d1a389f45396abc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/2680310040c6a8ef1c4f9d2fede7fe5b27d6feb0a327fb44a7e1068e439d4f65.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/6656a3f4b834adc434ca4c893c7476e9fa99f43e00207025ecace82f11f57115.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/04dd1571a76cf56ff94608a0573587c9716b6cbac830ab51dd9e5cde0fa34595.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/d896b44e873a70db65aee378835eb92ff6ec4e719b28fe67ab723b9dada7da63.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/1b820b83447502bc863827f1253a5532c1ed1333a55bf867519560c676bbe642.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/ed95d60acd217699fac347b6aeddfb740dc3b0d0aff45a0ee3672196e5e578df.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/7935fde2736036d9a9669bf8a9a392210807cfcd01fc6e6eb13a72ec73c1abaf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/27e3c099b03d8d0859dc0d25f035fbc36b3998bb8124367f46a5b3cdba5b3eb8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/39805e6dcf58f67199da2e827854b02976e19a8382e9f62c47c3854eaa2e8d7b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/04d59f5e5628b1efac138283ad7197173e5b64bcea3042d5d32cc396c346a77b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/ef38f4c6d1366eff9e13c94060ecd839857c9640621e257ac3fa1516d76f361f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/71b452c791aadeaef2ebaa0d1232aa6b3ce133c74a03676ed3c4f9b0478a83ed.jpg)



Fig. 2. Effectiveness of correct and invalid SNs ( $\%$ )


<table><tr><td rowspan="3">Task</td><td rowspan="2" colspan="2">Student Model</td><td colspan="4">Model Pruning Attack</td></tr><tr><td>5%</td><td>10%</td><td>15%</td><td>20%</td></tr><tr><td>AX⊕K</td><td>AX</td><td colspan="4">AX⊕K / AX</td></tr><tr><td>MNIST</td><td>99.8</td><td>9.2</td><td>98.4 / 8.7</td><td>98.4 / 8.3</td><td>98.4 / 9.7</td><td>98.4 / 9.9</td></tr><tr><td>GTSRB</td><td>97.2</td><td>8.2</td><td>97.2 / 8.2</td><td>97.2 / 8.2</td><td>97.1 / 8.3</td><td>96.8 / 9.5</td></tr><tr><td>Pubfig</td><td>87.3</td><td>7.3</td><td>87.3 / 7.3</td><td>87.3 / 7.4</td><td>87.1 / 8.1</td><td>82.5 / 9.7</td></tr></table>


Table 3. DSN against Model-Puning Attack


# 3.3 Prediction Reliability Analysis

In Table 1, we also report the model performance with and without an embedded serial number (SN). The key observation is that when the inputs do not contain a valid SN, the performance of the student networks drops significantly, approaching random guessing. Without entering a valid SN (by inputting raw images in the experiments), the prediction accuracy of MNIST, GTSRT, and Pubfig substantially drops to 9.2%, 8.2%, and 7.3%, respectively. This performance is close to random guessing, which is $\textstyle { \frac { 1 } { N } }$ , where N represents the number of classes. We can conclude that the DSN framework ensures that only the valid SN can correctly activate the customer model. We further assess the effectiveness of invalid SN. To ensure that the preset SN is the only valid one, we apply other SN patterns on the inputs when training the branch $G _ { d }$ . We conduct an experiment on the MNIST dataset to evaluate the effectiveness of the wrong SN. For the $2 \times 2$ SN pattern, we evaluate the model performance with 1 correct SN and 15 invalid SNs. As shown in Fig. 2, the average accuracy $\left( \mathcal { A } _ { X \oplus K } \right)$ of the 15 incorrect SNs is only 13.5% (the highest is $2 7 . 2 \%$ ). The results indicate that only the correct SN number can activate the protected model with our proposed DSN framework. All invalid SNs will cause a significant performance drop. 

# 3.4 Attacking Robustness Analysis

In this section, we further investigate the robustness of the proposed framework. The embedded SN should be robust against various attack methods [2, 29, 33]. In this work, we group the existing attack approaches into two typical scenarios: 1) The adversaries do not know the SN. An example of this scenario is that the DNN is accidentally stolen by the adversary. In this case, the adversary’s purpose is to either remove or reverse engineer the SN pattern. 2) The adversaries know SN. In this case, the adversary could be a legal buyer who wants to illegally distribute models to other parties. To redistribute the model, the adversary expects to remove or tamper the embedded SN and thus reclaims the ownership of the tampered model. 

Adversary without Knowledge of SN For Adversaries without knowledge of SN, we consider three commonly used attack methods, including fine-tuning, transfer learning, model pruning, and reverse engineering. 

<table><tr><td rowspan="3">Task</td><td rowspan="2" colspan="2">Student</td><td colspan="4">Transfer-Learning</td></tr><tr><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>AX⊕K</td><td>AX</td><td>AX</td><td>AX</td><td>AX</td><td>AX</td></tr><tr><td>MNIST</td><td>99.8</td><td>9.2</td><td>85.2</td><td>89.5</td><td>90.6</td><td>93.2</td></tr><tr><td>MNIST*</td><td>-</td><td>-</td><td>93.6</td><td>94.5</td><td>95.6</td><td>96.9</td></tr><tr><td>GTSRB</td><td>97.2</td><td>8.2</td><td>81.7</td><td>83.2</td><td>85.3</td><td>87.9</td></tr><tr><td>GTSRB*</td><td>-</td><td>-</td><td>91.8</td><td>93.4</td><td>94.2</td><td>95.5</td></tr><tr><td>Pubfig</td><td>97.3</td><td>7.3</td><td>82.3</td><td>85.3</td><td>87.2</td><td>88.9</td></tr><tr><td>Pubfig*</td><td>-</td><td>-</td><td>92.3</td><td>94.4</td><td>96.7</td><td>97.1</td></tr></table>


Table 4. DSN against transfer-leaning


<table><tr><td rowspan="3">Task</td><td rowspan="2" colspan="2">Student</td><td colspan="4">Overwriting Attack</td></tr><tr><td>10%</td><td>20%</td><td>30%</td><td>40%</td></tr><tr><td>AX⊕K</td><td>AX</td><td colspan="4">AX</td></tr><tr><td>MNIST</td><td>99.8</td><td>9.2</td><td>93.5</td><td>94.2</td><td>95.1</td><td>96.8</td></tr><tr><td>MNIST*</td><td>-</td><td>-</td><td>95.1</td><td>95.5</td><td>96.9</td><td>98.3</td></tr><tr><td>GTSRB</td><td>97.2</td><td>8.2</td><td>64.4</td><td>72.3</td><td>74.9</td><td>79.0</td></tr><tr><td>GTSRB*</td><td>-</td><td>-</td><td>65.3</td><td>73.2</td><td>85.3</td><td>87.8</td></tr><tr><td>Pubfig</td><td>87.3</td><td>7.3</td><td>51.0</td><td>52.7</td><td>59.3</td><td>61.8</td></tr><tr><td>Pubfig*</td><td>-</td><td>-</td><td>55.2</td><td>61.3</td><td>65.7</td><td>73.2</td></tr></table>


Table 5. DSN against SN Overwriting


Fine-Tuning. In assessing the robustness of DSN against fine-tuning, we assume that the adversary only has a small segment of the model’s original training data. Otherwise, an adversary could train the model from scratch. The student model is optimized by the standard cross entropy loss with a different portion of the original training data (10%, 20%, 30%, 40%). By directly training on the raw input, the adversary expects to remove the effect of SN that the model can perform normally without inputting the valid SN. Tab. 2 reports the experimental results. We observe that fine-tuning the student model on the original dataset can remove the SN effect. However, it also causes a notable performance drop. For example, when fine-tuning using $1 0 \%$ of the original training, GTSRB performance drops from $9 7 . 2 \%$ to $6 5 . 2 \%$ . We also train the model from scratch using the same portion of the original training data, which denotes DATASET $^ *$ . We find that the performance of the fine-tuned student model is comparable to or worse than training from scratch, which implies that the cost of removing the SN through fine-tuning is nearly equivalent to training a new model from scratch. Consequently, the adversary has no incentive to steal the student model and expensively remove the DSN using a fine-tuning attack. 

Model-Pruning. The pruning attack aims to remove redundant parameters and obtain a new student model that appears different from the original model but still maintains competitive accuracy. If the removed parameters contain the SN function, verifying the embedded SN would no longer be possible. Tab. 3 reports the experimental results. In these experiments, we adopt the commonly used L1- norm global pruning strategy [9] and prune the model by eliminating the lowest 5%-20% of connections across the entire model. The results indicate that model pruning has no impact on the DSN student model in terms of LowDistortion and Reliability. The performance of the pruned model with SN does not change significantly with increasing pruning strength. The increase in accuracy without SN is less than 2% when pruning 20% of the model weight, which suggests that SN protection remains highly effective. We can conclude that DSN is robust against model pruning. 

Transfer-Learning. Different from fine-tuning settings, an adversary in transfer learning does not have the original training dataset but a small-scale private dataset. The motivation of the adversary is to use the features extracted from $G _ { e }$ to train a new model adapted to the private task. Following the common 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/7878eb512f853387875ba14b3d46a215341a45ddf436b012fad79768a44d1840.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/55502ea72a4e64287df817e6a031a3d4670ad2d4ecc28d6957c0ae2e0a3a524e.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/92f8c99a437fed686c65bdf26800961c8a357212575180e57193346677788163.jpg)



(c)



Fig. 3. Visualization of Embedding. (a): Blue/red points show the embedding without/with SN. (b): Embedding without SN. (c): Embedding with SN.


transfer-learning paradigm, we replace all fully connected layers according to the new task requirements, such as adjusting the very last original fully connected layers based on the prediction class numbers. Here, we randomly choose a half class from the original dataset and apply AdaIN style transformation [11] on them as the new private dataset. Tab. 4 reports the experimental results. We observe a similar result as in the fine-tuning attack. The computational cost of removing SN in the student model is close to learning from scratch. This is because the DSN framework guarantees that the features generated by $G _ { e }$ are indistinguishable. We show the visualization of the image embedding in Fig. 3, and we observe that images’ embedding without SN is randomly distributed while clustered with the valid SN. 

Reverse Engineer Attack. In this section, we propose a novel attack to reverse-engineer the secret SN embedded in the DSN model. The optimization objective has two goals. For a given DSN model $y = f ( x )$ , the first goal is to generate a functionally similar proxy serial number SN that enables the modelˆ to work properly. The second goal is to find a "concise" SN, which means the generated SN modifies only a limited portion of the input. We formulate this as ˆ a multi-objective optimization task by optimizing the weighted sum of the two objectives. The loss function is formulated as follows: 

$$
\min  \mathcal {L} (f (A (x, \hat {\mathrm {S N}})), y) + \lambda | \hat {\mathrm {S N}} |, f o r x \in X, \tag {10}
$$

where $A ( . )$ represents the function that applies a generated SN to the original ˆ input, and $| \mathrm { S N } |$ is used to regularize the size of the proxy serial number. $\mathcal { L }$ specifies the loss function of the model output $f ( x )$ and the ground truth label $y$ . $\lambda$ is the weight for the second objective, where a smaller $\lambda$ gives a lower weight to controlling the size of SN. The Adam optimizer is employed to solve ˆ this optimization problem. We conduct experiments on MNIST, GTSRB, and Pubfig datasets. In Tab. 6, we present the results of our reverse engineering attack. The column “ $\boldsymbol { \mathcal { A } _ { \boldsymbol { X } \oplus \hat { \boldsymbol { K } } } } "$ specifies the model performance with the reverseengineered SN. Our key observation is that the proposed framework can partially 

<table><tr><td rowspan="2">Task</td><td>Teacher Model</td><td colspan="3">Student Model</td></tr><tr><td>AX</td><td>AX ⊕ K</td><td>AX ⊕ K̂</td><td>AX</td></tr><tr><td>MNIST</td><td>99.9 (± 0.1)</td><td>99.8 (± 0.1)</td><td>48.8 (± 25.1)</td><td>9.2 (± 1.4)</td></tr><tr><td>GTSRB</td><td>97.0 (± 0.3)</td><td>97.2 (± 0.1)</td><td>22.3 (± 22.3)</td><td>8.2 (± 1.5)</td></tr><tr><td>Pubfig</td><td>87.9 (± 1.2)</td><td>87.3 (± 0.3)</td><td>24.5 (± 17.8)</td><td>7.3 (± 0.8)</td></tr></table>


Table 6. DSN against Reverse Engineering Attack


reverse engineer the functionality of the SN. For instance, the accuracy of the MNIST classifier increases from 9.2% (invalid SN) to $4 8 . 8 \%$ (reverse-engineered SN). The accuracy of the GTSRB classifier increases from 8.2% (invalid SN) to $2 2 . 3 \%$ (reverse-engineered SN). The accuracy of the Pubfig classifier increases from 7.3% (invalid SN) to $2 4 . 5 \%$ (reverse-engineered SN). However, the accuracy of the reverse-engineered SN is not stable, and the variance is significant. Sometimes the generated SN can perform very well, such as $6 8 . 9 \%$ on MNIST. Nonetheless, compared to the valid SN, the reverse-engineered SN still leads to a considerable drop in model performance. Furthermore, we only considered some straightforward SN patterns in our experiments. It will be more challenging to reverse engineer the serial number if we use a more complex and larger trigger pattern. 

Adversary with Knowledge of SN In this attack scenario, the adversary has knowledge of the DSN framework as well as the legitimate owner’s SN. We consider the overwriting attack, in which the adversary includes an additional watermark on top of the original one. 

Overwriting Attack. In the case of the overwriting attack, we assume that an adversary seeks to replace the original sensitive neuron (SN) $\hat { k }$ with a new one, denoted as $\hat { k } ^ { * }$ . To accomplish this, they train the student model with a new SN pattern using the Deep Sensitive Neuron (DSN) framework. Similar to previous attack scenarios, we consider that the adversary has access to only a limited portion of the model’s original training data. The student model is optimized using the standard cross-entropy loss with 10%, 20%, 30%, and 40% of the original training data. The experimental results are presented in Table 5. Our observations reveal that, like transfer learning attacks, overwriting negatively impacts the student model’s performance. The cost and performance of overwriting an SN are similar to those of training a model from scratch. Overwriting attacks, when compared to fine-tuning attacks, demand more training effort since they involve introducing a new SN via the DSN framework and eliminating the original SN pattern. Our empirical findings indicate that the computational cost of overwriting attacks is nearly double that of training a model from scratch. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/0540ebd05c8e630c1ed2e82231a3d95d26a36fc9e6795c1e84378b1a6cf72161.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/dc9c8515-eae4-4f76-847a-27b0127cd214/07af06445fb39a983351bc8a8c10929964fd392397784caad0f26a4045f72af8.jpg)



Fig. 4. DSN on the PDF OCR model. Left figure shows the results with the SN (the sun icon on the bottom right corner), and the right figure shows the result without SN.


# 4 A Case Study on PDF OCR Model

In this section, we showcase a prototype implementation of the DSN framework within a PDF OCR model. Following the pipeline depicted in Fig. 1, we changed the classification objective into the region proposal task and customize a PDF OCR student model to identify a unique icon in a PDF document, specifically embedding a sun icon as the serial number within the OCR region proposal module. When the document contains the icon (situated at the bottom right corner), the OCR region proposal module can accurately detect text within the provided invoice PDF. Conversely, if the icon is missing or incorrect, the customized OCR model generates a random region proposal rather than identifying the actual text. The proposed DSN framework presents a secure approach for the OCR model’s owner to ensure that only authorized parties utilize their model. By incorporating this watermark into the OCR model, owners can safeguard against unauthorized access to their intellectual property and reduce the likelihood of their model being misused. 

# 5 Related Work

In this section, we review two directions of research that are most relevant to ours, including embedding watermarks to DNNs and attacks against watermarks. 

# 5.1 Digital Watermarks for DNNs.

There are some initial attempts to verify the practicability of embedding watermarks into DNNs [12, 17, 19, 21, 25, 30, 34]. According to their embedding and verification mechanism, we group them into two categories [1, 22, 35]. 

Embedding Watermark into DNN Parameters. Uchida et.al [31] firstly proposes to embed watermarks into the parameters of DNNs by imposing an additional regularization term on the distribution of weights. By verifying the specific statistical bias in weights, the developers can claim ownership of the model. A more recent work [4] proposes a new ownership verification scheme by embedding special "passport" layers into the model architecture. Model owners keep the passport layer weights secret from unauthorized parties. For this series of work, model owners usually need white-box access for watermark verification, which is not piratical in many real-world scenarios. 

Embedding Watermark in DNN Outcomes. The second category of watermarking techniques works by embedding watermarks in the prediction results of models. A frequently used technique is the emerging backdoor attack approach [8, 29], where applying pre-designed trigger patterns on the input could precisely manipulate the outputs of DNNs, e.g., misclassifying inputs into a target label. Taking inspiration from the threat model of a backdoor attack, the model owners could inject a backdoor into the DNNs during the training process and utilize the secret trigger pattern as the watermark for remote ownership verification. For this kind of work, model owners only need black-box access (e.g., requiring prediction results remotely from APIs) for watermark verification, which is more practical in real-world scenarios. 

# 6 Acknowledgement

The authors thank the anonymous reviewers for their helpful comments. The work is in part supported by NSF grants NSF CNS-1816497, IIS-1849085 and IIS-2224843. The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing any funding agencies. 

# 7 Conclusions and Future Work

In this paper, we introduce DSN (Deep Serial Number), a new watermarking method that can prevent adversaries from deploying stolen deep neural networks, where the customer DNN function normally only if a potential user enters a valid serial number. Experiments on various applications indicate that DSN is effective in terms of preventing unauthorized applications while not sacrificing the original DNN performance. The experimental analysis further demonstrates that DSN is resistant to various attack methods. In this study, we mainly focus on computer vision tasks. In the future, we will apply our DSN framework to more applications and models, such as natural language processing and large language models [13, 28, 36]. 

# 8 Limitations and Ethical Statement

While the proposed DSN demonstrates robust defense in various attack scenarios, it remains vulnerable to several potential attack surfaces. For instance, adversaries may employ unlabeled data alongside the output from the protected model to train a local copy, referred to as a model extraction attack [23,26]. The current DSN framework cannot defend against such an attack. Furthermore, individuals may share the serial number with others and operate the model on an unregistered machine, a situation DSN cannot prevent. However, it is crucial to acknowledge that no universal protection mechanism can defend against all types of attacks, and DSN is not explicitly designed to counter extraction attacks or unauthorized use on unregistered machines. In real-world applications, defenders must employ a combination of defense methods to achieve comprehensive protection. Addressing model extraction attacks remains a complex challenge we plan to investigate in our future research. 

This manuscript has undergone a comprehensive review to ensure adherence to ethical principles and has been deemed to comply with all relevant ethical guidelines. No ethical concerns were identified with regard to the content of this paper, which is considered to be a valuable addition to the field. 

# References



1. Boenisch, F.: A survey on model watermarking neural networks. arXiv preprint arXiv:2009.12153 (2020) 





2. Chen, H., Fu, C., Zhao, J., Koushanfar, F.: Deepinspect: A black-box trojan detection and mitigation framework for deep neural networks. In: IJCAI. pp. 4658–4664 (2019) 





3. Chen, X., Wang, W., Bender, C., Ding, Y., Jia, R., Li, B., Song, D.: Refit: a unified watermark removal framework for deep learning systems with limited data. arXiv preprint arXiv:1911.07205 (2019) 





4. Fan, L., Ng, K.W., Chan, C.S.: Rethinking deep neural network ownership verification: Embedding passports to defeat ambiguity attacks. In: NIPS. pp. 4714–4723 (2019) 





5. Furlanello, T., Lipton, Z.C., Tschannen, M., Itti, L., Anandkumar, A.: Born again neural networks. arXiv preprint arXiv:1805.04770 (2018) 





6. Ganin, Y., Lempitsky, V.: Unsupervised domain adaptation by backpropagation. In: ICML. pp. 1180–1189. PMLR (2015) 





7. Gou, J., Yu, B., Maybank, S.J., Tao, D.: Knowledge distillation: A survey. arXiv preprint arXiv:2006.05525 (2020) 





8. Gu, T., Liu, K., Dolan-Gavitt, B., Garg, S.: Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access 7, 47230–47244 (2019) 





9. Han, S., Mao, H., Dally, W.J.: Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149 (2015) 





10. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015) 





11. Huang, X., Belongie, S.: Arbitrary style transfer in real-time with adaptive instance normalization. In: ICCV. pp. 1501–1510 (2017) 





12. Kapusta, K., Thouvenot, V., Bettan, O., Beguinet, H., Senet, H.: A protocol for secure verification of watermarks embedded into machine learning models. In: Proceedings of the 2021 ACM Workshop on Information Hiding and Multimedia Security. pp. 171–176 (2021) 





13. Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., Goldstein, T.: A watermark for large language models. arXiv preprint arXiv:2301.10226 (2023) 





14. Kornblith, S., Norouzi, M., Lee, H., Hinton, G.: Similarity of neural network representations revisited. In: International Conference on Machine Learning. pp. 3519– 3529. PMLR (2019) 





15. Kumar, N., Berg, A.C., Belhumeur, P.N., Nayar, S.K.: Attribute and simile classifiers for face verification. In: ICCV. pp. 365–372. IEEE (2009) 





16. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.: Gradient-based learning applied to document recognition. Proceedings of the IEEE 86(11), 2278–2324 (1998) 





17. Li, G., Li, S., Qian, Z., Zhang, X.: Encryption resistant deep neural network watermarking. In: ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 3064–3068. IEEE (2022) 





18. Li, H., Wenger, E., Zhao, B.Y., Zheng, H.: Piracy resistant watermarks for deep neural networks. arXiv preprint arXiv:1910.01226 (2019) 





19. Li, Y., Bai, Y., Jiang, Y., Yang, Y., Xia, S.T., Li, B.: Untargeted backdoor watermark: Towards harmless and stealthy dataset copyright protection. In: Advances in Neural Information Processing Systems (2022) 





20. Li, Y., Jiang, Y., Li, Z., Xia, S.T.: Backdoor learning: A survey. IEEE Transactions on Neural Networks and Learning Systems (2022) 





21. Li, Y., Zhu, M., Yang, X., Jiang, Y., Wei, T., Xia, S.T.: Black-box dataset ownership verification via backdoor watermarking. IEEE Transactions on Information Forensics and Security (2023) 





22. Lounici, S., Njeh, M., Ermis, O., Önen, M., Trabelsi, S.: Yes we can: Watermarking machine learning models beyond classification. In: 2021 IEEE 34th Computer Security Foundations Symposium (CSF). pp. 1–14. IEEE (2021) 





23. Oliynyk, D., Mayer, R., Rauber, A.: I know what you trained last summer: A survey on stealing machine learning models and defences. arXiv preprint arXiv:2206.08451 (2022) 





24. Parkhi, O.M., Vedaldi, A., Zisserman, A.: Deep face recognition. British Machine Vision Association (2015) 





25. Regazzoni, F., Palmieri, P., Smailbegovic, F., Cammarota, R., Polian, I.: Protecting artificial intelligence ips: a survey of watermarking and fingerprinting for machine learning. CAAI Transactions on Intelligence Technology 6(2), 180–191 (2021) 





26. Sanyal, S., Addepalli, S., Babu, R.V.: Towards data-free model stealing in a hard label setting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15284–15293 (2022) 





27. STALLKAMP, J., SCHLIPSING, M., SALMEN, J., IGEL, C.: Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural networks 32, 323–332 (2012) 





28. Tang, R., Chuang, Y.N., Hu, X.: The science of detecting llm-generated texts. arXiv preprint arXiv:2303.07205 (2023) 





29. Tang, R., Du, M., Liu, N., Yang, F., Hu, X.: An embarrassingly simple approach for trojan attack in deep neural networks. In: KDD. pp. 218–228 (2020) 





30. Tang, R., Feng, Q., Liu, N., Yang, F., Hu, X.: Did you train on my dataset? towards public dataset protection with clean-label backdoor watermarking. arXiv preprint arXiv:2303.11470 (2023) 





31. Uchida, Y., Nagai, Y., Sakazawa, S., Satoh, S.: Embedding watermarks into deep neural networks. In: MM. pp. 269–277 (2017) 





32. Vincent, J.: Meta’s powerful ai language model has leaked online: what happens now? The Verge (2023), https://www.theverge.com/2023/3/8/23629362/ meta-ai-language-model-llama-leak-online-misuse 





33. Wang, B., Yao, Y., Shan, S., Li, H., Viswanath, B., Zheng, H., Zhao, B.Y.: Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. In: 2019 IEEE Symposium on Security and Privacy (SP). pp. 707–723. IEEE (2019) 





34. Wang, L., Song, Y., Xia, D.: Deep neural network watermarking based on a reversible image hiding network. Pattern Analysis and Applications pp. 1–14 (2023) 





35. Wang, R., Li, H., Mu, L., Ren, J., Guo, S., Liu, L., Fang, L., Chen, J., Wang, L.: Rethinking the vulnerability of dnn watermarking: Are watermarks robust against naturalness-aware perturbations? In: Proceedings of the 30th ACM International Conference on Multimedia. pp. 1808–1818 (2022) 





36. Yang, J., Jin, H., Tang, R., Han, X., Feng, Q., Jiang, H., Yin, B., Hu, X.: Harnessing the power of llms in practice: A survey on chatgpt and beyond. arXiv preprint arXiv:2304.13712 (2023) 





37. Yang, Z., Dang, H., Chang, E.C.: Effectiveness of distillation attack and countermeasure on neural network watermarking. arXiv preprint arXiv:1906.06046 (2019) 

