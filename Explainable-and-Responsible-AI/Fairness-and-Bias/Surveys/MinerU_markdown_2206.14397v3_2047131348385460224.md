# Fair Machine Learning in Healthcare: A Survey

Qizhang Feng, Mengnan Du, Na Zou, and Xia Hu 

Abstract—The digitization of healthcare data coupled with advances in computational capabilities has propelled the adoption of machine learning (ML) in healthcare. However, these methods can perpetuate or even exacerbate existing disparities, leading to fairness concerns such as the unequal distribution of resources and diagnostic inaccuracies among different demographic groups. Addressing these fairness problem is paramount to prevent further entrenchment of social injustices. In this survey, we analyze the intersection of fairness in machine learning and healthcare disparities. We adopt a framework based on the principles of distributive justice to categorize fairness concerns into two distinct classes: equal allocation and equal performance. We provide a critical review of the associated fairness metrics from a machine learning standpoint and examine biases and mitigation strategies across the stages of the ML lifecycle, discussing the relationship between biases and their countermeasures. The paper concludes with a discussion on the pressing challenges that remain unaddressed in ensuring fairness in healthcare ML, and proposes several new research directions that hold promise for developing ethical and equitable ML applications in healthcare. 

Impact Statement—Along with the rapid growth in the use of machine learning in healthcare in recent years, there has been a growing concern about the fairness problems that come along with it. This survey article helps break down the barriers between fair machine learning and healthcare, and aims to: 1) improve healthcare practitioners’ understanding of the bias of machine learning in healthcare from a computational perspective; 2) assist machine learning researchers in establishing a clear picture on how to develop fair algorithms in various healthcare scenarios from a healthcare perspective; and 3) increase public trust in machine learning algorithms and promote the use of machine learning methods in real-world healthcare settings. 

Index Terms—Artificial Intelligence, Fairness, Healthcare, Machine Learning. 

# I. INTRODUCTION

W ITH the advent of sophisticated machine learning(ML) applications in healthcare, from medical image (ML) applications in healthcare, from medical image analysis to electronic health records processing, we stand on the cusp of a transformative era in medicine [84], [116], [117], [137], [92]. Despite these advancements, there remains a significant yet understudied challenge: ensuring fairness in algorithmic decisions, particularly as they relate to the equitable treatment of diverse patient populations [135]. 

Fairness in healthcare ML refers to the equitable distribution of benefits and burdens across all demographic groups, with 

Manuscript submitted June 17, 2022; date of current version Nov 7, 2023. This work is in part supported by NSF grants IIS-1939716 and IIS-1900990. 

Qizhang Feng is with the Department of Computer Science & Engineering, Texas A&M University, TX 77843, US (e-mail: qf31@tamu.edu). 

Mengnan Du is with the Department of Data Science, New Jersey Institute of Technology, NJ 07102, US (e-mail: mengnan.du@njit.edu). 

Na Zou is with the Department of Engineering Technology & Industrial Distribution, Texas A&M University, TX 77843, US (e-mail: nzou1@tamu.edu). Xia Hu is with the Department of Computer Science, Rice University, TX 77251, US (e-mail: xia.hu@rice.edu). 

This paragraph will include the Associate Editor who handled your paper. 

particular attention to historically marginalized communities. It encompasses a range of issues, from the allocation of healthcare resources to diagnostic accuracy across different patient demographics. Notable instances include genetic tests where AI models disproportionately misrepresent risks for minority groups [102], and diagnostic discrepancies exacerbated by incomplete medical records among Black and Hispanic patients [119]. The Covid-19 pandemic has further highlighted these disparities, intensifying the urgency to address them [61]. 

Recognizing the potential of ML to either perpetuate or mitigate existing disparities, this survey seeks to fill the critical gap in the literature by providing a comprehensive analysis of fairness-oriented ML strategies in healthcare. We acknowledge the socio-technical nature of fairness challenges in healthcare ML, which encompasses algorithmic aspects and extends to societal, ethical, and regulatory dimensions. This survey synthesizes insights from previous works, including the categorization of fairness problems [112] and solutions, and charts a path forward for equitable ML applications in healthcare. The commitment to fair and inclusive AI development is echoed by governmental bodies, such as the National Institutes of Health, through initiatives like AIM-AHEAD and Bridge2AI [20]. 

Distinguishing from related reviews. While the domains of fairness in machine learning and health disparities are well-researched, their intersection remains nascent. Several surveys [112], [33], [57], [136], [31], [115] have attempted to address fairness problems in machine learning methods for healthcare. Yet, a holistic approach that captures both the ethical and technical detail is missing in the literature. The key point of fair machine learning in healthcare contain both ethical consideration and also technical details. However, We have observed that existing works tend to focus on one aspect while neglecting the other. For instance, some works [112], [31] discuss fairness from an ethical standpoint but lack a detailed connection with technical mitigation methods and metrics for fairness in machine learning. Conversely, another line of work [136] provides a technical perspective by categorizing fairness metrics and mitigation methods but does not establish a strong link with the ethical aspects of healthcare fairness. Additionally, some studies [33] focus narrowly on data shifts and federated learning, while work [115] limits its scope to fairness in artificial intelligence for medical imaging. Our survey seeks to establish a comprehensive link between the ethical and technical dimensions of fair machine learning in healthcare. Our survey is motivated by the need to bridge this evident gap, providing a comprehensive perspective that ties together the ethical considerations and technical details of fairness in healthcare machine learning. Specifically, our contributions are summarized as follows: 

1) Connect the ethical and technical aspects of fairness in 

healthcare by adopting the concept of distributive justice from works [112], [82]. Classify healthcare fairness problems into equal allocation and equal performance, and provide a comprehensive summary of fairness measurement methods within the fair machine learning domain, categorizing them accordingly. 

2) Provide a comprehensive overview of biases at various stages of the machine learning model development. Conduct a structured analysis of fairness mitigation methods, surpassing previous surveys in exhaustiveness. Highlight the critical gap in current mitigation methods, focusing on the necessity to discuss and analyze their applicability to scenarios of equal allocation and equal performance. 

3) Discuss challenges and opportunities in creating a fair and reliable machine learning ecosystem for healthcare, with an emphasis on the unique aspects of healthcare applications. 

The structure of this article is listed as follows. The definition of fairness problems in healthcare is given in Section II. On the basis of this definition, measurements of fairness are given in Section III. Then the biases at various stages of model development are introduced in Section IV. Similarly, according to the same categorization, methods for mitigating fairness problems are discussed in Section V. Finally, we highlight the challenges and opportunities for a fair and trustworthy machine learning healthcare ecosystem based on uniqueness in healthcare applications in Section VI. 

# II. FAIRNESS PROBLEMS IN ML FOR HEALTHCARE

Because of the digitization of medical data collection, we can now collect large amounts of medical data and develop machine learning algorithms for a variety of medical tasks. First, machine learning models have been used in pioneering applications on medical images (e.g., NIH Chest-Xray14, CheXpert, MIMIC-CXR and Chest-Xray8 [132], [73], [78], [119]). For example, a large-scale study built a deep neural network on the NIH Chest-XRay14 dataset and the CheXpert dataset to diagnose various chest diseases [86]. Second, machine learning models have also been applied to the structured electronic health record (EHR), which contains information on demographics, diagnoses, laboratory tests, medications, etc. For instance, a gradient boosting model was used to predict cardiovascular disease risk based on the Stanford Translational Research Integrated Database Environment (STRIDE 8) dataset [101]. Third, advancements in natural language processing (NLP) have greatly enhanced our ability to process unstructured electronic health record (EHR) data, such as clinical narratives, medical examinations, clinical laboratory reports, surgical notes, and discharge summaries. These NLP methods facilitate a range of critical tasks including medical concept extraction, disease inference, and clinical decision support [35], [142], [100]. Particularly, large language models (LLMs) such as GPT-2 and GPT-3 have demonstrated their utility in medical question-answering tasks, including those in pain management domains [111], [22]. The recent surge in conversational language models, exemplified by ChatGPT, underscores their transformative potential in healthcare. Recent 

study evaluates the feasibility of ChatGPT across multiple clinical and research scenarios, showcasing the model’s substantial impact and the breadth of its applications in the healthcare field [25]. The recent surge in conversational language models, such as ChatGPT, marks a significant shift in healthcare technology. Notably, Microsoft’s Azure Health Bot [1] and NHS-LLM [2] represent pioneering applications of LLMs in healthcare. These models are designed to assist users in assessing healthcare needs, particularly when they are uncertain about the severity of their condition. However, this raises critical fairness considerations, as the algorithms’ decision-making processes must account for diverse patient populations and their unique healthcare requirements. This is crucial to ensure equitable access and outcomes across different demographics, thus avoiding exacerbation of existing disparities in healthcare [25]. 

# A. Distributive Justice in Machine learning for Healthcare

Although the use of machine learning techniques in healthcare has been shown to correct clinical inadequacies and increase operational efficiency by reducing resource waste [127], various fairness problems have also been raised. The problem of fairness in machine learning methods is reflected in the discrimination of different groups [61]. For example, state-ofthe-art convolutional neural network (CNN) classifiers were found to differ in the true positive rate across protected attributes (e.g., patient gender, age, race, and insurance type) on 14 diagnostic tasks in 3 well-known public chest x-ray datasets [119]. 

Discrimination can be understood as a distributive problem [85]. In studies developed related to machine learning in healthcare, fairness problems often refer to the unequal distribution of resources such as medical care, clinical services and health facilities [58]. Distributive justice is concerned with the distribution of resources among members of a society, and the underlying idea of distributive justice theories are distribution principles and metrics of justice [82]. The distribution principles specify how resources should be distributed [85]. The justice metric specifies the type of resources to be allocated [82]. In the context of the fairness problem of machine learning methods in healthcare, resources often refer to the medical services allocated by the system, or the error rate of the predictions it gives. Fairness problems can be grouped into two categories based on differences in the resources allocated: equal allocation and equal performance [112]. 

1) Equal Allocation: Machine learning models are often used to allocate medical supplies such as vaccines, medicines and organ transplants. Accordingly, fairness problems of machine learning methods in healthcare occur if the model determines that the allocation of resources is not equal between groups. For example, a recently published work focused on building models to help determine which patients with chronic kidney disease should undergo kidney transplantation. The study found that the model was biased towards black patients and tended to classify black patients as having more severe kidney disease [7]. Another study examined how to effectively combat influenza in the early stages of an influenza outbreak 

with minimal vaccine dosing in the early stages of vaccine production when production is limited [53]. They found that the optimal solution to the model was likely to produce a controversial distribution strategy of not distributing any vaccine to certain subgroups. 

2) Equal Performance: In some medical situations, machine learning models are used for medical tasks such as disease diagnosis, mortality prediction and multi-organ segmentation, etc. Consequently, it would be unfair if the performance and results of machine learning models are not equally accurate in terms of metrics such as accuracy for patients in different demographic groups. A recent study discovered that, despite having similar accuracy to board-certified dermatologists, machine learning algorithms used to classify images of benign and malignant moles are less accurate in the diagnostic task of melanoma on dark skin [5]. Another study analyzes the sex/racial bias in AI-based cine CMR segmentation using a large-scale database [110]. It is shown that state-of-theart deep learning models for automatic segmentation of the ventricle and myocardium based on cine short-axis CMR had statistically significant differences in errors between races. 

Different healthcare settings require different kinds of distributive justice. The various distributive justice options make it extremely difficult for ML models to satisfy all conditions [46], [36]. Thus, suitable metrics are crucial for evaluating the fairness of a machine learning model. 

# III. MEASUREMENT OF FAIRNESS

In the previous section, we introduce the fairness problems in machine learning for healthcare and categorize them into equal allocation problems and equal performance problems. Selecting an appropriate fairness metric is critical to measuring the fairness problem in various healthcare scenarios. In this section, we first introduce two principles of fairness following distributive justice and then summarize the common metrics of fairness that apply to them. 

To measure the fairness of a given decision algorithm $f ( \cdot )$ , we define $\textbf { x } \in \mathbb { R } ^ { d _ { x } }$ as the nonsensitive features vector and $\mathbf { z } \in \mathbb { R } ^ { d _ { z } }$ as the sensitive features vector. In most cases, only one sensitive feature is considered, so we use $z$ when $d _ { z } =$ 1. The prediction of the model $f ( \cdot )$ with input x as $\hat { y } =$ $f ( \mathbf { x } )$ , and $y$ is the corresponding ground truth label. In this survey, we mainly focus on the binary classification problem, while many works go beyond it into multi-class classification task, regression task, segmentation task with their own unique metric. Other symbols and definitions can be found in Table I. 

# A. Equal Allocation

Equal allocation is suitable in the healthcare setting when resources should be distributed proportionally to patients in protected groups. Equal allocation is also applicable when the label is historically biased [113]. For example, if historically African American women have been sent for such procedures at unduly low rates, then a ‘correct’ prediction based on historical data would underestimate the status of these women. From a computational point of view, it is desirable that the decisions made by the model differ as little as possible 

between the different demographic groups. In the following, we introduce some fairness metrics that follow the principle of equal allocation. 

• Demographic Parity (DP) is satisfied if a machine learning algorithm gives equal decision rates for different demographic subgroups a and b: 

$$
\mathbb {P} (\hat {y} = 1 \mid z = a) = \mathbb {P} (\hat {y} = 1 \mid z = b), \tag {1}
$$

DP can be extended for multi-class classification application such as image recognition, text categorization, etc [44]: 

$$
\sum_ {k = 1} ^ {K} \left| \mathbb {P} (\hat {y} = k | z = a) - \mathbb {P} (\hat {y} = k | z = b) \right| = 0, \quad \forall k \in [ K ], \tag {2}
$$

where $[ K ] = \{ 1 , \dots , K \}$ indicates $K$ number of classes. An alternative definition can constitute the summation with a maximum. DP can also be applied to the regression model rather than to the classification model [37]: 

$$
\sup  _ {t \in \mathbb {R}} | \mathbb {P} (\hat {y} \leq t | z = a) - \mathbb {P} (\hat {y} \leq t | z = b) | = 0. \tag {3}
$$

• General Demographic Parity (GDP) [75] extends the demographic parity on the continuous sensitive attribute: 

$$
\Delta G D P = \mathbb {E} _ {z} \left[ | \mathbb {E} [ \hat {y} \mid z ] - \mathbb {E} [ \hat {y} ] | \right], \tag {4}
$$

where $\mathbb { E } [ \hat { y } ~ \mid ~ s ]$ is the local average prediction of the model conditioned on the sensitive attribute, and $\mathbb { E } [ \hat { y } ]$ is the global prediction average. GDP degenerates into weighted demographic parity for the categorical sensitive attribute. 

• Fairness through Unawareness (FTU) [83] defines an algorithm as FTU fair as long as sensitive attributes are not used by the decision-making algorithm $f ( \cdot )$ : 

$$
\mathbb {P} (\hat {y} \mid \mathbf {x}, z) = \mathbb {P} (\hat {y} \mid \mathbf {x}) \tag {5}
$$

FTU will fail even if no sensitive attributes are present in the data, if a combination of non-sensitive features can act as a proxy for them. For example, an individual’s postal code might be used as a proxy for their income, race, or ethnicity [55]. 

• Fairness through Awareness [51] emphasizes that a fair algorithm should make similar decisions for two individuals $x$ and $x ^ { \prime }$ with similar non-sensitive attributes: 

$$
D (f (x), f \left(x ^ {\prime}\right)) \leq d (x, x ^ {\prime}) \tag {6}
$$

Note that the algorithm should satisfy the $( D , d )$ -Lipschitz property. 

• Counterfactual Fairness [83] is derived from causal theory. The intuition of counterfactual fairness is that a fair algorithm should provide the same decision for a real-world individual and its corresponding one in the counterfactual world: 

$$
\mathbb {P} \left[ \hat {y} _ {\{z \leftarrow a \}} = c | x, z = a \right] = \mathbb {P} \left[ \hat {y} _ {\{z \leftarrow b \}} = c | x, z = a \right] \tag {7}
$$

Achieving consensus on causal graphs is challenging due to the complexity of causal structure discovery, 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/8dc480bb-d4ad-4238-a27c-c9b1c4605b3d/27d6af067a192c9769438e1bd81c14fb2be5e26ea6a643931373a384678ae620.jpg)



Fig. 1. Bias at the different stages in machine learning systems: Red and blue represent two demographic groups. (a) The biases that exist at the data collection stage include minority bias, missing-data bias and label bias. Minority bias occurs when the sample size of the demographic groups are unbalanced. Missing data bias occurs when data may be missing in a non-random way. Label bias occurs when the quality of labels varies between different demographic groups. (b) Algorithm bias exists in model development stage, leads to systematical unfair results for certain demographic group. (c) The biases that exist at the data collection stage include interaction bias and training-serving skew bias. Training-serving skew bias occurs when the distribution of data in the deployment stage differs from the distribution of data in the training phase. Interaction bias occurs patients and healthcare professionals interact with machine learning models. Please refer to section IV for further details.



TABLE I



MAIN SYMBOLS AND DEFINITIONS.


<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td>f(·)</td><td>A machine learning model that maps attributes to predictions.</td></tr><tr><td>x ∈ Rdx</td><td>The non-sensitive attributes with a dimension of dx.</td></tr><tr><td>z ∈ Rdz</td><td>The sensitive attribute.</td></tr><tr><td>z</td><td>The sensitive attribute when dz = 1.</td></tr><tr><td>ŷ ∈ {0,1}</td><td>A binary prediction that indicates negative and positive outcomes for 0 and 1, respectively.</td></tr><tr><td>y ∈ {0,1}</td><td>A binary ground truth that indicates negative and positive outcomes for 0 and 1, respectively.</td></tr><tr><td>ŷ{z←a}</td><td>A prediction in the counterfactual world if z = a.</td></tr><tr><td>D</td><td>The training dataset.</td></tr><tr><td>d(·,·)</td><td>Distance of two individuals in the attribute space.</td></tr><tr><td>D(·,·)</td><td>Distance of two individuals in the prediction space.</td></tr><tr><td>θ</td><td>Parameters θ of the backbone network.</td></tr><tr><td>φ</td><td>Parameters φ of the adversarial network.</td></tr><tr><td>Aφ(·)</td><td>Adversarial network with parameters φ</td></tr><tr><td>L(D;θ)</td><td>The downstream task loss.</td></tr><tr><td>Ladv(D;φ)</td><td>The adversarial loss.</td></tr></table>

particularly without existing knowledge of causality. This complexity can lead to the incorrect assumption of causal structures from statistical model outputs, resulting in varying interpretations and difficulty in standardizing causal graphs [121]. 

# B. Equal Performance

Equal performance means that a model is guaranteed to be equally accurate for patients in protected and non-protected groups. The concept of accuracy can include equal sensitivity (also called equal opportunity [140]), equalized odds, and equal positive predictive value [36], or broader metrics such as AUC, etc. Equal performance metric is appropriate in the context where the accuracy of the machine learning model is crucial. For instance, the machine learning system can be 

introduced to build a monitoring system that is used to alert rapid response teams when hospitalized patients are at high risk of deterioration [54]. If the predictive model imposes a high false positive rate on the protected group, patients in the protected group may lose the opportunity to be identified, which can have serious consequences. However, forcing a model’s predictions to have one of the performance characteristics of equality [67] may have unintended consequences. For example, the model may achieve equal odds by sacrificing the accuracy of the unprotected group, which undermines the benefit principle [133]. In the following, we introduce some fairness measurements that follow the principle of equal performance. 

• Equal Opportunity is preferred when people care more about true positive rates. We say that a classifier satisfies 

equal opportunity if the true positive rate is the same across the groups [140]: 

$$
\mathbb {P} \{\hat {y} = 1 \mid y = 1, z = a \} = \mathbb {P} \{\hat {y} = 1 \mid y = 1, z = b \}. \tag {8}
$$

It can also be referred to as positive predictive value parity. Similarly, there is negative predictive value parity: 

$$
\mathbb {P} \{\hat {y} = 1 \mid y = 0, z = a \} = \mathbb {P} \{\hat {y} = 1 \mid y = 0, z = b \} \tag {9}
$$

Predictive value parity is also called sufficiency. 

• Equalized Odds requires that the decision rates across demographic subgroups be the same when their outcome is the same [27]: 

$$
\mathbb {P} \{\hat {y} = 1 \mid z = a, y = 0 \} = \mathbb {P} \{\hat {y} = 1 \mid z = b, y = 0 \},
$$

$$
\mathbb {P} \left\{\hat {y} = 1 \mid z = a, y = 1 \right\} = \mathbb {P} \left\{\hat {y} = 1 \mid z = b, y = 1 \right\} \tag {10}
$$

Equalized Odds requires the algorithm to have equal true positive rates and equal false positive rates at the same time. 

• Treatment Equality requires that the ratio of false negatives and false positives be the same for subgroups [14]: 

$$
\frac {\mathbb {P} (\hat {y} = 1 \mid y = 0 , z = a))}{\mathbb {P} (\hat {y} = 0 \mid y = 1 , z = a))} = \frac {\mathbb {P} (\hat {y} = 1 \mid y = 0 , z = b))}{\mathbb {P} (\hat {y} = 0 \mid y = 1 , z = b))}. \tag {11}
$$

# IV. SOURCES OF FAIRNESS PROBLEMS

In this section, we summarize the causes of fairness problems in healthcare machine learning and use the term ‘bias’ to denote them [98]. The process of building a machine learningbased healthcare system can be divided into three stages. First, the agency collects relevant clinical data for model development. Then, developers select and train a suitable model for the intended task, based on the data and the type of task. Finally, the institution involved can license the model for implementation in real clinical practice. We present the various complex biases that exist in healthcare based on the three different stages (see the overview in Figure 1). 

# A. Bias in Data Collection

Data collection is the first stage at which bias may be introduced. A machine learning model is trained to fit the distribution of the training data. When there is bias in the data, the model may perpetuates the bias (as shown in Figure 1.(a)). In the following paragraphs, we review several common types of data bias in clinical practice. 

1) Minority Bias: Minority bias occurs when the sample size of a demographic group is smaller than that of other groups. The development of machine learning algorithms in healthcare is currently highly dependent on public biobank databases [13], [23]. However, due to the uneven development of medical standards, most of the data collection is done in Europe. This has led to the study of human knowledge of the disease using biobank repositories that mainly represent individuals of European ancestry. For example, the vast majority of cases in the Cancer Genome Atlas (TCGA) are made up 

of whites, representing approximately $8 2 . 0 \%$ of the cases. In contrast, a very small proportion of the cases are from black, Asian, and other ethnic minorities [59]. In fact, demographic data such as ethnicity are crucial to determining the mutational profile and mechanisms of cancer. As a result, genetic risk models perform worse in ethnic minority populations. 

2) Missing-data Bias: Missing data bias occurs when data may be missing in a non-random way. Machine learning algorithms may cause harm to people with missing data in the dataset. For example, research has found that vulnerable people of low socioeconomic status are likely to be seen in a piecemeal fashion or cannot be seen. If patients are identified based on a certain number of ICD codes, records of the same number of visits to several different healthcare systems for these patients may be missing. Another example is that, despite numerous initiatives, sexual orientation and gender identity have been largely absent from electronic health records to date. Machine learning-based clinical decision support systems can misinterpret the lack of access to care as a lower burden of disease and therefore produce inaccurate predictions for these groups [33]. 

3) Label Bias: Label bias may also be present in data labels, and the quality of the labels can contribute to bias [112]. For example, people with low socioeconomic status may be more likely to be seen in teaching clinics, where documentation or clinical reasoning may be less accurate or systematically different from the care provided to patients with high socioeconomic status. Algorithms based on these data may reflect practitioner bias and misclassify patients based on these factors. The choice of inappropriate labels can also introduce bias. For example, some models use specific phrases that appear in clinical records as proxy labels that indicate the presence of cardiovascular disease. However, because women have different symptoms of acute coronary syndromes, proxy phrases have different meanings for men and women. As a result, women can receive delayed care, which causes discrimination against women. 

# B. Bias in Model Development

Bias in the model development phase can lead to machine learning models perpetuating or even amplifying existing biases in the data. This can stem from various sources, including inappropriate intrinsic hypotheses, the structure of the model, and biased loss estimators, all of which can potentially contribute to fairness problems [24], [134], [30]. A predominant concern in this context is algorithmic bias, where the source of bias is traceable back to the model itself, systematically leading to unfair results for certain groups as depicted in Figure 1.(b). 

In the discourse of algorithmic fairness, both shortcut learning and confounding effects epitomize pathways through which machine learning models may inadvertently perpetuate biases. Shortcut learning occurs when models exploit easy but unreliable correlations to make predictions [21], [12], often bypassing more substantive but complex relationships. For instance, a study shows that models can capture and amplify the association between labels and sensitive attributes, 


TABLE II MITIGATION METHODS CATEGORIZATION.


<table><tr><td colspan="2"></td><td>Reference</td><td>Task</td><td>Dataset</td><td>Data Type</td></tr><tr><td rowspan="9">Data Collection</td><td rowspan="7">Data Redistribution</td><td rowspan="2">Diversified Collection</td><td>[88]</td><td>Type II Diabetes Risk Prediction</td><td>eMERGE[97]</td></tr><tr><td>[56]</td><td>Skin Lesion Classification</td><td>Skin ISIC 2018[38]</td></tr><tr><td rowspan="2">Data Reweighting</td><td>[138]</td><td>Skin Lesion Classification</td><td>Skin ISIC 2017[65]</td></tr><tr><td>[130]</td><td>AD Classification</td><td>ANDI[103]</td></tr><tr><td>Data Resampling</td><td>[29]</td><td>Diabetes Classification</td><td>The Pima Indian Diabetes Dataset[122]</td></tr><tr><td rowspan="2">Synthetic Data</td><td>[114]</td><td>Skin Lesion Classification</td><td>HAM10000[129]</td></tr><tr><td>[15]</td><td>Mortality Prediction</td><td>MIMIC-III[77]</td></tr><tr><td rowspan="2">Data Purification</td><td rowspan="2"></td><td>[99]</td><td>Mortality Prediction</td><td>MIMIC-IV[76]</td></tr><tr><td>[100]</td><td>Health Condition Classification</td><td>n2c2[81], MIMIC-III[77]</td></tr><tr><td rowspan="7">Model Development</td><td rowspan="6">Model Desensitization</td><td rowspan="4">Adversarial Learning</td><td>[39]</td><td>Radiology Findings Identification</td><td>Private</td></tr><tr><td>[40]</td><td>ASCVD Classification</td><td>Stanford Medicine Research Data Repository[93]</td></tr><tr><td>[18]</td><td>In Hospital Mortality Prediction, Patient Membership Prediction</td><td>MIMIC-III[77]</td></tr><tr><td>[143]</td><td>HIV Diagnosis, Morphological Sex Identification, Bone Age Determination</td><td>HIV Dataset[104],NCANDA dataset[125],Bone-aging Dataset[66]</td></tr><tr><td>Disentanglement</td><td>[18]</td><td>Appointment No-show Prediction</td><td>Private</td></tr><tr><td>Contrastive Learning</td><td>[64]</td><td>Chest X-ray Classification</td><td>NIH-ChestXRay8[132]</td></tr><tr><td>Model Constraint</td><td></td><td>[106]</td><td>Inpatient Mortality Prediction, Length of Stay Prediction</td><td>Stanford Medicine Research Data Repository[93]</td></tr><tr><td rowspan="3">Model Deployment</td><td>Decision Explanation</td><td></td><td>[99]</td><td>In Hospital Mortality Prediction</td><td>MIMIC-IV[76]</td></tr><tr><td>Model Adjustment</td><td></td><td>[74]</td><td>Congestive Heart Failure Prediction</td><td>MIMIC-CXR[78], CheXpert[73]</td></tr><tr><td>Outcome Adjustment</td><td></td><td>[105]</td><td>Ten-year Atherosclerotic Cardiovascular Disease (ASCVD) Risk Prediction</td><td>Optum CDM[3]</td></tr></table>

even in balanced datasets [131]. The learned model may amplify the association between label and gender, mimicking an imbalanced dataset. This is akin to a model using confounding variables that correlate with both the input features and the output labels, thus rendering the predictions unfair [143], [43], [62]. The recent literature underscores the similarity between these phenomena: both are manifestations of models’ proclivity to capitalize on spurious correlations rather than causally relevant patterns. Such practices not only compromise the equity of the models but also their robustness and reliability. Confounder-aware approaches and mitigation strategies, as delineated in seminal works, are therefore critical in ensuring that machine learning contributes to the fair and just application of AI in healthcare, and does not inadvertently exacerbate existing disparities. Typically, machine learning models aim to maximize overall predictive performance on the training data. This focus may lead to optimizing for individuals that occur more frequently, while neglecting underrepresented groups due to sampling bias. Consequently, a model may exhibit superior overall performance but fail to generalize well for underrepresented groups [30]. For instance, in the field of radiology, convolutional neural networks (CNNs) have been found to exhibit inconsistencies in diagnosis, particularly for underserved groups such as Hispanic patients and Medicaid recipients in the United States, leading to a higher rate of underdiagnosis or misdiagnosis compared to White patients [120]. Furthermore, studies suggest that different machine learning algorithms can exhibit varying degrees of bias when applied to the same dataset [139]. The study assessed Logistic Regression, Random Forest, and XGBoost for their performance and fairness in healthcare tasks like predicting hospital stays and diagnosing diseases. It highlighted significant variations in how these algorithms handled sensitive data like race and gender across identical datasets. 

# C. Bias in Model Deployment

A trained machine learning model can be applied to clinical practice when it has passed regulatory authorization. Bias is likely to occur at this stage, and there may contain two types of bias (as shown in Figure 1(c)). 

1) Training–serving Skew Bias: The training service skew bias is due to the fact that the data distribution encountered by the model in the deployment environment is different from the data distribution at the time of training. This phenomenon is known as the distributional shift [33], [123]. During model training, a strong assumption is that the training and test datasets are drawn independently and exactly from the same distribution (i.i.d.). This can lead to fairness problems when the model is deployed, even if it satisfies the notion of fairness in the training dataset. The phenomenon of distributional shift can occur with racially skewed public biobank datasets, which has a differential impact on ethnic subpopulations. For example, the first AI model to surpass clinical rank in predicting lymph node metastasis was trained and evaluated on the CAMELYON16/17 dataset, which is unique to the Netherlands [13], [72]. In addition to changes in ethnicity in the population, changes in medical equipment, such as image capture and biometrics, can also lead to bias. For example, in radiology, there may be differences in radiation dose that affect the signal-to-noise ratio of the images obtained. In pathology, there is also a great deal of heterogeneity in tissue preparation, staining protocols, and specific scanner camera parameters, which has been shown to affect model performance in cancer diagnostic tasks [33], [26]. Data sets may also change in response to technological developments or changes in human behavior. A typical example includes the migration of ICD-8 to ICD-9 [70]. Another example is the discontinuation of the Epic sepsis model (ESM) due to changes in patient demographics as a result of COVID-19 [33]. 

Most of the work has focused on short-term learning of 

fairness classifiers, and there has been few research on the analysis of fairness metrics under temporal or spatial dataset transfer. One work [118] uses a causal framing help diagnose failures of fairness transfer. 

2) Interaction Bias: This type of bias arises from the interaction of the model with its users. On the one hand, protected groups may distrust a model’s predictions in light of a history of exploitation and unethical behavior, believing that the model is biased against them. This is also referred as informed mistrust bias [61]. On the other hand, clinicians can also place too much trust in machine learning models and inappropriately act on inaccurate predictions, which can be called automation bias [112]. 

# V. MITIGATION OF FAIRNESS PROBLEMS

A variety of approaches have been developed to address fairness concerns in machine learning applications within the healthcare domain. These methods can be categorized based on the stage of the machine learning life cycle at which they are applied. We delineate these approaches across three key stages: data collection, model development, and model deployment. Furthermore, we meticulously align the motivations behind each mitigation method with the sources of bias identified in the previous section, providing a cohesive overview of how these strategies correspond to specific biases encountered in the machine learning pipeline. Table II presents a taxonomy of mitigation methods utilized in the domain of fair machine learning for healthcare, detailing the specific tasks, datasets, and data types to which they are applied. 

# A. Mitigating Fairness Problems in Data Collection

Data bias can be transferred and embedded in machine learning models. Therefore, we can mitigate fairness problems during the data collection phase. These methods are divided into two groups, data redistribution methods and data purification methods. 

1) Data Redistribution: Data distribution discrepancies, as discussed in Section IV-A, often lead to fairness problems in machine learning models. For instance, minority bias arises from the imbalanced data of different demographic groups, while missing data bias emerges due to the uneven distribution of unseen data. Several data redistribution techniques aim to rectify these imbalances, including diversified collection, data reweighting, data resampling, and data synthesis. 

a) Diversified Collection: While the direct collection of more diverse data is a straightforward solution, practical challenges like patient privacy and data collection costs often hinder such efforts. Federated learning offers a solution by enabling model training across multiple decentralized datasets without directly sharing the data [33], [88]. An instance of this is Swarm Learning (SL) which, when evaluated on the Skin ISIC 2018 dataset, exhibited enhanced fairness compared to centralized training [56]. However, federated learning does not guarantee balanced data. 

b) Data Reweighting: By assigning importance weights to training data, reweighting adjusts for data distribution imbalances. Applications of reweighting are seen in skin lesion classification and Alzheimer’s disease diagnosis [138], [130]. A notable drawback is that models trained with weighted samples might lack robustness, leading to estimator variance. 

c) Data Resampling: Resampling rectifies underrepresentation by adjusting the sub-samples of the original dataset. Techniques like SMOTE combine oversampling of minority groups with undersampling of majority ones, proving beneficial in tasks like heart failure survival prediction [29]. However, such methods may reduce the diversity of data characteristics. 

d) Synthetic Data: Synthetic data, often generated using algorithms like GANs, can enhance data distribution [126], [114]. By imposing fairness constraints during the generation process, biases in synthetic data can be controlled. 

Synthetic data alleviates data privacy and cost concerns [15], consistent with HIPAA’s stipulations [52]. It generates deidentified datasets that preserve statistical properties without revealing personal health information (PHI), thus supporting HIPAA’s objective to protect patient privacy. Federated learning enhances this by allowing institutions to collaboratively train models while each entity maintains control over its PHI, a process in harmony with HIPAA’s privacy and security rules. 

2) Data Purification: Data purification approaches aim to mitigate fairness problems by adjusting data features or labels, often by addressing biases related to sensitive attributes. 

a) Removing Sensitive Attributes: A common intuition in data purification is to remove sensitive attributes from the dataset, a method known as fairness through unawareness. However, this approach has limitations, as protected attributes can still be inferred from other features or their combinations, which act as proxy variables correlating with protected group membership [99]. 

b) Mitigation in Language Models: In the realm of Natural Language Processing (NLP), data purification has been explored for clinical notes. One study [100] quantified the “genderedness” of n-grams in clinical notes using cosine similarity between word vectors generated by BERT-base and Clinical BERT word embeddings [45], [10]. The most biased n-grams were then identified using rank perturbation dispersion (RTD) and subsequently removed from the clinical notes [47]. 

c) Addressing Label Bias: Label bias, which occurs when labels in the dataset are biased, represents another challenge. Data massaging tackles this by changing the labels of some objects in the dataset [79]. This method’s application in healthcare remains an area yet to be explored. 

# B. Mitigating Fairness Problems in Model Development

As discussed in Section IV-B, algorithmic bias during the model development stage can result in machine learning models that inherit and potentially amplify biases, leading to fairness problems. Two key drivers of this bias are identified: first, shortcut learning, where models rely on sensitive information for predictions; and second, optimization processes that fail to generalize for underrepresented groups. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/8dc480bb-d4ad-4238-a27c-c9b1c4605b3d/04b17c9f53017fa7bdc4846fce5cf83978bfc73ea3bde4e13d6b7eb299ea4bac.jpg)



Fig. 2. An illustration of methods mitigating the fairness problem in data collection stage: (a) Data redistribution methods adjust the distribution of the data. The diversified collection method collects data from other hospitals. The reweighting method assigns the weights to minority data. The resampling method seeks to create fair training samples in the sampling strategy. The synthetic method generates fake data. (b) Data purification methods remove sensitive information directly from the data. For example, removing sensitive attributes from tabular data or removing gender-specific pronouns from textual data.


To address these issues, we introduce two categories of approaches to mitigate fairness problems during the model development stage: model desensitization and model constraint. Model desensitization involves techniques that reduce a model’s reliance on sensitive attributes, thereby preventing it from making biased predictions based on those attributes. On the other hand, model constraint methods impose restrictions on the model training process to ensure fair treatment of all groups, especially those underrepresented in the training data. 

1) Model Desensitization: Model desensitization focuses on preventing models from retaining or utilizing sensitive attribute information from the data. Simply removing sensitive attributes from the data features is not a failproof solution, as machine learning models have shown the capability to differentiate sensitive information even in their absence [89], [80]. To effectively mitigate fairness problems, model desensitization approaches such as adversarial learning, representation disentanglement, and contrastive learning aim to eliminate the models’ ability to discriminate based on sensitive information. 

Adversarial learning is a widely used method to debias a model. Specifically, the goal of adversarial learning is to allow the model to complete downstream tasks while not predicting sensitive attributes. Adversarial learning is first introduced in Generative Adversarial Networks (GANs) [63] and then applied to fair machine learning [96]. Adversarial learning generally contains two branches: one is for downstream tasks, while the other is to remove sensitive attribute information: 

$$
\min  _ {\theta} \max  _ {\phi} L (\mathcal {D}; \theta) + L _ {a d v} (\mathcal {D}; \phi), \tag {12}
$$

where $\mathcal { D }$ is the training dataset. $\theta$ is the parameter for the downstream task and $\phi$ is the parameter for adversarial classification. $L$ is the normal object function and $\boldsymbol { L _ { a d v } }$ is the adversarial object function that indicates the error in predicting 

sensitive attributes. Adversarial learning is applied to debias a model for the diagnosis of chest X-ray and mammograms [39]. The authors use CNN with two branches, where one predicts the classification target and the other predicts the sensitive attributes. The training has two steps. The first step minimizes the loss for both branches. In the second step, a flipped sign gradient of adversarial branch is backpropagated, with the aim of suppressing learning of protected variables. Similar strategy is used to reduce the confounding effect from sensitive attribute [143]. In addition to the use cases for medical image data, adversarial learning has also been used to build fair machine learning models that can handle EHR data and textual data [107], [142]. 

Some other model desensitization approaches have been proposed in the context of general fairness problems instead of healthcare. The disentanglement method assumes that the entangled information from the input space could be disentangled in the latent embedding space. To make downstream tasks fair, the disentanglement method separates and removes sensitive information from the latent embedding space. Existing work has explored the use of the Variational Autoencoder (VAE) to achieve group and subgroup fairness with respect to multiple sensitive attributes [40], [18]. The contrastive learning method projects the input data into the latent space and encourages data points with various sensitive attributes to be close in the latent space and data points with the same sensitive attributes to be scattered. Some work has explored the use of contrastive learning methods to debias the pre-trained text encoder [34], image encoder [64], or to remove the effect of gender information on self-supervised embedding [128]. 

2) Model Constraint: Addressing another potential driver of algorithmic bias—namely, the failure of the optimization goal to generalize to underrepresented groups—model constraint 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/8dc480bb-d4ad-4238-a27c-c9b1c4605b3d/08e5c2077e6f489a17c4f10c6639d65c8f7f96a474b839e7f715dbf6635c52e9.jpg)



Fig. 3. An illustration of methods mitigating the fairness problem in model development stage: (a) Model desensitization removes the ability of the model to discriminate between sensitive attribute information. Adversarial learning disables the model of predicting sensitive attributes. Disentanglement method separates and removes the sensitive attribute information from latent embedding. Contrastive learning enforces the samples with various sensitive attributes to be close in latent space. (b) Model constraint methods add additional constraints or regularization term.


methods take a direct approach. Unlike model desensitization methods that implicitly debias the model, model constraint methods mitigate fairness problems by explicitly incorporating constraints into the optimization goal. 

This often involves adding fairness-specific optimization objectives. For instance, these objectives might directly improve fairness metrics [6] or include regularization terms to enforce non-discrimination principles or counterfactual fairness [106]. One notable approach involves developing an augmented counterfactual fairness criterion to reduce biases in Electronic Health Record (EHR) data. This method requires the machine learning model to make consistent predictions for a patient and a counterfactual version of the patient after altering the sensitive attribute. The optimization objective function comprises three components: prediction losses for factual and counterfactual samples, and an additional regularization term designed to meet the proposed fairness criteria [60]. 

Despite their direct approach to addressing fairness, model constraint methods are not without drawbacks. It has been observed that stringent optimization constraints can sometimes reduce predictive performance. Moreover, the impact of regularization strength on fairness metrics can vary, presenting challenges in balancing performance and fairness. 

# C. Mitigating Fairness Problems in Model Deployment

Deploying machine learning models in clinical settings often surfaces biases not apparent during training or testing. Constructing entirely unbiased models from the outset 

is challenging and resource-intensive. Thus, post-deployment mitigation strategies are essential for addressing biases as they emerge. This section explores three key methods: decision explanation, model adjustment, and outcome adjustment, each addressing specific biases such as interaction bias and trainingserving skew bias. 

1) Decision Explanation: Fairness in deployed machine learning systems is not solely a technical challenge but a socio-technical one, where human interaction with the model is pivotal. Interaction bias, as delineated in Section IV-C2, contributes to unfair outcomes during deployment. To mitigate such biases, the application of explainable artificial intelligence (XAI) is crucial, enabling users to understand and appropriately trust the model’s decisions. 

XAI can demystify model predictions, which is critical when balancing the trust in an algorithm’s decisions against the risk of perpetuating unfairness. It is particularly important in healthcare, where decisions have profound implications. For instance, studies have shown that demographic features can disproportionately influence algorithmic decisions, potentially leading to differential treatment across patient groups [99]. XAI techniques have revealed such biases by highlighting the varying importance of sensitive attributes across different demographics. 

Conversely, mistrust in fair models can also undermine their utility, prompting patients to eschew treatments or withhold information [48], [112]. Addressing this, research indicates that clear explanations of model decisions can foster trust both 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/8dc480bb-d4ad-4238-a27c-c9b1c4605b3d/2e52b040162095d53df63eca58c8da2c2ede907eb39c251f374281acca15a151.jpg)



Fig. 4. An illustration of methods mitigating the fairness problem in model deployment stage: (a) The decision explanation method offers the explanation to the outcome via XAI tool. (b) The model adjustment method fine-tunes the last few layers of the deployed mode. (c) The outcome adjustment method adjusts the original outcome to meet fairness requirement.


in the system and among medical professionals [4], [71]. 

Ultimately, integrating fairness-oriented knowledge into XAI methods not only clarifies model decisions but also guides the refinement of models to ensure equitable outcomes [50]. The synergistic relationship between fairness and explainability in machine learning models is an emergent field of research that warrants further exploration, as will be discussed in Section VI-D. 

2) Model Adjustment: Training-serving skew bias leads to fairness problem since it violates the assumption that data in the deployment phase are i.i.d. with the data in the training phase. The model adjustment method, such as transfer learning, seeks to solve this problem by fine-tuning part of the model. 

Since naively retraining the entire model can be expensive and impossible, transfer learning can provide a simple and effective way to mitigate the problem [74]. This work proposes solving the shortcut problem where the model relies on simple and shallow features (e.g. the sensitive attribute) to make the decision. Specifically, the training pipeline contains two stages. The authors first train the model on a biased dataset. Then the model is tuned on a new unbiased dataset with only the last few layers being fine-tuned. The results show that the proposed approach improves the generalization performance in older people. 

3) Outcome Adjustment: Outcome adjustment strategies are employed to enhance fairness for protected groups by altering the model’s outputs or decision boundaries [105]. 

Calibration, for instance, aims to align the proportion of positive predictions with the actual rate of positive outcomes across various subgroups [42]. Fairness in this context demands that such alignment is maintained across both protected and non-protected subgroups alike [27]. Nevertheless, the challenge arises when calibration efforts confront the incompatibility between different fairness standards. Notably, attempts to calibrate across multiple protected groups often find themselves at odds with criteria like equalized odds or disparate impact [109]. This conundrum necessitates a nuanced approach to calibration, where the trade-offs between 

competing fairness dimensions are carefully balanced. 

Thresholding takes a different tack by redefining decision boundaries. It can be particularly effective in situations where the model’s default decision threshold does not accommodate the protected group adequately. By employing variable thresholds based on sensitive attributes, a model can be tuned to fulfill fairness metrics such as equal odds or equal opportunity [68]. For example, applying a lower threshold for a minority group could increase their representation in positive predictions, aligning with the goal of equal opportunity. 

Discussion of the applicability of the mitigation method. In this review, we have previously highlighted in Section II the need for distinct forms of distributive justice in various healthcare settings, specifically equal allocation and equal performance. And it is important to note that simultaneously achieving these fairness constraints can be challenging [98]. 

Only few of existing literature on mitigation methods study their applicability to different measures of fairness metric. For example, the model constraint approach [6] offers flexibility in satisfying either equal allocation or equal performance, as the fairness constraint can be incorporated as an optimization objective. Recent studies have also demonstrated the effectiveness of certain model desensitization methods in ensuring either equal allocation or equal performance through appropriate optimization. For instance, some work [96] proposed the use of adversarial objects to achieve demographic parity and equalized of odds. However, the previous two work are in the field of general fair machine learning. Few work on fair machine learning for healthcare discuss the applicability of appropriate fairness metrics, which is an obstacle to the deployment of mitigation methods in real healthcare scenarios and may even exacerbate fairness problems. As a result, we advocate for additional research efforts to conduct detailed experiments and discussions on the suitability of different fairness metrics for different mitigation methods in order to ensure their effective implementation in healthcare settings. 

# VI. RESEARCH CHALLENGES

Despite current progress, there are numerous research challenges that must be addressed before machine learning methods can be used in clinical practice. 

# A. Uncertainty and Fairness in Healthcare

Machine learning and probabilistic methods have become ubiquitous across various domains, with their application in medical data being particularly critical due to the inherent uncertainty from noise in the data. Capturing and analyzing the uncertainty in data and models is paramount, more so in high-stakes environments such as clinical settings. In such contexts, physicians might leverage the quantified uncertainty to prioritize manual review of cases that the model deems highly uncertain. The advent of new deep learning techniques has seen a significant rise in addressing such uncertainties [9]. Despite this, the interplay between fairness and uncertainty has not been explored thoroughly in research. 

Uncertainty can play a pivotal role in highlighting fairness problems within machine learning applications in healthcare [94], [95]. Addressing epistemic uncertainty, which arises from incomplete knowledge, often involves integrating more data into the model. On the other hand, aleatoric uncertainty, which is inherent and irreducible, demands distinct strategies. The measurement and communication of uncertainties are crucial for identifying potential unfairness in model predictions [16]. Incorporating model uncertainty into fairness metrics can provide a more comprehensive view of model performance across different groups, ensuring that disparities in prediction confidence do not go unnoticed [8]. An understanding of aleatoric uncertainty can lead to models that are inherently fairer, offering improved outcomes for underrepresented groups in the data [124]. Furthermore, active learning techniques, which focus on the selection of diverse and representative data during model training, have been proposed as a means to preemptively mitigate bias [19]. There is a clear need for further investigation into how uncertainty impacts the fairness of machine learning models, a step that is crucial for the responsible deployment of AI in sensitive sectors. 

# B. Long-term Fairness in Healthcare

Another distributive justice called equal outcome or equal benefit is not mentioned in Section II-A. It refers to the assurance that protected groups have the same benefit from the deployment of machine learning models. The gap between equal allocation and equal benefit occurs when a fair decision cannot guarantee fair benefit to patients in the future. Most current research has focused on fairness problems in machine learning in static classification scenarios and has not examined how these decisions will affect the future [69]. It is often assumed that unfairness can be improved better after imposing fairness constraints on machine learning models. However, this is not the case in healthcare settings in practice. Even in a onestep feedback model, ordinary fairness standards generally do not promote improvement over time and may cause harm [91]. The key difficulty in alleviating the long-term fairness problem 

is to simulate the long-term dynamics and predict the future benefit [41]. 

Another research challenge is that the healthcare system is not an isolated system. When machine learning algorithms are embedded in clinical systems, the diagnostic decisions they make are collected and combined into new clinical data. These data then have an impact on the performance of future machine learning algorithms. This is also called a feedback loop. When bias appears in the feedback loop, it can exacerbate the bias or create new biases and further compromise the benefit of certain demographic groups. A similar feedback loop has been discussed in the context of the recommender system [141]. To the best of our knowledge, no research has been conducted on the long-term fairness problem in the context of the healthcare domain. We encourage more work on the longterm fairness of machine learning algorithms in healthcare, in particular on equal benefit and feedback loop fairness in clinical applications. 

# C. Fairness of Multi-modality Model for Healthcare

A research question is described as multimodal when it includes multiple data types. The human experience of the world is multimodal. Multi-modal machine learning aims to build models that can process and correlate information from multiple modalities, thus enabling advances in artificial intelligence in understanding the world around us. One of the key driving forces of the intelligent medical system is the multimodal method. The combination of different modalities of healthcare data, each providing information about a patient’s treatment from a specific perspective, overlays and complements each other to further improve the accuracy of diagnosis and treatment. For example, the visual quest answering task [11] combines computer vision and natural language processing, and the model can answer relevant questions based on medical images and clinical notes [90]. However, multimodal models face more serious bias and fairness problems than uni-modal models, despite improvements in performance [17]. Only a few works have focused on multimodality fairness problems in healthcare systems [32]. The forms in which bias exists vary across modality data, as do the methods used to mitigate it. As previous work has focused on the fairness problem in uni-modal data, we encourage the discovery and mitigation of bias in healthcare of multimodal data. 

# D. Ethical Machine Learning in Healthcare

The ethical landscape of machine learning within healthcare encompasses pivotal concepts such as fairness, interpretability, privacy, robustness, and security. These facets are deeply intertwined, with their relationships characterized by both synergy and tension. Fairness in healthcare AI seeks to ensure equitable treatment and outcomes across diverse patient groups. Interpretability contributes to this goal by demystifying model predictions, thereby fostering trust and enabling the identification of potential biases—critical in a clinical setting [49], [72], [99]. However, the pursuit of fairness may inadvertently conflict with privacy, particularly for 

underprivileged groups who may suffer disproportionate privacy losses [28]. Conversely, the alliance between robustness and fairness is more harmonious in healthcare AI. Robust fair training aims to inoculate models against perturbations that could skew decision-making, thus safeguarding equitable outcomes [87]. This is paramount in clinical environments where decisions must remain stable despite data variability and adversarial conditions. Furthermore, the convergence of differential privacy and adversarial robustness underscores a promising avenue where privacy-preserving techniques also fortify models against malicious attacks, a duality of particular relevance to safeguarding sensitive health data [108]. Yet, the interplay of interpretability, fairness, robustness, and privacy in healthcare AI is nascent. Research often probes these dimensions in isolation, seldom navigating their intersections. Given their mutual reinforcement and constraints, an integrated approach is imperative. Advancing multi-faceted ethical frameworks that concurrently address these dimensions will be instrumental in realizing the full potential of AI in healthcare—delivering models that are not only technically proficient but also ethically sound and clinically viable. 

# VII. CONCLUSIONS

In this survey, we have synthesized the existing literature on the intersection of machine learning and fairness within healthcare. Drawing from the foundational work in distributive justice, we have applied the classification of fairness problems in healthcare-focused machine learning methods, as identified by existing research, into two principal categories: equal allocation and equal performance. This has allowed us to map the metrics commonly used in fair machine learning to these categories specifically in the healthcare context. We have delineated biases according to the three distinct stages of the machine learning lifecycle: data collection, model development, and model deployment. For each stage, we have discussed targeted mitigation methods and examined their interconnections with the sources of bias they aim to address. Our survey reveals a gap in the critical evaluation of the effectiveness of these mitigation methods when applied to healthcare-specific fairness metrics. We underscore the pressing nature of fairness concerns in healthcare machine learning applications and propose future research directions that promise to address these challenges. 

# VIII. ACKNOWLEDGEMENT

We extend our sincere thanks for the support from the National Institutes of Health (NIH) grant 1OT2OD032581- 02-211 and the National Science Foundation (NSF) grants IIS 1900990, 1939716, and 2239257, which have significantly contributed to this survey paper. 

# REFERENCES



[1] Azure health bot — microsoft azure. https://azure.microsoft.com/en-us/ products/bot-services/health-bot. (Accessed on 01/02/2024). 





[2] A large language model for healthcare nhs-llm and opengpt. https://aiforhealthcare.substack.com/p/ a-large-language-model-for-healthcare. (Accessed on 01/02/2024). 





[3] Optum - health services innovation company. https://www.optum.com/. (Accessed on 11/07/2023). 





[4] A. Adadi and M. Berrada. Peeking inside the black-box: A survey on explainable artificial intelligence (xai). IEEE Access, 6:52138–52160, 2018. 





[5] A. S. Adamson and A. Smith. Machine learning and health care disparities in dermatology. JAMA dermatology, 154(11):1247–1248, 2018. 





[6] A. Agarwal, A. Beygelzimer, M. Dud´ık, J. Langford, and H. M. Wallach. A reductions approach to fair classification. ArXiv, abs/1803.02453, 2018. 





[7] S. Ahmed, C. T. Nutt, N. D. Eneanya, P. P. Reese, K. Sivashanker, M. Morse, T. Sequist, and M. L. Mendu. Examining the potential impact of race multiplier utilization in estimated glomerular filtration rate calculation on african-american care outcomes. Journal of general internal medicine, 36(2):464–471, 2021. 





[8] J. Ali, P. Lahoti, and K. P. Gummadi. Accounting for model uncertainty in algorithmic discrimination. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, pages 336–345, 2021. 





[9] R. Alizadehsani, M. Roshanzamir, S. Hussain, A. Khosravi, A. Koohestani, M. H. Zangooei, M. Abdar, A. Beykikhoshk, A. Shoeibi, A. Zare, M. Panahiazar, S. Nahavandi, D. Srinivasan, A. F. Atiya, and U. R. Acharya. Handling of uncertainty in medical data using machine learning and probability theory techniques: a review of 30 years (1991–2020). Annals of Operations Research, pages 1 – 42, 2021. 





[10] E. Alsentzer, J. R. Murphy, W. Boag, W.-H. Weng, D. Jin, T. Naumann, and M. B. A. McDermott. Publicly available clinical bert embeddings. ArXiv, abs/1904.03323, 2019. 





[11] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015. 





[12] I. Banerjee, K. Bhattacharjee, J. L. Burns, H. Trivedi, S. Purkayastha, L. Seyyed-Kalantari, B. N. Patel, R. Shiradkar, and J. Gichoya. “shortcuts” causing bias in radiology artificial intelligence: causes, evaluation and mitigation. Journal of the American College of Radiology, 2023. 





[13] B. E. Bejnordi, M. Veta, P. J. van Diest, B. van Ginneken, N. Karssemeijer, G. J. S. Litjens, J. A. van der Laak, M. Hermsen, Q. F. Manson, M. C. A. Balkenhol, O. G. F. Geessink, N. Stathonikos, M. C. van Dijk, P. Bult, F. Beca, A. H. Beck, D. Wang, A. Khosla, R. Gargeya, H. Irshad, A. Zhong, Q. Dou, Q. Li, H. Chen, H. Lin, P.-A. Heng, C. Hass, E. Bruni, Q. K.-S. Wong, U. Halici, M. U. ¨ Oner, R. Cetin-Atalay, M. Berseth, V. Khvatkov, A. Vylegzhanin, ¨ O. Z. Kraus, M. Shaban, N. M. Rajpoot, R. Awan, K. Sirinukunwattana, T. Qaiser, Y.-W. Tsang, D. Tellez, J. Annuscheit, P. Hufnagl, M. Valkonen, K. Kartasalo, L. Latonen, P. Ruusuvuori, K. Liimatainen, S. Albarqouni, B. Mungal, A. A. George, S. Demirci, N. Navab, S. Watanabe, S. Seno, Y. Takenaka, H. Matsuda, H. A. Phoulady, V. A. Kovalev, A. Kalinovsky, V. Liauchuk, G. Bueno, M. del Milagro Fernandez-Carrobles, I. Serrano, O. Deniz, D. Racoceanu, and ´ R. Venancio. Diagnostic assessment of deep learning algorithms for ˆ detection of lymph node metastases in women with breast cancer. JAMA, 318:2199–2210, 2017. 





[14] R. A. Berk, H. Heidari, S. Jabbari, M. Kearns, and A. Roth. Fairness in criminal justice risk assessments: The state of the art. Sociological Methods & Research, 50:3 – 44, 2018. 





[15] K. Bhanot, M. Qi, J. S. Erickson, I. Guyon, and K. P. Bennett. The problem of fairness in synthetic healthcare data. Entropy, 23, 2021. 





[16] U. Bhatt, J. Antoran, Y. Zhang, Q. V. Liao, P. Sattigeri, R. Fogliato, ´ G. Melanc¸on, R. Krishnan, J. Stanley, O. Tickoo, et al. Uncertainty as a form of transparency: Measuring, communicating, and using uncertainty. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, pages 401–413, 2021. 





[17] B. M. Booth, L. Hickman, S. K. Subburaj, L. Tay, S. E. Woo, and S. K. D’Mello. Bias and fairness in multimodal machine learning: A case study of automated video interviews. Proceedings of the 2021 International Conference on Multimodal Interaction, 2021. 





[18] S. Boughorbel, F. Jarray, and A. Kadri. Fairness in tabnet model by disentangled representation for the prediction of hospital no-show. arXiv preprint arXiv:2103.04048, 2021. 





[19] F. Branchaud-Charron, P. Atighehchian, P. Rodr´ıguez, G. Abuhamad, and A. Lacoste. Can active learning preemptively mitigate fairness issues? arXiv preprint arXiv:2104.06879, 2021. 





[20] J. Brogan. The next era of biomedical research: Prioritizing health equity in the age of digital medicine. Voices in Bioethics, 7, 2021. 





[21] A. Brown, N. Tomasev, J. Freyberg, Y. Liu, A. Karthikesalingam, 





and J. Schrouff. Detecting shortcut learning for fair medical ai using shortcut testing. Nature Communications, 14(1):4314, 2023. 





[22] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. J. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are fewshot learners. ArXiv, abs/2005.14165, 2020. 





[23] G. Campanella, M. G. Hanna, L. Geneslaw, A. P. Miraflor, V. W. K. Silva, K. J. Busam, E. Brogi, V. E. Reuter, D. S. Klimstra, and T. J. Fuchs. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nature Medicine, pages 1–9, 2019. 





[24] J. G. Carbonell, R. S. Michalski, and T. M. Mitchell. An overview of machine learning. Machine learning, pages 3–23, 1983. 





[25] M. Cascella, J. Montomoli, V. Bellini, and E. Bignami. Evaluating the feasibility of chatgpt in healthcare: an analysis of multiple clinical and research scenarios. Journal of Medical Systems, 47(1):33, 2023. 





[26] D. C. Castro, I. Walker, and B. Glocker. Causality matters in medical imaging. Nature Communications, 11(1):3673, 2020. 





[27] S. Caton and C. Haas. Fairness in machine learning: A survey. ArXiv, abs/2010.04053, 2020. 





[28] H. Chang and R. Shokri. On the privacy risks of algorithmic fairness. In 2021 IEEE European Symposium on Security and Privacy (EuroS&P), pages 292–303. IEEE, 2021. 





[29] N. Chawla, K. Bowyer, L. O. Hall, and W. P. Kegelmeyer. Smote: Synthetic minority over-sampling technique. J. Artif. Intell. Res., 16:321–357, 2002. 





[30] I. Chen, F. D. Johansson, and D. Sontag. Why is my classifier discriminatory? Advances in neural information processing systems, 31, 2018. 





[31] I. Y. Chen, E. Pierson, S. Rose, S. Joshi, K. Ferryman, and M. Ghassemi. Ethical machine learning in healthcare. Annual review of biomedical data science, 4:123–144, 2021. 





[32] J. Chen, I. Berlot-Attwell, S. Hossain, X. Wang, and F. Rudzicz. Exploring text specific and blackbox fairness algorithms in multimodal clinical nlp. ArXiv, abs/2011.09625, 2020. 





[33] R. J. Chen, T. Y. Chen, J. Lipkova, J. J. Wang, D. F. K. Williamson, ´ M. Y. Lu, S. Sahai, and F. Mahmood. Algorithm fairness in ai for medicine and healthcare. ArXiv, abs/2110.00603, 2021. 





[34] P. Cheng, W. Hao, S. Yuan, S. Si, and L. Carin. Fairfil: Contrastive neural debiasing method for pretrained text encoders, 2021. 





[35] Y. Choi, C. Y.-I. Chiu, and D. Sontag. Learning low-dimensional representations of medical concepts. AMIA Summits on Translational Science Proceedings, 2016:41, 2016. 





[36] A. Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. Big data, 5(2):153–163, 2017. 





[37] E. Chzhen, C. Denis, M. Hebiri, L. Oneto, and M. Pontil. Fair regression with wasserstein barycenters. ArXiv, abs/2006.07286, 2020. 





[38] N. C. Codella, D. Gutman, M. E. Celebi, B. Helba, M. A. Marchetti, S. W. Dusza, A. Kalloo, K. Liopyris, N. Mishra, H. Kittler, et al. Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic). In 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018), pages 168–172. IEEE, 2018. 





[39] R. Correa, J. J. Jeong, B. Patel, H. Trivedi, J. W. Gichoya, and I. Banerjee. Two-step adversarial debiasing with partial learning - medical image case-studies. ArXiv, abs/2111.08711, 2021. 





[40] E. Creager, D. Madras, J.-H. Jacobsen, M. Weis, K. Swersky, T. Pitassi, and R. Zemel. Flexibly fair representation learning by disentanglement. In International conference on machine learning, pages 1436–1445. PMLR, 2019. 





[41] A. D’Amour, H. Srinivasan, J. Atwood, P. Baljekar, D. Sculley, and Y. Halpern. Fairness is not static: deeper understanding of long term fairness via simulation studies. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, pages 525–534, 2020. 





[42] A. P. Dawid. The well-calibrated bayesian. Journal of the American Statistical Association, 77:605–610, 1982. 





[43] J. Deng, J. Yang, L. Hou, J. Wu, Y. He, M. Zhao, B. Ni, D. Wei, H. Pfister, C. Zhou, et al. Genopathomic profiling identifies signatures for immunotherapy response of lung adenocarcinoma via confounderaware representation learning. Iscience, 25(11), 2022. 





[44] C. Denis, R. Elie, M. Hebiri, and F. Hu. Fairness guarantee in multiclass classification. 2021. 





[45] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. 





[46] W. Dieterich, C. Mendoza, and T. Brennan. Compas risk scales: Demonstrating accuracy equity and predictive parity. Northpointe Inc, 7(4), 2016. 





[47] P. S. Dodds, J. R. Minot, M. V. Arnold, T. Alshaabi, J. L. Adams, D. R. Dewhurst, T. J. Gray, M. R. Frank, A. J. Reagan, and C. M. Danforth. Allotaxonometry and rank-turbulence divergence: a universal instrument for comparing complex systems. arXiv preprint arXiv:2002.09770, 2020. 





[48] J. Dodge, Q. V. Liao, Y. Zhang, R. K. E. Bellamy, and C. Dugan. Explaining models: an empirical study of how explanations impact fairness judgment. Proceedings of the 24th International Conference on Intelligent User Interfaces, 2019. 





[49] M. Du, N. Liu, and X. Hu. Techniques for interpretable machine learning. Communications of the ACM, 63(1):68–77, 2019. 





[50] M. Du, N. Liu, F. Yang, and X. Hu. Learning credible deep neural networks with rationale regularization. 2019 IEEE International Conference on Data Mining (ICDM), pages 150–159, 2019. 





[51] C. Dwork, M. Hardt, T. Pitassi, O. Reingold, and R. S. Zemel. Fairness through awareness. ArXiv, abs/1104.3913, 2012. 





[52] P. F. Edemekong, P. Annamaraju, and M. J. Haydel. Health insurance portability and accountability act. 2018. 





[53] S. Enayati and O. Y. Ozaltın. Optimal influenza vaccine distribution ¨ with equity. European Journal of Operational Research, 283(2):714– 725, 2020. 





[54] G. J. Escobar, B. J. Turk, A. I. Ragins, J. Ha, B. Hoberman, S. M. Levine, M. A. Ballesca, V. X. Liu, and P. Kipnis. Piloting electronic medical record-based early detection of inpatient deterioration in community hospitals. Journal of hospital medicine, 11 Suppl 1:S18–S24, 2016. 





[55] A. Fabris, A. Esuli, A. Moreo, and F. Sebastiani. Measuring fairness under unawareness of sensitive attributes: A quantification-based approach. Journal of Artificial Intelligence Research, 76:1117–1180, 2023. 





[56] D. Fan, Y. Wu, and X. Li. On the fairness of swarm learning in skin lesion classification. ArXiv, abs/2109.12176, 2021. 





[57] R. R. Fletcher, A. Nakeshimana, and O. Olubeko. Addressing fairness, bias, and appropriate use of artificial intelligence and machine learning in global health, 2021. 





[58] S. A. Friedler, C. Scheidegger, and S. Venkatasubramanian. On the (im) possibility of fairness. arXiv preprint arXiv:1609.07236, 2016. 





[59] J. Gao, B. A. Aksoy, U. Dogrusoz, G. Dresdner, B. E. Gross, S. O. Sumer, Y. Sun, A. S. Jacobsen, R. Sinha, E. Larsson, E. G. Cerami, C. Sander, and N. D. Schultz. Integrative analysis of complex cancer genomics and clinical profiles using the cbioportal. Science Signaling, 6:pl1 – pl1, 2013. 





[60] S. Garg, V. Perot, N. Limtiaco, A. Taly, E. H. Chi, and A. Beutel. Counterfactual fairness in text classification through robustness, 2019. 





[61] B. Giovanola and S. Tiribelli. Beyond bias and discrimination: redefining the ai ethics principle of fairness in healthcare machinelearning algorithms. AI & society, pages 1–15, 2022. 





[62] B. Glocker, C. Jones, M. Bernhardt, and S. Winzeck. Algorithmic encoding of protected characteristics in chest x-ray disease detection models. Ebiomedicine, 89, 2023. 





[63] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial networks, 2014. 





[64] V. Gorade, S. Mittal, and R. Singhal. Pacl: Patient-aware contrastive learning through metadata refinement for generalized early disease diagnosis. Computers in Biology and Medicine, page 107569, 2023. 





[65] D. A. Gutman, N. C. F. Codella, M. E. Celebi, B. Helba, M. A. Marchetti, N. K. Mishra, and A. C. Halpern. Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic). 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018), pages 168–172, 2018. 





[66] S. S. Halabi, L. M. Prevedello, J. Kalpathy-Cramer, A. B. Mamonov, A. Bilbily, M. Cicero, I. Pan, L. A. Pereira, R. T. Sousa, N. Abdala, et al. The rsna pediatric bone age machine learning challenge. Radiology, 290(2):498–503, 2019. 





[67] M. Hardt, E. Price, and N. Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems, 29:3315– 3323, 2016. 





[68] M. Hardt, E. Price, and N. Srebro. Equality of opportunity in supervised learning. In NIPS, 2016. 





[69] H. Heidari, V. Nanda, and K. P. Gummadi. On the long-term impact of algorithmic decision policies: Effort unfairness and feature segregation through social learning, 2019. 





[70] K. C. Heslin, P. L. Owens, Z. Karaca, M. L. Barrett, B. J. Moore, and A. Elixhauser. Trends in opioid-related inpatient stays shifted after the us transitioned to icd-10-cm diagnosis coding in 2015. Medical Care, 55:918–923, 2017. 





[71] A. Holzinger, G. Langs, H. Denk, K. Zatloukal, and H. Muller. Caus- ¨ ability and explainability of artificial intelligence in medicine. Wiley Interdisciplinary Reviews. Data Mining and Knowledge Discovery, 9, 2019. 





[72] F. M. Howard, J. M. Dolezal, S. E. Kochanny, J. J. Schulte, H. I.- H. Chen, L. R. Heij, D. Huo, R. Nanda, O. I. Olopade, J. N. Kather, N. A. Cipriani, R. L. Grossman, and A. T. Pearson. The impact of sitespecific digital histology signatures on deep learning model accuracy and bias. Nature Communications, 12, 2021. 





[73] J. A. Irvin, P. Rajpurkar, M. Ko, Y. Yu, S. Ciurea-Ilcus, C. Chute, H. Marklund, B. Haghgoo, R. L. Ball, K. S. Shpanskaya, J. Seekins, D. A. Mong, S. S. Halabi, J. K. Sandberg, R. Jones, D. B. Larson, C. Langlotz, B. N. Patel, M. P. Lungren, and A. Ng. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In AAAI, 2019. 





[74] S. Jabbour, D. F. Fouhey, E. A. Kazerooni, M. W. Sjoding, and J. Wiens. Deep learning applied to chest x-rays: Exploiting and preventing shortcuts. In MLHC, 2020. 





[75] Z. Jiang, X. Han, C. Fan, F. Yang, A. Mostafavi, and X. Hu. Generalized demographic parity for group fairness. In International Conference on Learning Representations, 2021. 





[76] A. Johnson, L. Bulgarelli, T. Pollard, S. Horng, L. Celi, and R. Mark. Mimic-iv (version 0.4), physionet, 2020. 





[77] A. E. Johnson, T. J. Pollard, L. Shen, H. L. Li-Wei, M. Feng, M. Ghassemi, B. Moody, P. Szolovits, L. A. Celi, and R. G. Mark. Mimic-iii, a freely accessible critical care database. Scientific data, 3(1):1–9, 2016. 





[78] A. E. W. Johnson, T. J. Pollard, S. J. Berkowitz, N. R. Greenbaum, M. P. Lungren, C. ying Deng, R. G. Mark, and S. Horng. Mimic-cxr: A large publicly available database of labeled chest radiographs. ArXiv, abs/1901.07042, 2019. 





[79] F. Kamiran and T. Calders. Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems, 33:1–33, 2011. 





[80] N. M. Kinyanjui, T. Odonga, C. Cintas, N. C. Codella, R. Panda, P. Sattigeri, and K. R. Varshney. Fairness of classifiers across skin tones in dermatology. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 320–329. Springer, 2020. 





[81] V. Kumar, A. Stubbs, S. Shaw, and O. Uzuner. Creation of a new ¨ longitudinal corpus of clinical narratives. Journal of biomedical informatics, 58:S6–S10, 2015. 





[82] M. Kuppler, C. Kern, R. L. Bach, and F. Kreuter. Distributive justice and fairness metrics in automated decision-making: How much overlap is there? arXiv preprint arXiv:2105.01441, 2021. 





[83] M. J. Kusner, J. R. Loftus, C. Russell, and R. Silva. Counterfactual fairness. In NIPS, 2017. 





[84] G. H. Kwak and P. Hui. Deephealth: Review and challenges of artificial intelligence in health informatics. arXiv: Learning, 2019. 





[85] J. Lamont. Distributive justice. Routledge, 2017. 





[86] A. J. Larrazabal, N. Nieto, V. Peterson, D. H. Milone, and E. Ferrante. Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis. Proceedings of the National Academy of Sciences of the United States of America, 117:12592 – 12594, 2020. 





[87] J.-G. Lee, Y. Roh, H. Song, and S. E. Whang. Machine learning robustness, fairness, and their convergence. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 4046–4047, 2021. 





[88] S. Li, T. Cai, and R. Duan. Targeting underrepresented populations in precision medicine: A federated transfer learning approach. The Annals of Applied Statistics, 17(4):2970–2992, 2023. 





[89] X. Li, Z. Cui, Y. Wu, L. Gu, and T. Harada. Estimating and improving fairness with adversarial learning. arXiv preprint arXiv:2103.04243, 2021. 





[90] Z. Lin, D. Zhang, Q. Tac, D. Shi, G. Haffari, Q. Wu, M. He, and Z. Ge. Medical visual question answering: A survey. arXiv preprint arXiv:2111.10056, 2021. 





[91] L. T. Liu, S. Dean, E. Rolf, M. Simchowitz, and M. Hardt. Delayed impact of fair machine learning. ArXiv, abs/1803.04383, 2018. 





[92] Q. Liu, L. Yu, L. Luo, Q. Dou, and P.-A. Heng. Semi-supervised medical image classification with relation-driven self-ensembling model. IEEE Transactions on Medical Imaging, 39:3429–3440, 2020. 





[93] H. J. Lowe, T. A. Ferris, P. M. Hernandez, and S. C. Weber. Stride–an integrated standards-based translational research informatics platform. In AMIA Annual Symposium Proceedings, volume 2009, page 391. American Medical Informatics Association, 2009. 





[94] C. Lu, A. Lemay, K. Chang, K. Hoebel, and J. Kalpathy-Cramer. Fair conformal predictors for applications in medical imaging. ArXiv, abs/2109.04392, 2021. 





[95] C. Lu, A. Lemay, K. Hoebel, and J. Kalpathy-Cramer. Evaluating subgroup disparity using epistemic uncertainty in mammography. ArXiv, abs/2107.02716, 2021. 





[96] D. Madras, E. Creager, T. Pitassi, and R. Zemel. Learning adversarially fair and transferable representations. In International Conference on Machine Learning, pages 3384–3393. PMLR, 2018. 





[97] C. A. McCarty, R. L. Chisholm, C. G. Chute, I. J. Kullo, G. P. Jarvik, E. B. Larson, R. Li, D. R. Masys, M. D. Ritchie, D. M. Roden, et al. The emerge network: a consortium of biorepositories linked to electronic medical records data for conducting genomic studies. BMC medical genomics, 4:1–11, 2011. 





[98] N. Mehrabi, F. Morstatter, N. A. Saxena, K. Lerman, and A. G. Galstyan. A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54:1 – 35, 2021. 





[99] C. Meng, L. Trinh, N. Xu, and Y. Liu. Mimic-if: Interpretability and fairness evaluation of deep learning models on mimic-iv dataset. ArXiv, abs/2102.06761, 2021. 





[100] J. R. Minot, N. Cheney, M. E. Maier, D. C. Elbers, C. M. Danforth, and P. S. Dodds. Interpretable bias mitigation for textual data: Reducing gender bias in patient notes while maintaining classification performance. ArXiv, abs/2103.05841, 2021. 





[101] M. Nguyen. Predicting cardiovascular risk using electronic health records. 2019. 





[102] R. B. Parikh, S. Teeple, and A. S. Navathe. Addressing bias in artificial intelligence in health care. JAMA, 2019. 





[103] R. C. Petersen, P. S. Aisen, L. A. Beckett, M. C. Donohue, A. C. Gamst, D. J. Harvey, C. R. Jack, W. J. Jagust, L. M. Shaw, A. W. Toga, et al. Alzheimer’s disease neuroimaging initiative (adni): clinical characterization. Neurology, 74(3):201–209, 2010. 





[104] A. Pfefferbaum, N. M. Zahr, S. A. Sassoon, D. Kwon, K. M. Pohl, and E. V. Sullivan. Accelerated and premature aging characterizing regional cortical volume loss in human immunodeficiency virus infection: contributions from alcohol, substance use, and hepatitis c coinfection. Biological Psychiatry: Cognitive Neuroscience and Neuroimaging, 3(10):844–859, 2018. 





[105] S. Pfohl, Y. Xu, A. Foryciarz, N. Ignatiadis, J. Genkins, and N. Shah. Net benefit, calibration, threshold selection, and training objectives for algorithmic fairness in healthcare. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 1039–1052, 2022. 





[106] S. R. Pfohl, T. Duan, D. Y. Ding, and N. H. Shah. Counterfactual reasoning for fair clinical risk prediction. ArXiv, abs/1907.06260, 2019. 





[107] S. R. Pfohl, B. J. Marafino, A. Coulet, F. Rodriguez, L. P. Palaniappan, and N. H. Shah. Creating fair models of atherosclerotic cardiovascular disease risk. Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, 2019. 





[108] R. Pinot, F. Yger, C. Gouy-Pailler, and J. Atif. A unified view on differential privacy and robustness to adversarial examples. arXiv preprint arXiv:1906.07982, 2019. 





[109] G. Pleiss, M. Raghavan, F. Wu, J. M. Kleinberg, and K. Q. Weinberger. On fairness and calibration. In NIPS, 2017. 





[110] E. Puyol-Anton, B. Ruijsink, J. M. Harana, S. K. Piechnik, ´ S. Neubauer, S. E. Petersen, R. Razavi, P. J. Chowienczyk, and A. P. King. Fairness in cardiac magnetic resonance imaging: Assessing sex and racial bias in deep learning-based segmentation. In medRxiv, 2021. 





[111] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. 2019. 





[112] A. Rajkomar, M. Hardt, M. D. Howell, G. Corrado, and M. H. Chin. Ensuring fairness in machine learning to advance health equity. Annals of Internal Medicine, 169:866–872, 2018. 





[113] A. Rajkomar, M. Hardt, M. D. Howell, G. Corrado, and M. H. Chin. Ensuring fairness in machine learning to advance health equity. Annals of internal medicine, 169(12):866–872, 2018. 





[114] J.-F. Rajotte, S. Mukherjee, C. Robinson, A. Ortiz, C. West, J. L. Ferres, and R. T. Ng. Reducing bias and increasing utility by federated generative modeling of medical images using a centralized adversary. arXiv preprint arXiv:2101.07235, 2021. 





[115] M. A. Ricci Lara, R. Echeveste, and E. Ferrante. Addressing fairness in artificial intelligence for medical imaging. nature communications, 13(1):4581, 2022. 





[116] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 





[117] B. Scherrer, A. Gholipour, and S. Warfield. Super-resolution reconstruction to increase the spatial resolution of diffusion weighted images from orthogonal anisotropic acquisitions. Medical image analysis, 16 7:1465–76, 2012. 





[118] J. Schrouff, N. Harris, O. Koyejo, I. Alabdulmohsin, E. Schnider, K. Opsahl-Ong, A. Brown, S. Roy, D. Mincu, C. Chen, et al. Diagnosing failures of fairness transfer across distribution shift in real-world medical settings: Supplement. 





[119] L. Seyyed-Kalantari, G. Liu, M. B. A. McDermott, and M. Ghassemi. Chexclusion: Fairness gaps in deep chest x-ray classifiers. Pacific Symposium on Biocomputing. Pacific Symposium on Biocomputing, 26:232–243, 2021. 





[120] L. Seyyed-Kalantari, H. Zhang, M. B. McDermott, I. Y. Chen, and M. Ghassemi. Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. Nature medicine, 27(12):2176–2182, 2021. 





[121] X. Shen, S. Ma, P. Vemuri, and G. Simon. Challenges and opportunities with causal discovery algorithms: application to alzheimer’s pathophysiology. Scientific reports, 10(1):2975, 2020. 





[122] J. W. Smith, J. E. Everhart, W. Dickson, W. C. Knowler, and R. S. Johannes. Using the adap learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the annual symposium on computer application in medical care, page 261. American Medical Informatics Association, 1988. 





[123] A. Subbaswamy and S. Saria. From development to deployment: dataset shift, causality, and shift-stable models in health ai. Biostatistics, 21(2):345–352, 2020. 





[124] A. Tahir, L. Cheng, and H. Liu. Fairness through aleatoric uncertainty. arXiv preprint arXiv:2304.03646, 2023. 





[125] R. J. Tibshirani and B. Efron. An introduction to the bootstrap. Monographs on statistics and applied probability, 57(1), 1993. 





[126] P. Tiwald, A. Ebert, and D. Soukup. Representative & fair synthetic data. ArXiv, abs/2104.03007, 2021. 





[127] E. J. Topol. High-performance medicine: the convergence of human and artificial intelligence. Nature medicine, 25(1):44–56, 2019. 





[128] Y.-H. H. Tsai, M. Q. Ma, H. Zhao, K. Zhang, L.-P. Morency, and R. Salakhutdinov. Conditional contrastive learning: Removing undesirable information in self-supervised representations. arXiv preprint arXiv:2106.02866, 2021. 





[129] P. Tschandl, C. Rosendahl, and H. Kittler. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 2018. 





[130] C. Wachinger and M. Reuter. Domain adaptation for alzheimer’s disease diagnostics. NeuroImage, 139:470–479, 2016. 





[131] T. Wang, J. Zhao, M. Yatskar, K.-W. Chang, and V. Ordonez. Balanced datasets are not enough: Estimating and mitigating gender bias in deep image representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5310–5319, 2019. 





[132] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3462–3471, 2017. 





[133] X. Wang, Y. Zhang, and R. Zhu. A brief review on algorithmic fairness. Management System Engineering, 1(1):7, 2022. 





[134] J. R. Williams and N. Razavian. Towards quantification of bias in machine learning for healthcare: A case study of renal failure prediction. ArXiv, abs/1911.07679, 2019. 





[135] J. Xu, Y. Xiao, W. H. Wang, Y. Ning, E. A. Shenkman, J. Bian, and F. Wang. Algorithmic fairness in computational medicine. medRxiv, 2022. 





[136] J. N. Xu, Y. Xiao, W. Wang, Y. Ning, E. A. Shenkman, J. Bian, and F. Wang. Algorithmic fairness in computational medicine. In medRxiv, 2022. 





[137] Y. Xu, T. Mo, Q. Feng, P. Zhong, M. Lai, and E. I.-C. Chang. Deep learning of feature representation with multiple instance learning for medical image analysis. 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1626–1630, 2014. 





[138] C. Xue, Q. Dou, X. Shi, H. Chen, and P.-A. Heng. Robust learning at noisy labeled medical images: Applied to skin lesion classification. 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), pages 1280–1283, 2019. 





[139] M. Yuan, V. Kumar, M. A. Ahmad, and A. Teredesai. Assessing fairness in classification parity of machine learning models in healthcare. arXiv preprint arXiv:2102.03717, 2021. 





[140] M. B. Zafar, I. Valera, M. Gomez-Rodriguez, and K. P. Gummadi. Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. Proceedings of the 26th International Conference on World Wide Web, 2017. 





[141] D. Zhang and J. Wang. Recommendation fairness: From static to dynamic. arXiv preprint arXiv:2109.03150, 2021. 





[142] H. Zhang, A. X. Lu, M. Abdalla, M. B. A. McDermott, and M. Ghassemi. Hurtful words: quantifying biases in clinical contextual word embeddings. Proceedings of the ACM Conference on Health, Inference, and Learning, 2020. 





[143] Q. Zhao, E. Adeli, and K. M. Pohl. Training confounder-free deep learning models for medical applications. Nature communications, 11(1):6010, 2020. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/8dc480bb-d4ad-4238-a27c-c9b1c4605b3d/0a645b469abd54ca57887f2520af6b0ae8798bf3c587c0fc0b0fe0d5fce9aa71.jpg)


Qizhang Feng received the B.Eng. degree in Electrical Engineering and Automation from the Huazhong University of Science and Technology, Hubei, China, in 2017, and the master’s degree in Electrical and Computer Engineering from Duke University, NC, USA, in 2020. He is currently working toward the Ph.D. degree in computer engineering with DATA Lab, Texas A&M University, TX, USA. His research interests include XAI, machine learning fairness and graph learning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/8dc480bb-d4ad-4238-a27c-c9b1c4605b3d/3214c71771dc6c34ded2cdac3557497bc55c5e6d6da398361c483cead1c414a8.jpg)


Dr. Mengnan Du is currently an is an Assistant Professor in the Department of Data Science, New Jersey Institute of Technology (NJIT). Mengnan Du earned his Ph.D. in Computer Science from Texas A&M University. He has previously worked/interned with Microsoft Research (MSR), Adobe Research, Intel, Baidu Research, Baidu Search Science and JD Explore Academy. His research covers a wide range of trustworthy machine learning topics, such as model explainability, fairness, and robustness. He has had more than 40 papers published in prestigious 

venues such as NeurIPS, AAAI, KDD, WWW, ICLR, and ICML. He received over 2,300 citations with an H-index of 16. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/8dc480bb-d4ad-4238-a27c-c9b1c4605b3d/4f6393df8bd6c769f32f467296ac7f341d5af563e9e9e051295e68cccc0aedbf.jpg)


Dr. Na Zou is currently a Corrie&Jim Furber ’64 assistant professor in Engineering Technology and Industrial Distribution at Texas A&M University. She was an Instructional Assistant Professor in Industrial and Systems Engineering at Texas A&M University from 2016 to 2020. She holds both a Ph.D. in Industrial Engineering and a MSE in Civil, Environmental and Sustainable Engineering from Arizona State University. Her research focuses on fair and interpretable machine learning, transfer learning, network modeling and inference, supported 

by NSF and industrial sponsors. The research projects have resulted in publications at prestigious journals such as Technometrics, IISE Transactions and ACM Transactions, including one Best Paper Finalist and one Best Student Paper Finalist at INFORMS QSR section and two featured articles at ISE Magazine. She was the recipient of IEEE Irv Kaufman Award and Texas A&M Institute of Data Science Career Initiation Fellow. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-23/8dc480bb-d4ad-4238-a27c-c9b1c4605b3d/a8a7b1182b0945c2d90bbf2b4c9cdbd9b8415d4c51666d7c994cf8682b52478e.jpg)


Dr. Xia “Ben” Hu is an Associate Professor at Rice University in the Department of Computer Science. Dr. Hu has published over 100 papers in several major academic venues, including NeurIPS, ICLR, KDD, WWW, IJCAI, AAAI, etc. An open-source package developed by his group, namely AutoKeras, has become the most used automated deep learning system on Github (with over 8,000 stars and 1,000 forks). Also, his work on deep collaborative filtering, anomaly detection and knowledge graphs have been included in the TensorFlow package, Apple 

production system and Bing production system, respectively. His papers have received several Best Paper (Candidate) awards from venues such as WWW, WSDM and ICDM. He is the recipient of NSF CAREER Award and ACM SIGKDD Rising Star Award. His work has been cited more than 12,000 times with an h-index of 43. He was the conference General Co-Chair for WSDM 2020. 