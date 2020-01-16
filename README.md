# Awesome-Weakly/Semi-Supervised-Learning
For anyone who wants to do research about weakly-/semi-supervised learning.   

If you find the awesome paper/code/dataset or have some suggestions, please contact cuiyiming@ufl.edu. Thanks for your valuable contribution to the research community :smiley:   

There is another awesome collections for [[awesome point cloud](https://github.com/Yochengliu/awesome-point-cloud-analysis)]. Thanks Yongcheng and Hua for their templates.

<h1> 

```diff
- Recent papers (from 2015)
```

</h1>

<h3> Keywords </h3>

__`dat.`__: dataset &emsp; | &emsp; __`cls.`__: classification &emsp; | &emsp; __`rel.`__: retrieval &emsp; | &emsp; __`seg.`__: segmentation     
__`det.`__: detection &emsp; | &emsp; __`tra.`__: tracking &emsp; | &emsp; __`pos.`__: pose &emsp; | &emsp; __`dep.`__: depth     
__`reg.`__: registration &emsp; | &emsp; __`rec.`__: reconstruction &emsp; | &emsp; __`aut.`__: autonomous driving     
__`oth.`__: other, including normal-related, correspondence, mapping, matching, alignment, compression, generative model...

---
## 2019
<!---
- [[CVPR](https://arxiv.org/pdf/1904.11693.pdf)] Box-driven Class-wise Region Masking and Filling Rate Guided Loss for Weakly Supervised Semantic Segmentation. [__`seg.`__] 
--->
- [[CVPR](https://arxiv.org/pdf/1902.10421.pdf)] FickleNet: Weakly and Semi-supervised Semantic Image Segmentation using Stochastic Inference. [__`seg.`__] 
- [[CVPR](https://arxiv.org/pdf/1904.05044.pdf)] Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shen_Cyclic_Guidance_for_Weakly_Supervised_Joint_Detection_and_Segmentation_CVPR_2019_paper.pdf)] Cyclic Guidance for Weakly Supervised Joint Detection and Segmentation. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Learning_Instance_Activation_Maps_for_Weakly_Supervised_Instance_Segmentation_CVPR_2019_paper.pdf)] Learning Instance Activation Maps for Weakly Supervised Instance Segmentation. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_D3TW_Discriminative_Differentiable_Dynamic_Time_Warping_for_Weakly_Supervised_Action_CVPR_2019_paper.pdf)] D3TW: Discriminative Differentiable Dynamic Time Warping for Weakly Supervised Action Alignment and Segmentation. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_Collaborative_Learning_of_Semi-Supervised_Segmentation_and_Classification_for_Medical_Images_CVPR_2019_paper.pdf)] Collaborative Learning of Semi-Supervised Segmentation and Classification for Medical Images. [__`seg.`__] 
- [[CVPR](https://arxiv.org/pdf/1904.01665.pdf)] Activity Driven Weakly Supervised Object Detection. [__`det.`__] 
- [[CVPR](https://arxiv.org/abs/1903.02827.pdf)] Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification from the Bottom Up. [__`cls.`__] 
- [[CVPR](https://arxiv.org/abs/1905.01298.pdf)] SCOPS: Self-Supervised Co-Part Segmentation. [[pytorch](https://github.com/NVlabs/SCOPS)][__`seg.`__] 
- [[CVPR](https://arxiv.org/abs/1902.09868.pdf)] RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation. [__`pos.`__] 
- [[CVPR](https://arxiv.org/abs/1903.02330.pdf)] Self-Supervised Learning of 3D Human Pose using Multi-view Geometry. [[pytorch](https://github.com/mkocabas/EpipolarPose)][__`pos.`__] 
- [[CVPR](https://arxiv.org/abs/1903.08839.pdf)] Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation. [__`pos.`__] 
- [[CVPR](http://www.vision.ee.ethz.ch/~wanc/papers/cvpr2019.pdf)] Self supervised 3D hand pose estimation. [[pytorch](https://github.com/melonwan/sphereHand)][__`pos.`__] 
- [[CVPR](https://research.fb.com/wp-content/uploads/2019/05/3D-human-pose-estimation-in-video-with-temporal-convolutions-and-semi-supervised-training.pdf)] 3D human pose estimation in video with temporal convolutions and semi-supervised training. [__`pos.`__] 
- [[CVPR](https://arxiv.org/abs/1811.11212.pdf)] Self-Supervised Generative Adversarial Networks. [[tensorflow](https://github.com/google/compare_gan)][__`oth.`__] 
- [[CVPR](https://arxiv.org/abs/1811.11212.pdf)] Self-Supervised GANs via Auxiliary Rotation Loss. [__`oth.`__] 
- [[CVPR](https://arxiv.org/abs/1904.10037.pdf)] LBS Autoencoder: Self-supervised Fitting of Articulated Meshes to Point Clouds. [__`oth.`__] 
- [[CVPR](https://arxiv.org/abs/1811.10092.pdf)] Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1806.05804.pdf)] Weakly Supervised Deep Image Hashing through Tag Embeddings. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1812.02415.pdf)] Self-supervised Learning of Dense Shape Correspondence. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1808.06088.pdf)] Tangent-Normal Adversarial Regularization for Semi-supervised Learning. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1903.08225.pdf)] TCross-task weakly supervised learning from instructional videos. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1903.11412.pdf)] Self-Supervised Learning via Conditional Motion Propagation. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1904.04717.pdf)] Label Propagation for Deep Semi-supervised Learning. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1904.05647.pdf)] C-MIL: Continuation Multiple Instance Learning for Weakly Supervised Object Detection. [__`det.`__]
- [[CVPR](https://arxiv.org/abs/1904.08208.pdf)] Guided Anisotropic Diffusion and Iterative Learning for Weakly Supervised Change Detection. [__`det.`__]
- [[CVPR](https://arxiv.org/abs/1904.09117.pdf)] SelFlow: Self-Supervised Learning of Optical Flow. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1904.13179.pdf)] Weakly Supervised Open-set Domain Adaptation by Dual-domain Collaboration. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1905.00149.pdf)] Self-Supervised Convolutional Subspace Clustering Network. [__`oth.`__]
- [[CVPR](https://arxiv.org/pdf/1901.09993.pdf)] Label Efficient Semi-Supervised Learning via Graph Filtering. [__`oth.`__]
- [[IJCAI](https://www.researchgate.net/publication/334844257_Boundary_Perception_Guidance_A_Scribble-Supervised_Semantic_Segmentation_Approach)] Boundary Perception Guidance: A Scribble-Supervised Semantic Segmentation Approach. [__`seg.`__] 
- [[IJCAI]()] Belief Propagation Network for Hard Inductive Semi-Supervised Learning. [__`oth.`__]
- [[IJCAI]()] Comprehensive Semi-Supervised Multi-Modal Learning. [__`oth.`__]
- [[IJCAI]()] Deep Correlated Predictive Subspace Learning for Incomplete Multi-View Semi-Supervised Classification. [__`cls.`__]
- [[IJCAI]()] Dual-View Variational Autoencoders for Semi-Supervised Text Matching. [__`oth.`__]
- [[IJCAI]()] GANs for Semi-Supervised Opinion Spam Detection. [__`det.`__]
- [[IJCAI]()] Graph Convolutional Networks using Heat Kernel for Semi-supervised Learning. [__`oth.`__]
- [[IJCAI]()] Hierarchical Graph Convolutional Networks for Semi-supervised Node Classification. [__`cls.`__]
- [[IJCAI]()] Interpolation Consistency Training for Semi-supervised Learning. [__`oth.`__]
- [[IJCAI]()] Low Shot Box Correction for Weakly Supervised Object Detection. [__`det.`__]
- [[IJCAI]()] MLRDA: A Multi-Task Semi-Supervised Learning Framework for Drug-Drug Interaction Prediction. [__`oth.`__]
- [[IJCAI]()] Prototype Propagation Networks (PPN) for Weakly-supervised Few-shot Learning on Category Graph. [__`oth.`__]
- [[IJCAI]()] Quadruply Stochastic Gradients for Large-Scale Nonlinear Semi-Supervised AUC Optimization. [__`oth.`__]
- [[IJCAI]()] Scalable Semi-Supervised SVM via Triply Stochastic Gradients. [__`oth.`__]
- [[IJCAI]()] Semi-supervised Three-dimensional Reconstruction Framework with GAN. [__`rec.`__]
- [[IJCAI]()] Semi-supervised User Profiling with Heterogeneous Graph Attention Networks. [__`oth.`__]
- [[IJCAI]()] Weakly Supervised Multi-Label Learning via Label Enhancement. [__`oth.`__]
- [[IJCAI]()] Weakly Supervised Multi-task Learning for Semantic Parsing. [__`oth.`__]
- [[AAAI](https://www.aaai.org/ojs/index.php/AAAI/article/view/3860)] Weakly-Supervised Simultaneous Evidence Identification and Segmentation for Automated Glaucoma Diagnosis. [__`seg.`__] 
- [[AAAI](https://arxiv.org/pdf/1807.11719.pdf)] A Two-Stream Mutual Attention Network for Semi-supervised Biomedical Segmentation with Noisy Labels. [__`seg.`__] 
- [[AAAI]()] Transferable Curriculum for Weakly-Supervised Domain Adaptation. [__`oth.`__]
- [[AAAI]()] Adversarial Learning for Weakly-Supervised Social Network Alignment. [__`oth.`__]
- [[AAAI]()] Joint Semi-supervised Feature Selection and Classification Through Bayesian Approach. [__`oth.`__]
- [[AAAI]()] Weakly Supervised Scene Parsing with Point-based Distance Metric Learning. [__`oth.`__]
- [[AAAI]()] Self-Supervised Video Representation Learning with Space-Time Cubic Puzzles. [__`oth.`__]
- [[AAAI]()] Segregated Temporal Assembly Recurrent Networks for Weakly Supervised Multiple
Action Detection. [__`det.`__]
- [[AAAI]()] Distribution-based Semi-Supervised Learning for Activity Recognition. [__`det.`__]
- [[AAAI]()] Self-Supervised Mixture-of-Experts by Uncertainty Estimation. [__`oth.`__]
- [[AAAI]()] Dual Semi-Supervised Learning for Facial Action Unit Recognition. [__`det.`__]
- [[AAAI]()] LabelForest: Non-Parametric Semi-Supervised Learning for Activity Recognition. [__`det.`__]
- [[AAAI]()] A Topic-Aware Reinforced Model for Weakly Supervised Stance Detection. [__`det.`__]
- [[AAAI]()] Weakly-Supervised Hierarchical Text Classification. [__`cls.`__]
- [[AAAI]()] Markov Random Field meets Graph Convolutional Network: End-to-End Learning for
Semi-Supervised Community Detection. [__`det.`__]
- [[AAAI]()] Mixture of Expert/Imitator Network: Scalable Semi-supervised Learning Framework. [__`oth.`__]
- [[AAAI]()] Human-like Delicate Region Erasing Strategy for Weakly Supervised Detection. [__`det.`__]
- [[AAAI]()] Exploiting synthetically generated data with semi-supervised learning for small and
imbalanced datasets. [__`oth.`__]
- [[AAAI]()] Distributionally Robust Semi-supervised Learning for People-centric Sensing. [__`oth.`__]
- [[AAAI]()] Bayesian Graph Convolutional Neural Networks for Semi-supervised Classification. [__`cls.`__]
- [[AAAI]()] AffinityNet: Semi-supervised Few-shot Learning for Disease Type Prediction. [__`oth.`__]
- [[AAAI]()] Revisiting LSTM Networks for Semi-Supervised Text Classification via Mixed Objective Function. [__`cls.`__]
- [[AAAI]()] Matrix Completion for Graph-Based Deep Semi-Supervised Learning. [__`oth.`__]
- [[AAAI]()] Towards Automated Semi-Supervised Learning. [__`oth.`__]
---
## 2018
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0812.pdf)] Revisiting Dilated Convolution: A Simple Approach for Weakly- and Semi-Supervised Semantic Segmentation. [__`seg.`__] 
- [[CVPR](https://arxiv.org/pdf/1806.04659.pdf)] Weakly-Supervised Semantic Segmentation by Iteratively Mining Common Object Features.
- [[CVPR](http://zpascal.net/cvpr2018/Huang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf)] Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing. [[caffe](https://github.com/speedinghzl/DSRG)] [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Bootstrapping_the_Performance_CVPR_2018_paper.pdf)] Bootstrapping the Performance of Webly Supervised Semantic Segmentation. [[caffe](https://github.com/ascust/BDWSS)] [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Weakly_Supervised_Instance_CVPR_2018_paper.pdf)] Weakly Supervised Instance Segmentation using Class Peak Response.  [[pytorch](https://github.com/ZhouYanzhao/PRM)][__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Fang_Weakly_and_Semi_CVPR_2018_paper.pdf)] Weakly and Semi Supervised Human Body Part Parsing via Pose-Guided Knowledge Transfer. [__`pos.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ren_Cross-Domain_Self-Supervised_Multi-Task_CVPR_2018_paper.pdf)] Cross-Domain Self-Supervised Multi-Task Feature Learning Using Synthetic Imagery. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_W2F_A_Weakly-Supervised_CVPR_2018_paper.pdf)] W2F: A Weakly-Supervised to Fully-Supervised Framework for Object Detection. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ge_Multi-Evidence_Filtering_and_CVPR_2018_paper.pdf)] Multi-Evidence Filtering and Fusion for Multi-Label Classification, Object Detection and Semantic Segmentation Based on Weakly Supervised Learning. [__`cls.`__, __`seg.`__,__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wan_Min-Entropy_Latent_Model_CVPR_2018_paper.pdf)] Min-Entropy Latent Model for Weakly Supervised Object Detection. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.pdf)] WAdversarial Complementary Learning for Weakly Supervised Object Localization. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf)] Weakly-Supervised Semantic Segmentation by Iteratively Mining Common Object Features. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Bootstrapping_the_Performance_CVPR_2018_paper.pdf)] Bootstrapping the Performance of Webly Supervised Semantic Segmentation. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cheng_Cube_Padding_for_CVPR_2018_paper.pdf)] Cube Padding for Weakly-Supervised Saliency Prediction in 360° Videos. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Honari_Improving_Landmark_Localization_CVPR_2018_paper.pdf)] Improving Landmark Localization With Semi-Supervised Learning. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Towards_Human-Machine_Cooperation_CVPR_2018_paper.pdf)] Towards Human-Machine Cooperation: Self-Supervised Sample Mining for Object Detection. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tang_Normalized_Cut_Loss_CVPR_2018_paper.pdf)] Normalized Cut Loss for Weakly-Supervised CNN Segmentation. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Learning_Facial_Action_CVPR_2018_paper.pdf)] Learning Facial Action Units From Web Images With Scalable Weakly Supervised Clustering. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Peng_Weakly_Supervised_Facial_CVPR_2018_paper.pdf)] Weakly Supervised Facial Action Unit Recognition Through Adversarial Training. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Weakly-Supervised_Deep_Convolutional_CVPR_2018_paper.pdf)] Weakly-Supervised Deep Convolutional Neural Network Learning for Facial Action Unit Intensity Estimation. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tewari_Self-Supervised_Multi-Level_Face_CVPR_2018_paper.pdf)] Self-Supervised Multi-Level Face Model Learning for Monocular Reconstruction at Over 250 Hz. [__`rec.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jenni_Self-Supervised_Feature_Learning_CVPR_2018_paper.pdf)] Self-Supervised Feature Learning by Learning to Spot Artifacts. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Novotny_Self-Supervised_Learning_of_CVPR_2018_paper.pdf)] Self-Supervised Learning of Geometrically Stable Features Through Probabilistic Introspection. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Weakly_Supervised_Instance_CVPR_2018_paper.pdf)] Weakly Supervised Instance Segmentation Using Class Peak Response. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Knowledge_Aided_Consistency_CVPR_2018_paper.pdf)] Knowledge Aided Consistency for Weakly Supervised Phrase Grounding. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.pdf)] Data Distillation: Towards Omni-Supervised Learning. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Self-Supervised_Adversarial_Hashing_CVPR_2018_paper.pdf)] Self-Supervised Adversarial Hashing Networks for Cross-Modal Retrieval. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Zigzag_Learning_for_CVPR_2018_paper.pdf)] Zigzag Learning for Weakly Supervised Object Detection. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Bouritsas_Multimodal_Visual_Concept_CVPR_2018_paper.pdf)] Multimodal Visual Concept Learning With Weakly Supervised Techniques. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.pdf)] Learning Pixel-Level Semantic Affinity With Image-Level Supervision for Weakly Supervised Semantic Segmentation. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Inoue_Cross-Domain_Weakly-Supervised_Object_CVPR_2018_paper.pdf)] Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gan_Geometry_Guided_Convolutional_CVPR_2018_paper.pdf)] Geometry Guided Convolutional Neural Networks for Self-Supervised Video Representation Learning. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Weakly_Supervised_Phrase_CVPR_2018_paper.pdf)] Weakly Supervised Phrase Localization With Multi-Scale Anchored Transformer Network. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Generative_Adversarial_Learning_CVPR_2018_paper.pdf)] Generative Adversarial Learning Towards Fast Weakly Supervised Detection. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Finding_It_Weakly-Supervised_CVPR_2018_paper.pdf)] Finding "It": Weakly-Supervised Reference-Aware Visual Grounding in Instructional Videos. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Richard_Action_Sets_Weakly_CVPR_2018_paper.pdf)] Action Sets: Weakly Supervised Action Segmentation Without Ordering Constraints. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ding_Weakly-Supervised_Action_Segmentation_CVPR_2018_paper.pdf)] Weakly-Supervised Action Segmentation With Iterative Soft Boundary Assignment. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Nguyen_Weakly_Supervised_Action_CVPR_2018_paper.pdf)] Weakly Supervised Action Localization by Sparse Temporal Pooling Network. [__`det.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Rocco_End-to-End_Weakly-Supervised_Semantic_CVPR_2018_paper.pdf)] End-to-End Weakly-Supervised Semantic Alignment. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf)] Weakly-Supervised Semantic Segmentation Network With Deep Seeded Region Growing. [__`seg.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Niu_Webly_Supervised_Learning_CVPR_2018_paper.pdf)] Webly Supervised Learning Meets Zero-Shot Learning: A Hybrid Approach for Fine-Grained Classification. [__`cls.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Richard_NeuralNetwork-Viterbi_A_Framework_CVPR_2018_paper.pdf)] NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Weakly_Supervised_Coupled_CVPR_2018_paper.pdf)] Weakly Supervised Coupled Networks for Visual Sentiment Analysis. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Smooth_Neighbors_on_CVPR_2018_paper.pdf)] Smooth Neighbors on Teacher Graphs for Semi-Supervised Learning. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mundhenk_Improvements_to_Context_CVPR_2018_paper.pdf)] Improvements to Context Based Self-Supervised Learning. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Noroozi_Boosting_Self-Supervised_Learning_CVPR_2018_paper.pdf)] Boosting Self-Supervised Learning via Knowledge Transfer. [__`oth.`__] 
- [[AAAI](https://arxiv.org/pdf/1711.06828.pdf)] Transferable Semi-supervised Semantic Segmentation. [__`seg.`__] 
- [[AAAI]()] A General Formulation for Safely Exploiting Weakly Supervised Data. [__`oth.`__] 
- [[AAAI]()] Adversarial Dropout for Supervised and Semi-Supervised Learnin. [__`oth.`__] 
- [[AAAI]()] ARC: Adversarial Robust Cuts for Semi-Supervised and Multi-Label Classification. [__`cls.`__] 
- [[AAAI]()] Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning. [__`oth.`__] 
- [[AAAI]()] DeepHeart: Semi-Supervised Sequence Learning for Cardiovascular Risk Prediction. [__`oth.`__] 
- [[AAAI]()] Inferring Emotion from Conversational Voice Data: A Semi-supervised Multi-path Generative Neural Network Approach. [__`oth.`__] 
- [[AAAI]()] Interpretable Graph-Based Semi-Supervised Learning via Flows. [__`oth.`__] 
- [[AAAI]()] Kill Two Birds with One Stone: Weakly-Supervised Neural Network for Image Annotation and Tag Refinement. [__`seg.`__] 
- [[AAAI]()] Learning from Semi-Supervised Weak-Label Data. [__`oth.`__] 
- [[AAAI]()] Mix-and-Match Tuning for Self-Supervised Semantic Segmentation. [__`seg.`__] 
- [[AAAI]()] SEE: Towards Semi-Supervised End-to-End Scene Text Recognition. [__`det.`__] 
- [[AAAI]()] Semi-Supervised AUC Optimization without Guessing Labels of Unlabeled Data. [__`oth.`__] 
- [[AAAI]()] Semi-supervised Bayesian Attribute Learning for Person Re-identification. [__`det.`__] 
- [[AAAI]()] Semi-supervised Biomedical Translation with Cycle Wasserstein Regression GANs. [__`oth.`__] 
- [[AAAI]()] Semi-supervised Learning from Crowds Using Deep Generative Models. [__`oth.`__] 
- [[AAAI]()] Transferable Semi-supervised Semantic Segmentation. [__`seg.`__] 
- [[AAAI]()] Weakly supervised collective feature learning from curated media. [__`oth.`__] 
- [[AAAI]()] Weakly Supervised Induction of Affective Events by Optimizing Semantic Consistency. [__`oth.`__] 
- [[AAAI]()] Weakly Supervised Salient Object Detection Using Image Labels. [__`det.`__] 
- [[ECCV](https://arxiv.org/abs/1807.04897.pdf)] TS2C: Tight Box Mining with Surrounding Segmentation Context for Weakly Supervised Object Detection. [__`seg.`__] 
- [[ECCV](https://arxiv.org/abs/1807.08902.pdf)] Self-produced Guidance for Weakly-supervised Object Localization. [__`seg.`__] 
- [[TPAMI](https://ieeexplore.ieee.org/document/8493315)] PCL: Proposal Cluster Learning for Weakly Supervised Object Detection. [__`det.`__]


---
## 2017
- [[AAAI](https://pdfs.semanticscholar.org/9345/23b3de05318606d4f550f5828cf30a56b1d3.pdf?_ga=2.30714812.2026882509.1564975284-400067050.1564456907)] Weakly Supervised Semantic Segmentation Using Superpixel Pooling Network. [__`seg.`__] 
- [[TPAMI](https://weiyc.github.io/assets/pdf/stc_tpami.pdf)] STC: A Simple to Complex Framework for Weakly-supervised Semantic Segmentation. [__`seg.`__] 

---
## 2016
- [[TPAMI](https://ieeexplore.ieee.org/document/7775087)] STC: A Simple to Complex Framework for Weakly-Supervised Semantic Segmentation. [__`seg.`__]
