# GONE
With the wide applications of deep neural networks (DNNs) in various fields, current research shows their serious security risks due to the lack of privacy protection. Observing the DNN’s outputs allows the adversary to perform a membership inference attack (MIA, i.e., infer the training dataset’s attributes) or a model stealing attack (MSA, i.e., replicate the model’s functionality by training a clone). Many defense solutions against the above attacks have been proposed. However, most of them are still limited from three aspects: defense generality, efficiency, and model performance. To overcome the challenges, we propose GONE, a Generic O(1) NoisE layer added behind any DNNs’ output layer at the testing phase, which differs from previous work in three key aspects: (1) generic - it not only protects model privacy but also resists adversarial attacks; (2) efficient - it only takes O(1) complexity through blurring the probability distribution; (3) model-free - it does not affect model classification performance of clean data due to its monotonic. We evaluate GONE on several popular DNNs and demonstrate its outperformance compared to state-of-the-art baselines. Its average F1 measure drops from 0.817 to 0.062 against MIA. Its average accuracy rate drops from 0.771 to 0.197 against MSA. Besides, GONE achieves superior performance in defending against score-based adversarial attacks, with the average attack success rate drops from 0.870 to 0.059. 
