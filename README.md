# Boosting Products of Decision Stumps

## Description
Boosting is a well-known ensemble learning strategy that often produces good results. As in other ensemble classiers, multiple base classifier models are combined to form an ensemble classier, and predictions of these base classiers form an ensemble prediction. These ensemble predictions are often much better than those of the individual base classifiers, and it is possible to combine so-called \weak" base classifiers to form powerful ensembles. In many practical applications, these weak learners are decision stumps (decision trees with a
single split) or, alternatively, depth-limited decision trees. Decision trees suffer from data fragmentation and are prone to overfitting, while decision stumps are sometimes too weak, even if combined into an ensemble. In (Kegl & Busa-Fekete, 2009), the authors propose products of decision stumps - models that are more complex than decision stumps but not as prone to overfitting as decision trees - as base classifiers in boosting. They claim very good results, in particular, on the MNIST data. This project is to verify this result on the MNIST data and also run experiments on the other image classification datasets used in "[K-means_Clustering](https://github.com/darwinsww/K-means_Clustering)", by writing a WEKA implementation of their method.

## Training Data
- [mnist-test](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/mnist-test.arff)
- [mnist-train](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/mnist-train.arff)
- [letter-test](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/letter-test.arff)
- [letter-train](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/letter-train.arff)
- [isolet-test](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/isolet-test.arff)
- [isolet-train](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/isolet-train.arff)
- [pendigits-test](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/pendigits-test.arff)
- [pendigits-train](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/pendigits-train.arff)
- [usps-test](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/usps-test.arff)
- [usps-train](https://www.cs.waikato.ac.nz/ml/521/old/2018/assignment2/usps-train.arff)

## System Environment
- OS - Ubuntu 18.04 on AWS EC2 
```
ubuntu@ip-172-31-8-70:~/ml/weka-3-8-2$ lsb_release -a
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 18.04.2 LTS
Release:	18.04
Codename:	bionic

ubuntu@ip-172-31-8-70:~/ml/weka-3-8-2$ uname -a
Linux ip-172-31-8-70 4.15.0-1044-aws #46-Ubuntu SMP Thu Jul 4 13:38:28 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
```

- JAVA
```
ubuntu@ip-172-31-8-70:~/ml/weka-3-8-2$ java -version
java version "1.8.0_171"
Java(TM) SE Runtime Environment (build 1.8.0_171-b11)
Java HotSpot(TM) 64-Bit Server VM (build 25.171-b11, mixed mode)
```

## Dependencies
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/downloading.html)  
Downlard ```"waka-x.x.x.zip"``` in the section ```"Other platforms (Linux, etc.)"```.   
Unzip the zip file and you will find the necessary jar packages. Here used the ```"weka-3-8-2.zip"```.  

- libxtst-dev 
  ![image](https://github.com/darwinsww/Boosting_products_of_decision_stumps/blob/master/img/libxtst-dev.png)
```
sudo apt-get install libxtst-dev
```

## Compile & Package
- Create Folders:   
```
mkdir -p ~/ml

cd ml
mkdir -p lib-stable-3-8-2
mkdir -p weka-3-8-2
mkdir -p Boosting_products_of_decision_stumps

cd Boosting_products_of_decision_stumps
mkdir -p ./data
mkdir -p ./src/weka/classifiers/meta
mkdir -p ./out/weka/classifiers/meta
```


- Copy Files:   
Copy ```"netlibNativeLinux1.0.2.zip"``` to the folder ```"~/ml/netlibNativeLinux1.0.2"```   
Copy ```"weka-3-8-2.zip"``` to the folder ```"~/ml/weka-3-8-2"```   
Copy ```"mtj.jar"``` and ```"weka.jar"``` to the folder ```"~/ml/lib-stable-3-8-2"```      
Copy ```"pendigits-train.arff"``` and ```"pendigits-test.arff"``` to the folder ```"~/ml/Boosting_products_of_decision_stumps/data"```     

- Extraction:   
```
cd ~/ml/weka-3-8-2
unzip weka-3-8-2.zip
```

- Compile:  


- Package:  


## Experiments
A lot of experiments had been done, following is an example command runs the test:



## Results
Several experiments have been conducted to test the performance of boosting products of decision stumps. The arff files with large amounts of features are the inputs, and the outputs are the classfications for each instance. In these experiments, I used 5 different classification problems (PENDIGITS, ISOLET, LETTER, USPS, and MNIST) with difierent parameter settings for each of them. The explanations of the parameters are as follows:   
-I: number of base learners h(x) in AdaBoost.MH, which also specifies the number of iterations boosting needs to perform to perform.  
-m: number of base learners in product learner, which specifies the size of product to use for the base classifiers.   

For comparison, I implemented at least 30 experiments with different combinations of I = 10, 50, 100, 500, 1000, 2000 and m = 1, 3, 5, 10, 30 for the 5 problems each. Some addtional tests for PENDIGITS when T = 5000, while tests of MNIST are not accomplished when T = 2000 due to memory limitation.  

By the experiments, the results are demonstrated in the following tables, which were generated using WEKA[5].  

### Accuracy
Regarding all the results, all the date sets have good accuracy (around 95%), when I = 1000 and m = 5. However, compared to the output of the C++ implementation of MultiBoost[4], there is still room for improvement of mine.
![image](https://github.com/darwinsww/Boosting_products_of_decision_stumps/blob/master/img/Accuracy.png)

## Conclusions
According to the researches above, we can draw a conclusion that:  

1. In most situations, the bigger m is, the higher accuracy we could get. However, with the increasing of m, the accuracy will not increase indefinitely. This is because the product learners are always converged before m base learners are all executed. So, there is no need to enlarge m continually in the experiments.  

2. It is obvious that more iterations would lead to higher accuracies in all the data sets. The reason is that one base learner, generated in each iteration, is prone to process the data with wrong classifications. So, theoretically, we could add more base learners to get better performace.   

The last but not the least, by these experiments, we can prove that the combination of Adaboost.MH and product learner with decision stumps could dramatically enhance the performance of individual base learner.

## References
[1] Djalel Benbouzid, Robert Busa-Fekete, Norman Casagrande, Francois-David Collin, and Balazs Kegl. MULTIBOOST: A multi-purpose boosting package. Journal of Machine Learning Research, 13:549-553, 2012.   
[Download Here](http://www.jmlr.org/papers/volume13/benbouzid12a/benbouzid12a.pdf)   

[2] Balazs Kegl. Open problem: A (missing) boosting-type convergence result for adaboost.mh with factorized multi-class classifiers. In Maria-Florina Balcan, Vitaly Feldman, and Csaba Szepesvari, editors, Proceedings of The 27th Conference on Learning Theory, COLT 2014, Barcelona, Spain, June 13-15, 2014, volume 35 of JMLR Workshop and Conference Proceedings, pages 1268{1275. JMLR.org, 2014.   
[Download Here](http://proceedings.mlr.press/v35/kegl14.pdf)   

[3] Balazs Kegl and Robert Busa-Fekete. Boosting products of base classifiers. In Andrea Pohoreckyj Danyluk, Leon Bottou, and Michael L. Littman, editors, Proceedings of the 26th Annual International Conference on Machine Learning, ICML 2009, Montreal, Quebec, Canada, June 14-18, 2009, volume 382 of ACM International Conference Proceeding Series, pages 497-504. ACM, 2009.      

[4] Robert E. Schapire and Yoram Singer. Improved boosting algorithms using confidence-rated predictions. Machine Learning, 37(3):297-336, 1999.    
[Download Here](https://link.springer.com/content/pdf/10.1023/A:1007614523901.pdf)   

[5] Ian H. Witten, Eibe Frank, and Mark A. Hall. Data Mining: Practical Machine Learning Tools and Techniques. Morgan Kaufmann, Burlington, MA, 3 edition, 2011.    

