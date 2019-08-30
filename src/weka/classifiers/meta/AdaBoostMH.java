/*
算法的当前实现是在 StumpAlgorithm.java 中在寻找最佳分叉点时，考虑首尾两个位置（即负无穷和正无穷），
使用的代码是 L81, L86, L107, L29, L182-L193. 用此方法更合理，因为是真的选择 energy 最小的位置，
从而可以将正负无穷作为 threshold。（实际最终代码也是使用此方法）

而为了方便调试，可以将自己的实现与 C++ Multiboost 的输出结果进行对比，
这就需要在 StumpAlgorithm.java 中在寻找最佳分叉点时，忽略首尾两个位置（即负无穷和正无穷），
此时使用的代码是 L82, L87, L108, L130, L194.
通过测试：
在不使用 Product Learner 方法对 IRIS 数据集测试时 如: I100M3，发现所得出的各个参数及 weights 值均与 C++ 实现相同；
在使用 Product Learner 方法对 IRIS 数据集测试时 如: I100M3，发现所得出的各个参数及 weights 值开始时与 C++ 实现相同，
但是后面就不一样了。
另外：用此方法并未在执行时间，提前收敛情况（甚至有时不如左面的方法：如 letter I500M30, I1000M30 等），
以及正确率上有明显优势，故实际执行时用的还是考虑首尾两个位置作为最佳分叉点。
 */
package weka.classifiers.meta;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.IterativeClassifier;
import weka.core.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class AdaBoostMH extends AbstractClassifier implements IterativeClassifier {
    // Extended instances, including all the original instances with wights and multiple labels, 
    // as well as all the sorted data by each attribute.
    // This object is the ONLY one data copy in the whole process. Several changes are made to the data in the process:
    // Their weights would be changed in the T (m_I here) iterations of getting T base learners.
    // Their labels would be changed in the m (m_M here) iterations of getting m product base learners, however, the
    // modified labels (virtual labels) would be restored to the original labels in the end of each of T iterations,
    // which outputs a product base learner.
    protected ExtendedInstances m_extendedInsts;

    // Keep all the base learners geeting from T (m_I here) iterations. They are not same because they have different
    // parameters: \alpha, vote vector V, function \varphi(x) (threshold and attribute used to split).
    // The final strong learner is consisted of these T (m_I here) base learners. It is defined as:
    // \vec{f}(x) = \sum_{t=1}^{T}h^{(t)}(x) = \sum_{t=1}^{T}\alpha^{(t)}*\vec{v}^{(t)}*\varphi^{(t)}(x)
    // Note: \vec{f} and \vec{v} are both column vectors with dimension K, each element of which is aimed at one class/labels.
    // Totally K classes/labels.
    protected ArrayList<BaseLearner> m_baseLearnerContainer;

    // The number of iterations have already executed.
    protected int m_numIterationsPerformed = 0;

    // The number of boosting iterations to perform (T in the paper).
    protected int m_I = 100;

    // The size of product to use for the base classifiers (m in the paper).
    protected int m_M = 3;

    @Override
    public void initializeClassifier(Instances rawdata) throws Exception {
        // Determine whether the classifier can handle the data
        getCapabilities().testWithFail(rawdata);

        // Make a copy of data and delete instances with a missing class value
        rawdata = new Instances(rawdata);
        rawdata.deleteWithMissingClass();

        m_extendedInsts = new ExtendedInstances(rawdata);
        m_baseLearnerContainer = new ArrayList<>();
    }

    // Iterate once, we get one base learner.
    @Override
    public boolean next() throws Exception {
        // Every time code goes here, a new iteration (totally m_I time) starts.
        // This process should totally irrelevant with all iterations.
        // So, we need to create a local variable as the new base learner to be a part of the strong learner.
        if (m_numIterationsPerformed >= m_I) {
            // The iterations are all finished.
            return false;
        } else if (m_extendedInsts.numAttributes() == 0) {
            // No attribute exists.
            return false;
        } else {
            System.out.println("<!-- ###############################" + " Iteration " + m_numIterationsPerformed
                    + " begins ############################### -->");

            if (m_Debug) {
                printWeights();
            }

            // Create a local variable as the new base learner.
            BaseLearner baseLearner = new ProductLearner();
            ((ProductLearner)baseLearner).setParameters ("DecisionStump", m_M);
            /*BaseLearner baseLearner = new SingleStumpLearner();*/

            // Different base learner uses the same training data "m_extendedInsts".
            baseLearner.buildClassifier(m_extendedInsts);

            // Update all the weights w_i_l of all the instances for next iteration.
            updateWeights(baseLearner);

            // Append current base learner to the end of base learner container.
            m_baseLearnerContainer.add(baseLearner);

            m_numIterationsPerformed++;
            return true;
        }
    }

    // Executed only after all the iterations are done.
    @Override
    public void done() throws Exception {
        // dummy
    }

    /**
     * Initialises the classifier from the given training instances.
     */
    public void buildClassifier(Instances rawdata) throws Exception {
        this.initializeClassifier(rawdata);

        if (m_Debug) {
            // Print labels after initialization of the raw data.
            printLabels();
        }

        // Execute the T iterations one by one
        while(this.next()) {
            // This iteration if finished, output the information of this base learner,
            // which is used to check all the parameters.

            BaseLearner baseLearner = m_baseLearnerContainer.get(m_numIterationsPerformed - 1);
            baseLearner.printLearnerInfo();
        }

        this.done();
    }

    // Do the classification of the given instance. The process handles the instances one by one instead of a bunch.
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (this.m_numIterationsPerformed == 0) {
            throw new Exception("No model built");
        }
        else {
            // We don't need to extend the instance to an ExtendedInstances class object, because we don't know
            // and don't need to know its labels, weights, what we need to do is to classify the given instance.
            double[] classification = new double[instance.numClasses()];

            // Iterate all the T base classifiers.
            for(int t = 0; t < m_numIterationsPerformed; t++) {
                // Get current base learner to get the classification.
                BaseLearner baseLearner = m_baseLearnerContainer.get(t);

                double alpha = baseLearner.getAlpha();
                for (int k = 0; k < instance.numClasses(); k++) {
                    classification[k] += alpha * baseLearner.doClassification(instance.toDoubleArray(), k);
                }
            }

            // For some reason, before the normalization, we need to exponent function to optimize our result.
            for(int l = 0; l < instance.numClasses(); l++) {
                classification[l] = Math.exp(classification[l]);
            }

            // This function (distributionForInstance) needs to return a probability distribution, so all the
            // values in the output array need to be in [0,1] and sum to 1.
            Utils.normalize(classification, Utils.sum(classification));

            return classification;
        }
    }

    // Update all the weights w_i_l of all the instances for next iteration.
    public void updateWeights (BaseLearner baseLearner) throws Exception {
        int numInstances = m_extendedInsts.numInstances();
        int numClasses = m_extendedInsts.numClasses();

        double alpha = baseLearner.getAlpha();

        // The normalization factor
        double Z = 0;

        // hy is a temporary variable for h(x)*y. Helps saving time during re-weighting.
        // The dimension of hy should be numInstances*numClasses (n*k), that is the size of weight(W) matrix.
        ArrayList<double[]> hy = new ArrayList<>();
        for (int i = 0; i < numInstances; ++i) {
            // Create n vectors (each for one instance) for hy, in which has a double array with the length of k(numClasses).
            hy.add(new double[numClasses]);
        }

        // Start to recompute weights and compute the normalization factor Z

        // For each instance (totally n instances)
        for (int i = 0; i < numInstances; ++i) {
            // Get all the labels and weights of the i-th instance.
            int[] labels = m_extendedInsts.getLabels(i);
            double[] weight = m_extendedInsts.getWeights(i);

            // For each label/weight (totally K)
            for (int l = 0; l < numClasses; l++) {
                // hy_i_l = h_l(x_i) * y_i_l, that is: hy_{i,l} = h_l\left ( x_{i} \right )y_{i,l} = v_l\varphi(x_i)y_{i,l}
                // The following function returns h_l(x_i) = v_l * \varphi(x_i)
                hy.get(i)[l] = baseLearner.doClassification(m_extendedInsts, i, l) * labels[l];

                // Formular on the right side is:   w_i_l * exp(-alpha * h_l(x_i) * y_i_l)
                // In latex, it is:  w_{i,l}e^{\alpha * h_l\left ( x_{i} \right ) * y_{i,l}}
                // So, Z equals to:  Z = \sum_{i=1}^{n}\sum_{l=1}^{k}w_{i,l}e^{-\alpha * h_l\left ( x_{i} \right ) * y_{i,l}}
                Z += weight[l] * Math.exp(-alpha * hy.get(i)[l]);
                // w_i_l  * exp( -alpha * h_l(x_i) * y_i )
            }
        }

        System.out.println("Update Weights: Z = " + Z);

        // Now do the actual re-weight
        // For each instance
        for (int i = 0; i < numInstances; ++i) {
            // Get all the weights of the i-th instance.
            double[] weight = m_extendedInsts.getWeights(i);

            // For each weight (totally K)
            // Calculate the new weight w'_i_l for next iteration.
            // w_{i,l}^{'}=w_{i,l}*\frac{e^{-\alpha*hy_{i,l}}}{Z}=w_{i,l}*\frac{e^{-\alpha*h_l\left ( x_{i} \right )*y_{i,l}}}{Z}=w_{i,l}*\frac{e^{-\alpha*v_l*\varphi(x_i)*y_{i,l}}}{Z}
            for (int l = 0; l < numClasses; l++) {
                weight[l] = weight[l] * Math.exp(-alpha * hy.get(i)[l]) / Z;
            }
        }
    }

    private void printLabels () {
        File labelsFile = new File("labels.txt");
        Writer outL = null;
        try {
            outL = new FileWriter(labelsFile,true);
            outL.write("---------------- Labels after initialization of the raw data ----------------\r\n");

            for (int i = 0; i < m_extendedInsts.numInstances(); i++) {
                int[] labels = m_extendedInsts.getLabels(i);
                for (int k = 0; k < m_extendedInsts.numClasses(); k++) {
                    outL.write((int)labels[k] + "        ");
                }
                outL.write("\r\n");
            }
            outL.write("\r\n\r\n");
            outL.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void printWeights () {
        File weightsFile = new File("weights.txt");
        Writer outW = null;
        try {
            outW = new FileWriter(weightsFile,true);
            outW.write("---------------- Iteration " + m_numIterationsPerformed + " ----------------\r\n");

            for (int i = 0; i < m_extendedInsts.numInstances(); i++) {
                double[] weights = m_extendedInsts.getWeights(i);
                for (int k = 0; k < m_extendedInsts.numClasses(); k++) {
                    DecimalFormat df = new DecimalFormat("#0.0000000000");
                    String weight = df.format(weights[k]);
                    outW.write(weight + "        ");
                }
                outW.write("\r\n");
            }
            outW.write("\r\n\r\n");
            outW.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Returns capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // predictor attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    @OptionMetadata(
            displayName = "I",
            description = "The number of boosting iterations to perform (T in the paper)",
            displayOrder = 1,
            commandLineParamName = "I",
            commandLineParamSynopsis = "I")
    public int getI() {
        return m_I;
    }
    public void setI(int I) {
        this.m_I = I;
    }

    @OptionMetadata(
            displayName = "M",
            description = "The size of product to use for the base classifiers (m in the paper)",
            displayOrder = 2,
            commandLineParamName = "M",
            commandLineParamSynopsis = "-M")
    public int getM() {
        return m_M;
    }
    public void setM(int M) {
        this.m_M = M;
    }

    /**
     * Returns a textual description of the classifier.
     */
    public String toString() {
        return "AdaBoostMH with " + m_I + " iterations and " + m_M + " products";
    }

    public static void main(String[] argv) {
        /*String[] file_arff = {""};
        weka.gui.explorer.Explorer exp = new weka.gui.explorer.Explorer();
        exp.main(file_arff);*/
        runClassifier(new AdaBoostMH(), argv);
    }
}