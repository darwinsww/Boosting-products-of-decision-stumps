package weka.classifiers.meta;

import weka.core.Attribute;
import weka.core.Instance;

// This is a Class of one instance x:
// It contains: attributes[m_NumAttrs] array, labels[m_NumClasses] array, weights[m_NumClasses] array of this instance x
public class WeightedMultiLabeledInstance {
    protected double[] m_AttrValues;
    protected int[] m_Labels;
    protected double[] m_Weights;

    public WeightedMultiLabeledInstance(Instance data, int numAttrs, int numClasses, int numInsts) {
        this.m_AttrValues = new double[numAttrs];
        this.m_Labels = new int[numClasses];
        this.m_Weights = new double[numClasses];

        this.initAttributes(data, numAttrs);
        this.initLabels(data, numClasses);              // must use it before function initWeights
        this.initWeights(data, numClasses, numInsts);
    }

    // copy all the attributes expect the class value to the attribute array
    private void initAttributes (Instance data, int numAttrs) {
        System.arraycopy(data.toDoubleArray(), 0, m_AttrValues, 0, numAttrs);
    }

    // initialize all the labels, set them +1 or -1, according to the actual class value of this instance
    private void initLabels (Instance data, int numClasses) {
        // initialize all the labels to -1
        for (int i = 0; i < numClasses; i++) {
            m_Labels[i] = -1;
        }

        // get all the labels (in a label value array) of this classification
        Attribute allLabels = data.classAttribute();

        // get the label which current instance belongs to
        String classValue = data.stringValue(allLabels);

        // get the index of the label in the label value array
        int index = allLabels.indexOfValue(classValue);

        // set the label to 1, means it is the classification the current instance belongs to
        m_Labels[index] = 1;
    }

    // initialize all the weights according to the labels:
    // if y_l[x_i] = +1, then w_l[x_i] = 1/(2*n)
    // if y_l[x_i] = -1, then w_l[x_i] = 1/(2*n*(K-1))
    // n is the number of instances, K is the number of classes.
    private void initWeights (Instance data, int numClasses, int numInsts) {
        for (int i = 0; i < numClasses; i++) {
            if (m_Labels[i] == 1) {
                m_Weights[i] = (double)1 / (2 * numInsts);
            }
            else if (m_Labels[i] == -1) {
                m_Weights[i] = (double)1 / (2 * numInsts * (numClasses - 1));
            }
            else {
                // should never happen
                assert (1 != 1);
            }
        }
    }

    // get the attributes array of the current instance
    public double[] getAttrValues () { return m_AttrValues; }

    // get the labels array of the current instance
    public int[] getLabels () { return m_Labels; }

    // get the weights array of the current instance
    public double[] getWeights () { return m_Weights; }
}

