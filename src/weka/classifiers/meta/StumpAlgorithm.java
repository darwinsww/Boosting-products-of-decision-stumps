package weka.classifiers.meta;

import java.util.ArrayList;
import java.util.Map;

// This is a class to get the best stump in a decision stump.
// Edge is used here. It is a vector with length of "numClasses", and in the thesis it defined as:
// \mbox{edge}:  \gamma = \sum_{l=1}^{k}\sum_{i=1}^{n}w_{i,l}v_{l}\varphi\left ( x_{i} \right )y_{i,l}  \\ (1)
// So, each element in the edge vector is represented as:
// \gamma_{l} = v_{l}\sum_{i=1}^{n}w_{i,l}\varphi\left ( x_{i} \right )y_{i,l} \quad l\in \left \{ 1 .. K \right \}  \\ (2)
// However, in the C++ implementation of multiboost, \gamma_{l} is used as:
// \gamma_{l} = \sum_{i=1}^{n}w_{i,l}y_{i,l} \quad l\in \left \{ 1 .. K \right \}  \\ (3)
// This is because:
// 1. v_{l} is a correction of \gamma_{l} to keep it positive. Detailed discussion is in the following comments.
// 2. \varphi\left ( x_{i} \right ) is used but hided. In the instance array sorted by one attribute, given a fixed position,
// on the right of this postion, all \varphi\left ( x_{i} \right ) = +1;
// on the left of this postion, all \varphi\left ( x_{i} \right ) = -1;
// So, the calculation of function \varphi hided in the formular: \gamma_{l} \leftarrow \gamma_{l} - 2w_{i,l}y_{i,l}  \\
// In summary, the \gamma_{l}(3) had already involved all the w_{i,l},\ \varphi\left ( x_{i} \right ),\ y_{i,l},\ except\ v_{l}.
// So, the \gamma_{l}(3) actually is the one before the correction of v_{l}.
public class StumpAlgorithm {
    // Use to determine whether two double variables are equal.
    protected double m_DoublePrecision = 0.000001;

    // Number of attributes
    protected int m_numAttrs = 0;
    // Number of weights/labels/classes
    protected int m_numClasses = 0;
    // Number of instances
    protected int m_numInstances = 0;

    // Instances to be trained or tested
    protected ExtendedInstances m_trainingInsts;

    // Half value of the total weights per class. The i-th array item represents a sum of weights of the i-th class.
    // \mbox{m\_halfWeightsPerClass[l]} = \frac{1}{2}\sum_{i=1}^{n}w_{i,l} \quad l\in \left \{ 1 .. K \right \}
    protected double[] m_halfWeightsPerClass;

    // Half value of the initial class-wise edges. Each array item is an element of edge(\gamma) vector.
    // \mbox{m\_initHalfEdges[l]}=\frac{1}{2}\gamma_{l}^{(0)} = \frac{1}{2}\sum_{i=1}^{n}w_{i,l}y_{i,l} \quad l\in \left \{ 1 .. K \right \}
    protected double[] m_initHalfEdgesArray;

    // The sum of all the elements in double array "m_initHalfEdges".
    protected double m_initHalfEdge = 0;


    public StumpAlgorithm (ExtendedInstances trainingInsts) {
        m_trainingInsts = trainingInsts;

        m_numAttrs = m_trainingInsts.numAttributes();
        m_numClasses = m_trainingInsts.numClasses();
        m_numInstances = m_trainingInsts.numInstances();

        m_halfWeightsPerClass = new double[m_numClasses];
        m_initHalfEdgesArray = new double[m_numClasses];
    }

    /**
     * Get the best split point of all the instances sorted by an assigned attribute.
     * split point(threshold), vote vector v(V), and best half edge(halfEdge) are acquired and returned to the caller function.
     *
     * @param attrIndex the index of the specific attribute used to find the best split point
     * @param V a vote vector with the length of number of the classes, in which each element only equals to +1 or -1
     *          and is a correction of \gamma_{l} to keep it positive.
     * @param halfEdge half value of the maximum of all the edges we could get in the n+1 position based on the specific attribute.
     * @return threshold: the mean values of the two attributes before and after the best split point.
     */
    public double findBestStumpOfSpecificAttr (int attrIndex, double[] V, double[] halfEdge) {
        // Get the ArrayList of all the instances which are sorted by index-th attribute ascendingly.
        ArrayList<Map.Entry<Integer, Double>> sortedInstances = m_trainingInsts.getSortedAttrsByIndex(attrIndex);

        // The threshold used in function \varphi\left ( x \right )
        double threshold = 0;

        // Half value of the current class-wise edges
        double[] currHalfEdgesArray = new double[m_numClasses];
        // Initialize currHalfEdges to the initial half edges "m_initHalfEdges".
        System.arraycopy(m_initHalfEdgesArray, 0, currHalfEdgesArray, 0, m_numClasses);

        // The position of the best split is between "bestSplitPos" and the one previous
        int bestSplitPos = 0;
        ////////////int bestSplitPos = 1;

        // The sum of all the elements in the best edge vector currently found.
        // Initialize it with threshold = -\infty.
        double bestHalfEdge = m_initHalfEdge;
        ////////////double bestHalfEdge = Double.MIN_VALUE;

        // Half value of the edges of the best found threshold. Each array item is an element of edge (\gamma) vector.
        double[] bestHalfEdgesArray = new double[m_numClasses];
        System.arraycopy(m_initHalfEdgesArray, 0, bestHalfEdgesArray, 0, m_numClasses);

        // Iterate all the sorted instances one by one to get all the edges related to the "n+1" positions;
        // then get best spilt point among the "n+1" edges, which has the biggest value.
        // Here, it is also involved when the situations when split point equals to -\infty or +\infty.
        // So, search the best split point from "n+1" position among "n" instances, -\infty and +\infty.
        // At the first split we have:
        //           index 0   1 2 3 4 5 6 7 8 ..
        //                x | x x x x x x x x ..
        //  previousPos -^   ^- currentPos
        // In the very beginning, the first split point(currentPos) is between instance 1 and 2 (between index 0 and 1).
        // So, in the i-th iteration, the i-th instance had already stepped over (i is in the range [1, m_numInstances]),
        // we need to recalculate the \gammer by subtracting 2w_{i,l}y_{i,l} of the instance which had stepped over.
        // It is notable that \varphi\left ( x_{i} \right ) is hided here:
        // on the right of current position, varphi\left ( x_{i} \right ) = +1;
        // on the left of current position, varphi\left ( x_{i} \right ) = -1;
        for (int currentPos = 1; currentPos <= m_numInstances; currentPos++) {
        ////////////////////////for (int currentPos = 1; currentPos < m_numInstances; currentPos++) {
            // In this iteration, we stepped over the instance with index of "currentPos - 1".
            int previousPos = currentPos - 1;

            // Get the instance stepped over in the ArrayList which is sorted by the "attrIndex"-th attribute.
            int instIndex = sortedInstances.get(previousPos).getKey();

            // Get all the labels of the "instIndex"-th instances.
            int[] labels = m_trainingInsts.getLabels(instIndex);
            // Get all the weights of the "instIndex"-th instances.
            double[] weights = m_trainingInsts.getWeights(instIndex);

            // Recompute half edges at the next point
            for (int l = 0; l < m_numClasses; l++) {
                // Following is the equation: \gamma_{l} \leftarrow \gamma_{l} - 2w_{i,l}y_{i,l} \\ in the paper.
                // However, we use half value of edge, so do not need "2" any more.
                currHalfEdgesArray[l] -= weights[l] * labels[l];
            }

            // Compare the attributes values of current and previous instances. If they are not same,
            // we need to recalculate the \gamma to check whether it is bigger then the current best edge "bestHalfEdge".
            double currentVal = (currentPos == m_numInstances) ? Double.MAX_VALUE : sortedInstances.get(currentPos).getValue();
            /////////////////double currentVal = sortedInstances.get(currentPos).getValue();
            double previousVal = sortedInstances.get(previousPos).getValue();
            if (Math.abs(currentVal - previousVal) > m_DoublePrecision) {   // check whether they are equal.
                double currHalfEdge = 0;          // the sum of all the elements in the current edge vector "halfEdges"

                for (int l = 0; l < m_numClasses; l++) {
                    // \gamma_{l} = v_{l}\left ( \mu_{l^{+}}-\mu_{l^{-}} \right ) = v_{l}\sum_{i=1}^{n}w_{i,l}\varphi\left ( x_{i} \right )y_{i,l}
                    // \gamma_{l} represents the difference between correct rate and error rate related to label l.
                    // The algorithm tries to adjust \gamma_{l} to stay positive by using v_{l}.
                    // That is the reason why \gamma_{l} won't be negative.
                    if ( currHalfEdgesArray[l] > 0 ) {
                        currHalfEdge += currHalfEdgesArray[l];
                    }
                    else {
                        currHalfEdge -= currHalfEdgesArray[l];
                    }
                }

                // check whether the current edge is the new maximum
                // We only allow it happen when ((currHalfEdge - bestHalfEdge) > m_DoublePrecision).
                // if (Math.abs(currHalfEdge - bestHalfEdge) <= m_DoublePrecision) is true,
                // we considered that currHalfEdge equals to bestHalfEdge,
                // under which circumstance, the following if statement should not be executed.
                if ((currHalfEdge - bestHalfEdge) > m_DoublePrecision)
                {
                    bestHalfEdge = currHalfEdge;
                    bestSplitPos = currentPos;

                    for (int l = 0; l < m_numClasses; ++l) {
                        // Fixme: Why, in the C++ implementation of multiboost, they use a minus before currHalfEdges[l]?
                        // That is in StumpAlgorithm.h line 514 : m_bestHalfEdges[l] = -currHalfEdges[l];
                        bestHalfEdgesArray[l] = currHalfEdgesArray[l];
                    }
                }
            }
        }

        // Return the best half edge of current attribute (the attrIndex-th attribute).
        halfEdge[0] = bestHalfEdge;

        // Calculate all the elements of vote vector V.
        for (int l = 0; l < m_numClasses; ++l)
        {
            // use v_{l} to correct \gamma to keep it positive.
            if (bestHalfEdgesArray[l] > 0) {
                V[l] = +1;
            }
            else {
                V[l] = -1;
            }
        }

        if (bestSplitPos == 0) {
            // Here, the best split point is before the first instance.
            threshold = -Double.MAX_VALUE;
        }
        else if (bestSplitPos == m_numInstances) {
            // Here, the best split point is after the last instance.
            threshold = Double.MAX_VALUE;
        }
        else {
            // The threshold is the average of the attributes values on ""bestSplitPos and its previous one.
            threshold = (sortedInstances.get(bestSplitPos).getValue() + sortedInstances.get(bestSplitPos-1).getValue()) / 2.0;
        }/**/
        /////////////////threshold = (sortedInstances.get(bestSplitPos).getValue() + sortedInstances.get(bestSplitPos-1).getValue()) / 2.0;

        return threshold;
    }

    // Calculate the initial element values of gamma/edge (weights edges) vector.
    // Here, however, we calculate half edge, so initial elements of half gamma/edge vector equals to:
    // \mbox{element\ in\ the\ initial\ half\ edge}:  \frac{1}{2}\gamma_{l}^{(0)} = \frac{1}{2}\sum_{i=1}^{n}w_{i,l}y_{i,l} \quad l\in \left \{ 1 .. K \right \}  \\
    public void initHalfEdge () {
        // Iterate all the instances to handle their weights and labels respectively.
        for (int i = 0; i < m_numInstances; i++) {
            // Get all the labels of the i-th instances.
            int[] labels = m_trainingInsts.getLabels(i);
            // Get all the weights of the i-th instances.
            double[] weights = m_trainingInsts.getWeights(i);

            // Iterate all the weights and labels of the current instance.
            for (int l = 0; l < m_numClasses; l++) {
                // Add all the weights respectively into their corresponding classes.
                m_halfWeightsPerClass[l] += weights[l];
                // Calculate the initial value of edges.
                m_initHalfEdgesArray[l] += weights[l] * labels[l];
            }
        }

        // Calculate half edges and half weights
        for (int l = 0; l < m_numClasses; l++)
        {
            m_halfWeightsPerClass[l] /= 2.0;
            m_initHalfEdgesArray[l] /= 2.0;

            m_initHalfEdge += Math.abs(m_initHalfEdgesArray[l]);
        }
    }
}
