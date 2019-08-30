package weka.classifiers.meta;

public class SingleStumpLearner extends BaseLearner {
    // An algorithm to get the best stump of decision stump
    protected StumpAlgorithm m_stumpAlgo;

    // Use to determine whether two double variables are equal.
    protected double m_DoublePrecision = 0.00000001;

    // Initialize Decision Stump Classifier
    public void initializeClassifier (ExtendedInstances trainingInsts)  throws Exception {
        super.initializeClassifier(trainingInsts);

        m_stumpAlgo = new StumpAlgorithm(trainingInsts);
    }

    // Build the Classifier of Decision Stump by using training data.
    public void buildClassifier(ExtendedInstances trainingInsts) throws Exception {
        initializeClassifier(trainingInsts);

        // Set the smoothing value when computing alpha
        setSmoothingVal( 1.0 / m_numInstances * 0.01);

        // Calculate the initial values of half gamma/edge (that is weights edges).
        m_stumpAlgo.initHalfEdge();

        // To keep the best energy currently found.
        // We need to minimize the energy, so we initialize it with the maximum of double.
        double bestEnergy = Double.MAX_VALUE;

        // Iterate all the sorting possibilities of the instances, which are sorted by each attribute respectively, to
        // find a best stump of each iteration. Then find the best of best stumps from these results.
        for (int j = 0; j < m_numAttrs; j++) {
            double[] tmpV = new double[m_numClasses];        // This is temp vote vector V.
            double[] tmphalfEdge = new double[1];             // This is temp \gamma, also named edge, which we need to maximize.

            // Get the best split point of all the instances sorted by an i-th attribute.
            // Acquire the temp threshold (tmpThreshold), temp vote vector (tmpV), and temp half edge (tmphalfEdge).
            double tmpThreshold = m_stumpAlgo.findBestStumpOfSpecificAttr(j, tmpV, tmphalfEdge);

            // Get temp \alpha (tmpAlpha) and temp energy (tmpEnergy).
            double eps_pls = 0.5 + tmphalfEdge[0];             // This is \frac{1+\gamma}{2}, tmphalfEdge had already divided by 2.
            double eps_min = 0.5 - tmphalfEdge[0];             // This is \frac{1-\gamma}{2}, tmphalfEdge had already divided by 2.
            double tmpAlpha = getAlpha(eps_pls, eps_min);      // This is temp \alpha
            double tmpEnergy = getEnergy(eps_pls, eps_min);    // This is temp Z, also named energy, which we need to minimize

            // We only allow it happen when ((bestEnergy - tmpEnergy) > m_DoublePrecision).
            // If (Math.abs(bestEnergy - tmpEnergy) <= m_DoublePrecision) is true,
            // we considered that tmpEnergy equals to bestEnergy,
            // under which circumstance, the following if statement should not be executed.
            if ((bestEnergy - tmpEnergy) > m_DoublePrecision)
            {
                m_Alpha = tmpAlpha;
                System.arraycopy(tmpV, 0, m_V, 0, m_numClasses);
                m_selectedAttr = j;
                m_threshold = tmpThreshold;

                bestEnergy = tmpEnergy;
            }
        }

        // Just to use to select the minimum in the Product Learner.
        m_Energy = bestEnergy;
    }

    // Classification for given instance which represented by a double array.
    // It is no need to use "ExtendedInstances" object because we just need all the attributes of instance in the classification.
    // Calculate h_l(x_i) = v_l * \varphi(x_i)
    public double doClassification(double[] testInst, int indexLabel) throws Exception {
        double v_l = m_V[indexLabel];     // v_l

        // \varphi(x_i)
        double attrValue = testInst[m_selectedAttr];
        double retPhi = phi(attrValue);

        return v_l * retPhi;
    }

    // Return a new SingleStumpLearner object.
    public BaseLearner subCreate () {
        return new SingleStumpLearner();
    }

    // Print Alpha, Vote Vector, Selected Attribute, Threshold, and Energy of the current learner.
    public void printLearnerInfo () {
        System.out.println("Alpha: " + m_Alpha);

        double [] v = getVoteVector();
        System.out.println("Vote Vector - size: " + v.length);
        for (int k = 0; k < v.length; k++) {
            System.out.println("    class " + k + " : " + v[k]);
        }

        System.out.println("Selected Attribute: " + m_selectedAttr);
        System.out.println("Threshold: " + m_threshold);
        System.out.println("Energy: " + m_Energy);

        System.out.println("\n");
    }
}
