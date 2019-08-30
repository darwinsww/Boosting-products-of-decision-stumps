package weka.classifiers.meta;

public abstract class BaseLearner {
    protected static double m_smoothingVal = 0;        // Smoothing Value when calculating \alpha and energy.

    // Number of attributes
    protected int m_numAttrs = 0;
    // Number of weights/labels/classes
    protected int m_numClasses = 0;
    // Number of instances
    protected int m_numInstances = 0;

    // Only the following 4 parameters are needed to form the base learner.
    // m_Alpha, m_V, m_selectedAttr, and m_threshold are ultimate parameters to get the best edge and energy when splitting the decision tree.
    protected double m_Alpha = 0;           // This is the \alpha. It is the weight of the current base learner, which is a part of the strong learner.
    protected double[] m_V;                 // This is the vote vector V with length K.
    protected int m_selectedAttr = 0;      // This is the selected attribute used to split the decision tree.
    protected double m_threshold = 0;      // This is the threshold used in function \varphi\left ( x \right )

    // This is the upper bound of the error rate, which we need to minimize.
    // Initialize energy as maximum of double, surely to decrease.
    protected double m_Energy = Double.MAX_VALUE;

    // Print Alpha, Vote Vector, Selected Attribute, Threshold, and Energy of the current learner.
    // If it is a Product Learner, in which all the information of the base learners would be print one by one.
    public abstract void printLearnerInfo ();

    // Build the classifier by using training data.
    public abstract void buildClassifier(ExtendedInstances trainingInsts) throws Exception;

    // Classification for given instance which represented by a double array.
    // It is no need to use "ExtendedInstances" object because we just need all the attributes of instance in the classification.
    // Calculate h_l(x_i) = v_l * \varphi(x_i)
    public abstract double doClassification(double[] testInst, int indexLabel) throws Exception;

    // When operating a bunch of training instances, we need to do as follows:
    // Get the target instance to be classified from the ExtendedInstances object, and transform it into a double array.
    // Then invoke another doClassification function to do the actual classification operation.
    public double doClassification(ExtendedInstances trainingInsts, int indexInstance, int indexLabel) throws Exception {
        return doClassification(trainingInsts.getAttrValues(indexInstance), indexLabel);
    }

    // Use to create an object of the class derived from BaseLearner.
    public abstract BaseLearner subCreate ();

    public void initializeClassifier(ExtendedInstances trainingInsts) throws Exception {
        m_numAttrs = trainingInsts.numAttributes();
        m_numClasses = trainingInsts.numClasses();
        m_numInstances = trainingInsts.numInstances();

        // Use this if statement, because m_V may be initialized in function "BaseLearner.copyState"in advance.
        // "BaseLearner.copyState" is invoked by function "ProductLearner.buildClassifier"
        if (m_V == null) {
            m_V = new double[m_numClasses];       // Should be initialized after the assignment of "m_numClasses".
        }
    }

    public void setSmoothingVal(double smoothingVal) { m_smoothingVal = smoothingVal; }

    // eps_pls is \frac{1+\gamma}{2}, while eps_min is \frac{1-\gamma}{2}
    public double getAlpha(double eps_pls, double eps_min)
    {
        return 0.5 * Math.log((eps_pls + m_smoothingVal) / (eps_min + m_smoothingVal));
    }

    // "getAlpha" should use after function "double getAlpha(double eps_pls, double eps_min)" is invoked in function "buildClassifier".
    public double getAlpha() { return m_Alpha; }
    public double[] getVoteVector () { return m_V; }
    public int getSelectedAttr () { return m_selectedAttr; }
    public double getThreshold () { return m_threshold; }

    public double getEnergy () { return m_Energy; }

    // eps_pls is \frac{1+\gamma}{2}, while eps_min is \frac{1-\gamma}{2}
    public double getEnergy(double eps_pls, double eps_min)
    {
        // In the thesis, energy is also name Z, and it is defined as: Z= \sqrt{1-\gamma^{2}}
        // Here, I used the same formular as C++ implementation in multiboost. It can be derived as follows:
        // Z = \sqrt{1-\gamma^{2}} = 2 * \sqrt{((1+r)/2)*((1-r)/2)}  this is 2 * Math.sqrt(eps_min * eps_pls).
        // Also, (1 - eps_min - eps_pls) actually is (1-(1+r)/2)-((1-r)/2)), which equals to 0.
        // So, the combination of 2 * Math.sqrt(eps_min * eps_pls) and (1 - eps_min - eps_pls) is exactly Z = \sqrt{1-\gamma^{2}}
        return 2 * Math.sqrt(eps_min * eps_pls) + (1 - eps_min - eps_pls);
    }

    // function definition is:
    /* \varphi_{j,b}(x) = \left\{\begin{matrix}
        +1\ if\ x^j\geqslant b\\s
        -1\ if\ x^j<b
       \end{matrix}\right.
    */
    // In the formular above: j refers to the selected attribute to split, b refres to the m_threshold, and x is the
    // j-th attribute value of instance vector x.
    public double phi(double attrValue)
    {
        if (attrValue > m_threshold)
            return +1;
        else
            return -1;
    }

    // Return a new BaseLearner object with same values of data members, used to reserve the status of the invoker.
    public BaseLearner copyState () {
        BaseLearner baseLearner = subCreate();

        baseLearner.m_numAttrs = this.m_numAttrs;
        baseLearner.m_numClasses = this.m_numClasses;
        baseLearner.m_numInstances = this.m_numInstances;
        baseLearner.m_Alpha = this.m_Alpha;
        baseLearner.m_selectedAttr = this.m_selectedAttr;
        baseLearner.m_threshold = this.m_threshold;
        baseLearner.m_Energy = this.m_Energy;

        // In the first loop of product learner, m_V may be null.
        if (this.m_V != null) {
            // We need to use deep copy to reserve the double array of vote vector.
            baseLearner.m_V = this.m_V.clone();
        }

        return baseLearner;
    }
}
