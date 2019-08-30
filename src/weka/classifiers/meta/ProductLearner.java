package weka.classifiers.meta;

import java.util.ArrayList;

public class ProductLearner extends BaseLearner{
    // Number of the base learners
    protected int m_numBaseLearners = 3;
    // Name of the base learners
    protected String m_nameBaseLearner = "";

    // Keeps all the base learners, such as Decision Stump.
    protected ArrayList<BaseLearner> m_BaseLearnersInProduct = new ArrayList<>();

    // Keep the original labels, which will be restored after building classifier of Product Learner.
    protected ArrayList<int[]> m_savedLabels = new ArrayList<>();

    protected double m_DoublePrecision = 0.00000001;

    public void setParameters (String nameBaseLearner, int numBaseLearners) {
        m_numBaseLearners = numBaseLearners;
        m_nameBaseLearner = nameBaseLearner;
    }

    // Initialize Product Learner Classifier
    public void initializeClassifier (ExtendedInstances trainingInsts)  throws Exception {
        super.initializeClassifier(trainingInsts);

        // Construct all the base learners
        for (int i = 0; i < m_numBaseLearners; i++) {
            if (m_nameBaseLearner.equals("DecisionStump")) {
                m_BaseLearnersInProduct.add(new SingleStumpLearner());
            }
        }
    }

    // Build the Classifier of Product Learner by using training data.
    public void buildClassifier(ExtendedInstances trainingInsts) throws Exception {
        initializeClassifier(trainingInsts);

        // Backup all the original labels
        backupOriginalLabels (trainingInsts);

        // In the first loop (ecah loop generates m base learners at most), if we find the energy increasing, we just
        // stop the current loop, and use all the previous base learners as parts of the product learner, such as:
        // in the first loop, we have already had 2 base learners, and when we generate the 3rd one, the energy increases,
        // so we just use the 2 base learners as part of current product learner. In the meantime, we update the variable
        // "m_numBaseLearners" to 2 to memorize the number of base learner in the current product learner.
        // On the other hand, if we find the energy increasing in other loops except the first loop, we need to restore
        // the last base learner we got which has a bigger energy to a former one, which is in the same position in the
        // previous loop.
        // So, we use the following variable to keep the base learner which may need to be restored.
        BaseLearner previousBaseLearner;

        boolean firstLoop = true;                 // Used to know whether the current loop is the first loop.
        int ib = -1;
        while (true) {
            ib += 1;
            // cycle through _numBaseLearners base learners as long as the edge increases
            if (ib >= m_numBaseLearners) {
                ib = 0;
                firstLoop = false;
            }

            // In the start of building a new classifier, we reserve the energy and alpha of the previous time.
            double previousEnergy = m_Energy;
            double previousAlpha = m_Alpha;

            // Get the current base learner, such as decision stump.
            BaseLearner curBaseLearner = m_BaseLearnersInProduct.get(ib);

            // Fixme: Without this if statement, smaller m_M may result in lower accuracy.
            // Fixme: Such as pendigits with m_I = 100 and m_M = 1, the accuracy is only around 10%.
            // Fixme: and when m_I = 100 and m_M = 3, the accuracy would be around 49.5%.
            // Fixme: But when the m_M became larger, the accuracy would recover.
            // Fixme: Such as pendigits with m_I = 100 and m_M = 5, the accuracy would be around 88.97%,
            // Fixme: and when m_I = 100 and m_M = 10, the accuracy would be around 94.88%,
            // Fixme: and when m_I = 100 and m_M = 30, the accuracy would be around 94.8%.
            // Fixme: The last accuracy is very similar to the one with the following if statement.
            // Fixme: Don't know wht?
            if ( !firstLoop ) {
                for (int i = 0; i < m_numInstances; i++) {
                    int[] labels = trainingInsts.getLabels(i);

                    // For each label of current instance.
                    for (int k = 0; k < m_numClasses; k++) {
                        if (labels[k] != 0) {
                            // h_l(x_i)=v_l * \varphi(x_i)
                            double hx = curBaseLearner.doClassification(trainingInsts, i, k);
                            if (hx < 0) {
                                labels[k] *= -1;
                            }
                            else if (hx == 0) {
                                labels[k] = 0;
                            }
                        }
                    }
                }
            }

            previousBaseLearner = curBaseLearner.copyState();
            // Use previously generated labels to build a new classifier of the current base learner.
            curBaseLearner.buildClassifier(trainingInsts);

            // Following m_Energy and m_Alpha are the energy and alpha of current product learner,
            // while the right sides of the equations are the energy and alpha of the ib-th base learner,
            // such as "Decision Stump"
            m_Energy = curBaseLearner.getEnergy();
            m_Alpha = curBaseLearner.getAlpha();

            // For each instance. We generate all the labels by using current classifier.
            // The instance with new labels will be used in next time (next while loop) to generate a new classifier.
            for (int i = 0; i < m_numInstances; i++) {
                int[] labels = trainingInsts.getLabels(i);

                // For each label of current instance.
                for (int k = 0; k < m_numClasses; k++) {
                    if (labels[k] != 0) {
                        // h_l(x_i)=v_l * \varphi(x_i)
                        double hx = curBaseLearner.doClassification(trainingInsts, i, k);
                        if (hx < 0) {
                            labels[k] *= -1;
                        }
                        else if (hx == 0) {
                            labels[k] = 0;
                        }
                    }
                }
            }

            // We only allow it happen when (m_Energy > previousEnergy) || (m_Energy == previousEnergy).
            // However, to determine whether two double values are equal, we need to use a precision, as follows:
            // (Math.abs(m_Energy - previousEnergy) <= m_DoublePrecision)
            // if it is true, we considered that m_Energy equals to previousEnergy,
            // under this condition, the following if statement would be executed.
            if ((m_Energy > previousEnergy) || (Math.abs(m_Energy - previousEnergy) <= m_DoublePrecision)) {
                m_Energy = previousEnergy;
                m_Alpha = previousAlpha;

                if (firstLoop) {
                    m_numBaseLearners = ib;
                }
                else {
                    // m_BaseLearnersInProduct.get(ib) is current base learner which has a bigger energy.
                    // We need to restore it to the previous one.
                    m_BaseLearnersInProduct.set(ib, previousBaseLearner);
                }

                break;
            }
        }

        // Restore all the original labels
        restoreOriginalLabels (trainingInsts);
    }

    // Classification for given instance which represented by a double array.
    // It is no need to use "ExtendedInstances" object because we just need all the attributes of instance in the classification.
    // Calculate h_l(x_i) = v_l * \varphi(x_i)
    public double doClassification(double[] testInst, int indexLabel) throws Exception {
        double result = 1;
        for( int ib = 0; ib < m_numBaseLearners; ++ib )
            result *= m_BaseLearnersInProduct.get(ib).doClassification(testInst, indexLabel);
        return result;
    }

    // Backup all the original labels
    private void backupOriginalLabels (ExtendedInstances trainingInsts) {
        for (int i = 0; i < m_numInstances; i++) {
            // A temporary double array to backup all the original labels
            int[] originalLabels = new int[m_numClasses];
            System.arraycopy(trainingInsts.getLabels(i), 0, originalLabels, 0, m_numClasses);
            m_savedLabels.add(originalLabels);
        }
    }

    // Restore all the orginal labels
    private void restoreOriginalLabels (ExtendedInstances trainingInsts) {
        for (int i = 0; i < m_numInstances; i++) {
            System.arraycopy(m_savedLabels.get(i), 0, trainingInsts.getLabels(i), 0, m_numClasses);
        }
    }

    // Return a new ProductLearner object.
    public BaseLearner subCreate () {
        return new ProductLearner();
    }

    // A Product Learner may have m_M base learners at most. In this function, all the information of these learners
    // would be printed, such as: Alpha, Vote Vector, Selected Attribute, Threshold, and Energy .
    public void printLearnerInfo () {
        System.out.println("    Alpha: " + m_Alpha);
        System.out.println("    Amount of Base Learners: " + m_numBaseLearners);

        for (int i = 0; i < m_numBaseLearners; i++) {
            BaseLearner baseLearner = m_BaseLearnersInProduct.get(i);
            System.out.println("    Base Learner: " + (i+1));
            System.out.println("        Alpha: " + baseLearner.getAlpha());

            double [] v = baseLearner.getVoteVector();
            System.out.println("        Vote Vector - size: " + v.length);
            for (int k = 0; k < v.length; k++) {
                System.out.println("            class " + k + " : " + v[k]);
            }

            System.out.println("        Selected Attribute: " + baseLearner.getSelectedAttr());
            System.out.println("        Threshold: " + baseLearner.getThreshold());
            System.out.println("        Energy: " + baseLearner.getEnergy());
        }

        System.out.println("\n");
    }
}
