package weka.classifiers.meta;

import weka.core.Instances;

import java.util.*;

// This is a Class of all the extended instances by adding several critical complements.
// It contains two critical data members: m_weightedInstances and m_sortedInstances.
// m_sortedInstances: Keeps all the original instances extended by adding wights and multiple labels.
// m_sortedInstances: Keeps all the sorted data by each attribute.
public class ExtendedInstances {
    // Number of attributes
    protected int m_numAttrs = 0;
    // Number of weights/labels/classes
    protected int m_numClasses = 0;
    // Number of instances
    protected int m_numInstances = 0;

    // Use to determine whether two double variables are equal.
    protected double m_DoublePrecision = 0.0001;

    // The ArrayList keeps all the original instances extended by appending wights and multiple labels.
    // Each item in the ArrayList represents an instance with all its attributes, weights and labels.
    protected ArrayList<WeightedMultiLabeledInstance> m_weightedInstances = new ArrayList<>();

    // Keeps all the data sorted by each attribute.
    // The first ArrayList represents all the attributes.
    // The second Arraylist represents all the instances sorted by the attributes above.
    // Map.Entry represents each instances contains two parts: First Integer is index of the instance, Second Double is its value.
    protected ArrayList<ArrayList<Map.Entry<Integer, Double>>> m_sortedInstances;

    // Construction function for all the training data, which is regarded as an Instances object.
    public ExtendedInstances (Instances rawTrainingData) {
        this.m_numAttrs = rawTrainingData.numAttributes() - 1;  // the last value is classification
        this.m_numClasses = rawTrainingData.numClasses();
        this.m_numInstances = rawTrainingData.numInstances();

        this.m_sortedInstances = new ArrayList<>();

        // Create the container of the sorted data, in which each item is an ArrayList.
        for (int j = 0; j < m_numAttrs; j++) {
            m_sortedInstances.add(new ArrayList<>());
        }

        // Iterate all the instances
        for (int i = 0; i < m_numInstances; i++) {
            // Append a weight and a label array on the current instance.
            m_weightedInstances.add(new WeightedMultiLabeledInstance(rawTrainingData.instance(i),
                    m_numAttrs, m_numClasses, m_numInstances));

            // Copy all the key value pairs (index, attribute) of current instance to the sorted array of instances.
            // The j-th attribute of current instance is copied to the j-th sorted array of instances.
            for (int j = 0; j < m_numAttrs; j++) {
                Map<Integer, Double> map = new HashMap<>();
                map.put(i, rawTrainingData.instance(i).value(j));
                m_sortedInstances.get(j).addAll(map.entrySet());
            }
        }

        // Ascendingly sort all the attributes array in m_sortedInstances.
        for (int j = 0; j < m_numAttrs; j++) {
            ValueComparator vc = new ValueComparator();
            ArrayList<Map.Entry<Integer, Double>> attrs = m_sortedInstances.get(j);
            Collections.sort(attrs, vc);
        }

        // Check whether the sum of all the weigths equals to 1
        double weightssum = getSumWeights(m_weightedInstances);
        if (Math.abs(weightssum - 1.0) > m_DoublePrecision) {
            System.err.println("Sum of weights (" + weightssum + ") != 1!");
        }
    }

    // Get the sum of all the weigths
    public double getSumWeights (ArrayList<WeightedMultiLabeledInstance> instances) {
        double weightssum = 0.0;

        for (int i = 0; i < m_numInstances; i++)
            for (int j = 0; j < m_numClasses; j++) {
                weightssum += instances.get(i).getWeights()[j];
        }

        return weightssum;
    }

    // get the attributes array of the index-th instance
    public double[] getAttrValues (int index) {
        return m_weightedInstances.get(index).getAttrValues();
    }

    // get the labels array of the index-th instance
    public int[] getLabels (int index) {
        return m_weightedInstances.get(index).getLabels();
    }

    // get the weights array of the index-th instance
    public double[] getWeights (int index) {
        return m_weightedInstances.get(index).getWeights();
    }

    // get the index-th sorted attributes ArrayList
    public ArrayList<Map.Entry<Integer, Double>> getSortedAttrsByIndex (int index) {
        return m_sortedInstances.get(index);
    }

    // get the number of attributes
    public int numAttributes () { return m_numAttrs; }

    // get the number of labels/weights
    public int numClasses () { return m_numClasses; }

    // get the number of instances
    public int numInstances () { return m_numInstances; }

    // A comparator used to compare the attribute value in the map.
    // The sorting is ascending by the attribute values.
    private class ValueComparator implements Comparator<Map.Entry<Integer, Double>>
    {
        public int compare(Map.Entry<Integer, Double> mp1, Map.Entry<Integer, Double> mp2)
        {
            // Would throw exception if I use: return (int)(mp1.getValue() - mp2.getValue()); here.
            // Would get wrong result if I use: return (mp1.getValue()).toString().compareTo(mp2.getValue().toString());
            return (new Double(mp1.getValue())).compareTo(new Double(mp2.getValue()));
        }
    }
}
