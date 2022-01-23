package moa.classifiers.meta;

import java.util.ArrayList;

import com.yahoo.labs.samoa.instances.Instance;

import moa.core.DoubleVector;

public class DSARF extends AdaptiveRandomForest {
	
	private static final long serialVersionUID = 1L;
	
	@Override
    public String getPurposeString() {
        return "Dynamic Selection in Adaptive Random Forest algorithm for evolving data streams";
    }
	
	public static ArrayList<Integer> slidingwindowarray(ArrayList<Integer> array){
        boolean correctlyClassifies = true;
        int b = correctlyClassifies? 1 : 0;
        array.add(b);
        if(array.size() == 101){
            array.remove(0);
        }  
        return array;
    }
	int numinst = 1;
	@Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        DoubleVector combinedVote = new DoubleVector();

        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                double acc = this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
                if(! this.disableWeightedVote.isSet() && acc > 0.0) {                        
                    for(int v = 0 ; v < vote.numValues() ; ++v) {
                        vote.setValue(v, vote.getValue(v) * acc);
                        
                    }
                }
                combinedVote.addValues(vote);
            }
        }
        numinst++;
        return combinedVote.getArrayRef();
    }
}
