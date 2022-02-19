package moa.classifiers.meta;

import java.util.ArrayList;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.AbstractMOAObject;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.AdaptiveRandomForest.ARFBaseLearner;
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;

public class DSARF extends AdaptiveRandomForest {
	
	private static final long serialVersionUID = 1L;
	public ArrayList<ArrayList<Integer>> ensemblearray;
    ArrayList<Integer> currentarray;
	public static boolean changetobkg = false;
	public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'z',
            "Size of the sliding window", 100, 1, Integer.MAX_VALUE);
	
	@Override
    public String getPurposeString() {
        return "Dynamic Selection in Adaptive Random Forest algorithm for evolving data streams";
    }
	
	/*
	//slidingdelayedwindow
	public static ArrayList<Integer> slidingdelayedwindowarray(ArrayList<Integer> array, boolean correctlyClassifies, int windowsize){
        int b = correctlyClassifies? 1 : 0;
        array.add(b);
        if (array.size() % 50 == 0 && array.size() > 50){
            for (int i = 0; i < 50; i++){
                array.remove(i);
            }
        }
        return array;
    }
    */
	@Override
    public double[] getVotesForInstance(Instance instance) {
		Instance testInstance = instance.copy();
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        if(this.ensemblearray == null) {
        	this.ensemblearray = new ArrayList<ArrayList<Integer>>();
        }
        DoubleVector combinedVote = new DoubleVector();
         
        for(int i = 0 ; i < this.ensemble.length ; ++i) {
        	//add ensemble array
        	if (changetobkg) {
        		this.ensemblearray.clear();
                changetobkg = false;
        	}
        	if (this.ensemblearray.size() < this.ensemble.length) {
        		ArrayList<Integer> array = new ArrayList<Integer>();
        		this.ensemblearray.add(array);
        	}
        	//slidingwindow
        	boolean correctlyClassifies = this.ensemble[i].classifier.correctlyClassifies(instance);
        	this.currentarray = this.ensemblearray.get(i);
        	int b = correctlyClassifies? 1 : 0;
        	this.currentarray.add(b);
            if(currentarray.size() == this.slidingWindowSizeOption.getValue() + 1){	           	
            	currentarray.remove(0);	
            }
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
        return combinedVote.getArrayRef();
    }
	
	protected final class DSARFBaseLearner extends AdaptiveRandomForest.ARFBaseLearner{
		
		@Override
		public void reset() {
            if(this.useBkgLearner && this.bkgLearner != null) {
                this.classifier = this.bkgLearner.classifier;
                changetobkg = true;
                this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;
                
                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;
                this.bkgLearner = null;
            }
	    }
	}
}