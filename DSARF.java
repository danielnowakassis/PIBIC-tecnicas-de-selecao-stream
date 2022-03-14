package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ExecutorService;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.AbstractMOAObject;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.AdaptiveRandomForest.ARFBaseLearner;
import moa.classifiers.meta.AdaptiveRandomForest.TrainingRunnable;
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;
public class DSARF extends AdaptiveRandomForest {
	
	private static final long serialVersionUID = 1L;
	public ArrayList<ArrayList<Integer>> ensemblearray;
	public static boolean changetobkg = false;
	public static int correctlyclassified;
	public static float accuracy;
	public static int decimal_accuracy;
	
	public ArrayList<Integer> decimal_accuracy_density;
	public ArrayList<Float> recurrency_control;
	public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'z',
            "Size of the sliding window", 100, 1, Integer.MAX_VALUE);
	
	@Override
    public String getPurposeString() {
        return "Dynamic Selection in Adaptive Random Forest algorithm for evolving data streams";
    }
	
	@Override
    public double[] getVotesForInstance(Instance instance) {
		Instance testInstance = instance.copy();
		if (changetobkg) {
    		this.ensemblearray.clear();
    		this.decimal_accuracy_density.clear();
    		this.recurrency_control.clear();
            changetobkg = false;
    	}
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        if(this.ensemblearray == null) {
        	this.ensemblearray = new ArrayList<ArrayList<Integer>>();
        }
        if(this.decimal_accuracy_density == null) {
        	this.decimal_accuracy_density = new ArrayList<>();
        	for (int k = 0; k < 11; k++) {
        		this.decimal_accuracy_density.add(0);
        	}
        }
        if(this.recurrency_control == null) {
        	this.recurrency_control = new ArrayList<>();         
        }
                
        DoubleVector combinedVote = new DoubleVector();
        for(int i = 0 ; i < this.ensemble.length ; ++i) {
        	//add ensemble array        	
        	if (this.ensemblearray.size() < this.ensemble.length) {
        		ArrayList<Integer> array = new ArrayList<Integer>();
        		this.ensemblearray.add(array);
        	}
        	
        	
        	//slidingwindow -> classifiers accuracy
        	boolean correctlyClassifies = this.ensemble[i].classifier.correctlyClassifies(instance);
        	int b = correctlyClassifies? 1 : 0;
        	this.ensemblearray.get(i).add(b); 
            if(this.ensemblearray.get(i).size() == this.slidingWindowSizeOption.getValue() + 1){	           	
            	this.ensemblearray.get(i).remove(0);	
            }
            for(int j = 0; j < this.ensemblearray.get(i).size(); ++j) {
            	correctlyclassified += this.ensemblearray.get(i).get(j);
		    }
            accuracy = (float) correctlyclassified/ this.ensemblearray.get(i).size();		    
		    correctlyclassified = 0;
		    decimal_accuracy = (int) (10 * accuracy);
		    
		    
		    //slidingwindow -> threshold
		    int most_recurrent = 0;
		    //int second_recurrent = 0;
		    int classification_threshold = 0;
		    if(this.instancesSeen > 99) {
			    this.recurrency_control.add(accuracy);
			    int add_decimal = this.decimal_accuracy_density.get(decimal_accuracy) + 1;
			    this.decimal_accuracy_density.set(decimal_accuracy, add_decimal);
			    
			    if(this.recurrency_control.size() == 101) {
			    	int index_decimal = (int) (10 * this.recurrency_control.get(0));
			    	int remove_decimal = this.decimal_accuracy_density.get(index_decimal) - 1;
			    	this.decimal_accuracy_density.set(index_decimal, remove_decimal);
			    	this.recurrency_control.remove(0);
			    }
			    //most recurring decimal_accuracy			    
			    for(int k = 0; k < 10; k++) {
			    	int current = this.decimal_accuracy_density.get(k);
			    	if(current > most_recurrent) {
			    		most_recurrent = current;
			    		classification_threshold = k;
			    	}/*else if(current < most_recurrent && current > second_recurrent && k > classification_threshold ){
			    		second_recurrent = current;
			    		if (second_recurrent >= 30){
			    			classification_threshold = k;
			    		}	
			    	}*/
			    	
			    }
		    }
		    //System.out.println(this.decimal_accuracy_density);
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                double acc = this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();      
                if(! this.disableWeightedVote.isSet() && acc > 0.0) {                        
                    for(int v = 0 ; v < vote.numValues() ; ++v) {
                        vote.setValue(v, vote.getValue(v) * acc);
                    }
                } 
                if(decimal_accuracy >= classification_threshold) {
                combinedVote.addValues(vote);}
                
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