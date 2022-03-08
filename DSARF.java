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
	public static int acertos;
	public static int acuracia;
	public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'z',
            "Size of the sliding window", 100, 1, Integer.MAX_VALUE);
	
	@Override
    public String getPurposeString() {
        return "Dynamic Selection in Adaptive Random Forest algorithm for evolving data streams";
    }
	
	private ExecutorService executor;
	
	@Override
    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        if(this.ensemble == null) 
            initEnsemble(instance);
        Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
        for (int i = 0 ; i < this.ensemble.length ; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            InstanceExample example = new InstanceExample(instance);
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
            if (k > 0) {
                if(this.executor != null) {
                    TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i], 
                        instance, k, this.instancesSeen);
                    trainers.add(trainer);
                }
                else { // SINGLE_THREAD is in-place... 
                    this.ensemble[i].trainOnInstance(instance, k, this.instancesSeen);
                }
            }
        }
        if(this.executor != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }
    }
	
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
        	//this.ensemblearray.get(i);
        	/*
        	try {
        		this.ensemblearray.get(i);
        	}catch(){
        		ArrayList<Integer> array = new ArrayList<Integer>();
        		this.ensemblearray.add(array);
        	}*/
        	
        	//slidingwindow
        	boolean correctlyClassifies = this.ensemble[i].classifier.correctlyClassifies(instance);
        	int b = correctlyClassifies? 1 : 0;
        	this.ensemblearray.get(i).add(b); 
            if(this.ensemblearray.get(i).size() == this.slidingWindowSizeOption.getValue() + 1){	           	
            	this.ensemblearray.get(i).remove(0);	
            }
            for(int j = 0; j < this.ensemblearray.get(i).size(); ++j) {
		    	acertos += this.ensemblearray.get(i).get(j);
		    }
		    float acuracia = (float) acertos/ this.ensemblearray.get(i).size();
		    acertos = 0;  
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