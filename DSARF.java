package moa.classifiers.meta;

import java.util.ArrayList;

import com.yahoo.labs.samoa.instances.Instance;

import moa.AbstractMOAObject;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.AdaptiveRandomForest.ARFBaseLearner;
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.core.DoubleVector;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

public class DSARF extends AdaptiveRandomForest {
	
	private static final long serialVersionUID = 1L;
	protected ArrayList<ArrayList<Integer>> ensemblearray;
	
	@Override
    public String getPurposeString() {
        return "Dynamic Selection in Adaptive Random Forest algorithm for evolving data streams";
    }
	//slidingwindow
	public static ArrayList<Integer> slidingwindowarray(ArrayList<Integer> array){
        boolean correctlyClassifies = true;
        int b = correctlyClassifies? 1 : 0;
        array.add(b);
        if(array.size() == 101){
            array.remove(0);
        }  
        return array;
    }
	//slidingdelayedwindow
	public static ArrayList<Integer> slidingdelayedwindowarray(ArrayList<Integer> array){
        boolean correctlyClassifies = true;
        int b = correctlyClassifies? 1 : 0;
        array.add(b);
        if (array.size() % 50 == 0 && array.size() > 50){
            for (int i = 0; i < 50; i++){
                array.remove(i);
            }
        }
        return array;
    }
	
	
	//inicialização dos arrays junto com a ensemble
	@Override
	protected void initEnsemble(Instance instance) {
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new ARFBaseLearner[ensembleSize];
        this.ensemblearray = new ArrayList<ArrayList<Integer>>();//inicialização dos arrays
        
        // TODO: this should be an option with default = BasicClassificationPerformanceEvaluator
//        BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();
        
        this.subspaceSize = this.mFeaturesPerTreeSizeOption.getValue();
  
        // The size of m depends on:
        // 1) mFeaturesPerTreeSizeOption
        // 2) mFeaturesModeOption
        int n = instance.numAttributes()-1; // Ignore class label ( -1 )
        
        switch(this.mFeaturesModeOption.getChosenIndex()) {
            case AdaptiveRandomForest.FEATURES_SQRT:
                this.subspaceSize = (int) Math.round(Math.sqrt(n)) + 1;
                break;
            case AdaptiveRandomForest.FEATURES_SQRT_INV:
                this.subspaceSize = n - (int) Math.round(Math.sqrt(n) + 1);
                break;
            case AdaptiveRandomForest.FEATURES_PERCENT:
                // If subspaceSize is negative, then first find out the actual percent, i.e., 100% - m.
                double percent = this.subspaceSize < 0 ? (100 + this.subspaceSize)/100.0 : this.subspaceSize / 100.0;
                this.subspaceSize = (int) Math.round(n * percent);
                break;
        }
        // Notice that if the selected mFeaturesModeOption was 
        //  AdaptiveRandomForest.FEATURES_M then nothing is performed in the
        //  previous switch-case, still it is necessary to check (and adjusted) 
        //  for when a negative value was used. 
        
        // m is negative, use size(features) + -m
        if(this.subspaceSize < 0)
            this.subspaceSize = n + this.subspaceSize;
        // Other sanity checks to avoid runtime errors. 
        //  m <= 0 (m can be negative if this.subspace was negative and 
        //  abs(m) > n), then use m = 1
        if(this.subspaceSize <= 0)
            this.subspaceSize = 1;
        // m > n, then it should use n
        if(this.subspaceSize > n)
            this.subspaceSize = n;
        
        ARFHoeffdingTree treeLearner = (ARFHoeffdingTree) getPreparedClassOption(this.treeLearnerOption);
        treeLearner.resetLearning();
        
        for(int i = 0 ; i < ensembleSize ; ++i) {
            treeLearner.subspaceSizeOption.setValue(this.subspaceSize);
            this.ensemble[i] = new ARFBaseLearner(
                i, 
                (ARFHoeffdingTree) treeLearner.copy(), 
                (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), 
                this.instancesSeen, 
                ! this.disableBackgroundLearnerOption.isSet(),
                ! this.disableDriftDetectionOption.isSet(), 
                driftDetectionMethodOption,
                warningDetectionMethodOption,
                false);
            this.ensemblearray.add(new ArrayList<Integer>());//adiciona novo array
        }
    }
	@Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        DoubleVector combinedVote = new DoubleVector();
        
        for(int i = 0 ; i < this.ensemble.length ; ++i) {
        	
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
            this.ensemblearray.get(i).add();
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
}



