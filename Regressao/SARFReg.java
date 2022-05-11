package moa.classifiers.ensemble_selection;

import java.util.LinkedList;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.meta.AdaptiveRandomForestRegressor;
import moa.classifiers.meta.AdaptiveRandomForestRegressor.ARFFIMTDDBaseLearner;
import moa.classifiers.trees.ARFFIMTDD;
import moa.core.DoubleVector;
import moa.evaluation.BasicRegressionPerformanceEvaluator;

public class SARFReg extends AdaptiveRandomForestRegressor{

	private static final long serialVersionUID = 1L;
	LinkedList<LinkedList<Integer>> in_error;
	LinkedList<Double> controle;
	int[] density = new int[11];
	LinkedList<Double> controle_densidade;
	@Override
	protected void initEnsemble(Instance instance) {
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new ARFFIMTDDBaseLearner[ensembleSize];
        
        in_error = new LinkedList<LinkedList<Integer>>();
        controle = new LinkedList<Double>();
        for(int i = 0; i < 100; i++) {
        	in_error.add(new LinkedList<Integer>());
        }
        
        // TODO: this should be an option with default = BasicClassificationPerformanceEvaluator
        BasicRegressionPerformanceEvaluator regressionEvaluator = new BasicRegressionPerformanceEvaluator();

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

        ARFFIMTDD treeLearner = (ARFFIMTDD) getPreparedClassOption(this.treeLearnerOption);
        treeLearner.resetLearning();

        for(int i = 0 ; i < ensembleSize ; ++i) {
            treeLearner.subspaceSizeOption.setValue(this.subspaceSize);
            this.ensemble[i] = new ARFFIMTDDBaseLearner(
                    i,
                    (ARFFIMTDD) treeLearner.copy(),
                    (BasicRegressionPerformanceEvaluator) regressionEvaluator.copy(),
                    this.instancesSeen,
                    ! this.disableBackgroundLearnerOption.isSet(),
                    ! this.disableDriftDetectionOption.isSet(),
                    driftDetectionMethodOption,
                    warningDetectionMethodOption,
                    false);
        }
    }
	
	@Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if(this.ensemble == null)
            initEnsemble(testInstance);

        double sumPredictions = 0;
        double threshold = 0;
        int n_classifiers = 0;
        DoubleVector ages = new DoubleVector();
        DoubleVector performance = new DoubleVector();

        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            double currentPrediction = this.ensemble[i].getVotesForInstance(testInstance)[0];
            System.out.println(Math.abs(currentPrediction - instance.classValue()));
            ages.addToValue(i, this.instancesSeen - this.ensemble[i].createdOn);
            performance.addToValue(i, this.ensemble[i].evaluator.getSquareError());
            controle.add(currentPrediction);
            if(media > threshold) {
            	sumPredictions += currentPrediction;
            	n_classifiers += 1;
            }

        }
        double predicted = sumPredictions / n_classifiers;
        double sumerror = 0;
        for(int j = 0; j < this.ensemble.length; j++) {
        	sumerror += Math.abs(controle.get(j) - instance.classValue());
        }
        double media_erro = sumerror / this.ensemble.length;
        for(int k = 0; k < this.ensemble.length; k++) {
        	
        }
        controle.clear();
        
        return new double[] {predicted};
    }
	
}