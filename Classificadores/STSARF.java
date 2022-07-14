package moa.classifiers.ensemble_selection;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.ExecutorService;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import cern.colt.Arrays;
import moa.AbstractMOAObject;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.meta.AdaptiveRandomForest.ARFBaseLearner;
import moa.classifiers.meta.AdaptiveRandomForest.TrainingRunnable;
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.options.ClassOption;
public class STSARF extends AdaptiveRandomForest {
	private static final long serialVersionUID = 1L;
	public int correctlyclassified;
	public int classification_threshold = 0;
	public int n_classifiers = 0;
	public int n_classifiers_4 = 0;
	public int correctlyclassified_at = 0;


	public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'z',
			"Size of the sliding window", 10, 1, Integer.MAX_VALUE);
	public IntOption density_control_size = new IntOption("density_control_size", 'ç',
			"Size of the sliding window", 300, 1, Integer.MAX_VALUE);
	public FlagOption selectionClassifiers = new FlagOption("selectionClassifiers", 't',
			"Should use selection of classifiers?");
	//public FlagOption weightOption = new FlagOption("weightOption", 't',
	//		"Should use selection of classifiers?");


	LinkedList<LinkedList<Integer>> slidingWindowarray;
	int[] density = new int[this.slidingWindowSizeOption.getValue() + 1];
	int[] resultados = new int[this.ensembleSizeOption.getValue()];
	public LinkedList<Integer> density_control;

	//INIT ENSEMBLE + DENSIDADE
	@Override
	public void initEnsemble(Instance instance) {
		// Init the ensemble.
		int ensembleSize = this.ensembleSizeOption.getValue();
		this.ensemble = new ARFBaseLearner[ensembleSize];

		// TODO: this should be an option with default = BasicClassificationPerformanceEvaluator
		//        BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
		BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();

		this.subspaceSize = this.mFeaturesPerTreeSizeOption.getValue();

		if (this.slidingWindowarray == null) {
			this.slidingWindowarray = new LinkedList<LinkedList<Integer>>();
		}
		if(this.density_control == null) {
			this.density_control = new LinkedList<Integer>();
		}


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
			this.slidingWindowarray.add(new LinkedList<Integer>());
		}
	}
	@Override
	public void trainOnInstanceImpl(Instance instance) {
		++this.instancesSeen;
		if(this.ensemble == null) 
			initEnsemble(instance);
		Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
		for (int i = 0 ; i < this.ensemble.length ; i++) {
			int correctlyClassifies = this.resultados[i] == (int) instance.classValue()? 1 : 0; 
			this.slidingWindowarray.get(i).add(correctlyClassifies);
			if(this.slidingWindowarray.get(i).size() == this.slidingWindowSizeOption.getValue() + 1){	           	
				this.slidingWindowarray.get(i).removeFirst();	
			}
			for(int j = 0; j < this.slidingWindowarray.get(i).size(); ++j) {
				correctlyclassified_at += this.slidingWindowarray.get(i).get(j);
			}
			this.density_control.add(correctlyclassified_at);
			this.density[correctlyclassified_at] += 1;
			if(this.density_control.size() == this.density_control_size.getValue() + 1) {

				int index_decimal = this.density_control.get(0);
				this.density[index_decimal] -= 1;
				this.density_control.removeFirst();
			}
			correctlyclassified_at = 0;
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
		if(selectionClassifiers.isSet()) {
			// cálculo do limiar de classificação 
			int most_recurrent = 0;
			for(int r = 5; r < this.density.length; ++r) {
				int current = this.density[r];
				n_classifiers_4 += current;
				if(current >= most_recurrent) {
					this.classification_threshold = r;
					most_recurrent = current;
				}
			}

			if(n_classifiers_4 < 20) {
				this.classification_threshold = 0;
			}
			if(this.classification_threshold == 10) {
				this.classification_threshold = 9;
			}
			this.n_classifiers_4 = 0;
			this.n_classifiers = 0;
		}
	}


	@Override
	public double[] getVotesForInstance(Instance instance) {
		Instance testInstance = instance.copy();
		if(this.ensemble == null) 
			initEnsemble(testInstance);
		DoubleVector combinedVote = new DoubleVector();

		for(int i = 0 ; i < this.ensemble.length ; ++i) {

			//learner accuracy
			for(int j = 0; j < this.slidingWindowarray.get(i).size(); ++j) {
				this.correctlyclassified += this.slidingWindowarray.get(i).get(j);
			}
			int correctlyclassified_c = this.correctlyclassified;
			this.correctlyclassified = 0;


			DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
			if (vote.sumOfValues() > 0.0) {
				vote.normalize();
				
				double acc = (correctlyclassified_c * 10 ) * this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
				if(! this.disableWeightedVote.isSet() && acc > 0.0) {                        
					for(int v = 0 ; v < vote.numValues() ; ++v) {
						vote.setValue(v, vote.getValue(v) * acc);
					}
				}
				if(selectionClassifiers.isSet()) {
					//selecao classificadores
					if(correctlyclassified_c >= this.classification_threshold) {
						combinedVote.addValues(vote);
						this.n_classifiers += 1;
					}
				}else {
					combinedVote.addValues(vote);
				}
			}
			resultados[i] = Utils.maxIndex(vote.getArrayRef());
		}
		return combinedVote.getArrayRef();
	}
}
