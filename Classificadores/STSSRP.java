package moa.classifiers.ensemble_selection;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.LinkedList;

import org.apache.mahout.math.Arrays;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.meta.StreamingRandomPatches;
import moa.classifiers.meta.StreamingRandomPatches.StreamingRandomPatchesClassifier;
import moa.core.DoubleVector;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
public class STSSRP extends StreamingRandomPatches{
	private static final long serialVersionUID = 1L;
	public int correctlyclassified;
	public int classification_threshold = 0;
	public int n_classifiers = 0;
	public int n_classifiers_4 = 0;
	public int correctlyclassified_at = 0;

	public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'z',
			"Size of the sliding window", 10, 1, Integer.MAX_VALUE);
	public IntOption density_control_size = new IntOption("density_control_size", 'ç',
			"Size of the sliding window", 100, 1, Integer.MAX_VALUE);

	LinkedList<LinkedList<Integer>> slidingWindowarray;
	int[] density = new int[this.slidingWindowSizeOption.getValue() + 1];
	int[] last_density = new int[this.slidingWindowSizeOption.getValue() + 1];
	int[] resultados = new int[this.ensembleSizeOption.getValue()];
	public LinkedList<Integer> density_control;
	
	@Override
	public void initEnsemble(Instance instance) {
		// Init the ensemble.
		int ensembleSize = this.ensembleSizeOption.getValue();
		this.ensemble = new StreamingRandomPatchesClassifier[ensembleSize];
		if (this.slidingWindowarray == null) {
			this.slidingWindowarray = new LinkedList<LinkedList<Integer>>();
		}
		if(this.density_control == null) {
			this.density_control = new LinkedList<Integer>();
		}
		BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();

		// #1 Select the size of k, it depends on 2 parameters (subspaceSizeOption and subspaceModeOption).
		int k = this.subspaceSizeOption.getValue();
		if(this.trainingMethodOption.getChosenIndex() != StreamingRandomPatches.TRAIN_RESAMPLING) {
			// PS: This applies only to subspaces and random patches option.
			int n = instance.numAttributes()-1; // Ignore the class label by subtracting 1

			switch(this.subspaceModeOption.getChosenIndex()) {
			case StreamingRandomPatches.FEATURES_SQRT:
				k = (int) Math.round(Math.sqrt(n)) + 1;
				break;
			case StreamingRandomPatches.FEATURES_SQRT_INV:
				k = n - (int) Math.round(Math.sqrt(n) + 1);
				break;
			case StreamingRandomPatches.FEATURES_PERCENT:
				double percent = k < 0 ? (100 + k)/100.0 : k / 100.0;
				k = (int) Math.round(n * percent);

				if(Math.round(n * percent) < 2)
					k = (int) Math.round(n * percent) + 1;
				break;
			}
			// k is negative, use size(features) + -k
			if(k < 0)
				k = n + k;

			// #2 generate the subspaces
			if(this.trainingMethodOption.getChosenIndex() == StreamingRandomPatches.TRAIN_RANDOM_SUBSPACES ||
					this.trainingMethodOption.getChosenIndex() == StreamingRandomPatches.TRAIN_RANDOM_PATCHES) {
				if(k != 0 && k < n) {
					// For low dimensionality it is better to avoid more than 1 classifier with the same subspaces,
					// thus we generate all possible combinations of subsets of features and select without replacement.
					// n is the total number of features and k is the actual size of the subspaces.
					if(n <= 20 || k < 2) {
						if(k == 1 && instance.numAttributes() > 2)
							k = 2;
						// Generate all possible combinations of size k
						this.subspaces = StreamingRandomPatches.allKCombinations(k, n);
						for(int i = 0 ; this.subspaces.size() < this.ensemble.length ; ++i) {
							i = i == this.subspaces.size() ? 0 : i;
							ArrayList<Integer> copiedSubspace = new ArrayList<>(this.subspaces.get(i));
							this.subspaces.add(copiedSubspace);
						}
					}
					// For high dimensionality we can't generate all combinations as it is too expensive (memory).
					// On top of that, the chance of repeating a subspace is lower, so we can just randomly generate
					// subspaces without worrying about repetitions.
					else {
						this.subspaces = StreamingRandomPatches.localRandomKCombinations(k, n,
								this.ensembleSizeOption.getValue(), this.classifierRandom);
					}
				}
				// k == 0 or k > n (subspace size greater than the total number of features), then default to resampling
				else {
					this.trainingMethodOption.setChosenIndex(StreamingRandomPatches.TRAIN_RESAMPLING);
				}
			}
		}

		// Obtain the base learner. It is not restricted to a specific learner.
		Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
		baseLearner.resetLearning();
		for(int i = 0 ; i < ensembleSize ; ++i) {
			switch(this.trainingMethodOption.getChosenIndex()) {
			case StreamingRandomPatches.TRAIN_RESAMPLING:
				this.ensemble[i] = new StreamingRandomPatchesClassifier(
						i,
						baseLearner.copy(),
						(BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(),
						this.instancesSeen,
						this.disableBackgroundLearnerOption.isSet(),
						this.disableDriftDetectionOption.isSet(),
						this.driftDetectionMethodOption,
						this.warningDetectionMethodOption,
						false);
				this.slidingWindowarray.add(new LinkedList<Integer>());
				break;
			case StreamingRandomPatches.TRAIN_RANDOM_SUBSPACES:
			case StreamingRandomPatches.TRAIN_RANDOM_PATCHES:
				int selectedValue = this.classifierRandom.nextInt(subspaces.size());
				ArrayList<Integer> subsetOfFeatures = this.subspaces.get(selectedValue);
				subsetOfFeatures.add(instance.classIndex());
				this.ensemble[i] = new StreamingRandomPatchesClassifier(
						i,
						baseLearner.copy(),
						(BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(),
						this.instancesSeen,
						this.disableBackgroundLearnerOption.isSet(),
						this.disableDriftDetectionOption.isSet(),
						this.driftDetectionMethodOption,
						this.warningDetectionMethodOption,
						subsetOfFeatures,
						instance,
						false);
				this.subspaces.remove(selectedValue);
				this.slidingWindowarray.add(new LinkedList<Integer>());
				break;
			}
		}

	}
	@Override
	public double[] getVotesForInstance(Instance instance) {
		Instance testInstance = instance.copy();
		testInstance.setMissing(instance.classAttribute());
		testInstance.setClassValue(0.0);
		if(this.ensemble == null)
			initEnsemble(testInstance);
		DoubleVector combinedVote = new DoubleVector();
		for(int i = 0 ; i < this.ensemble.length ; ++i) {
			DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
			//learner accuracy
			for(int j = 0; j < this.slidingWindowarray.get(i).size(); ++j) {
				this.correctlyclassified += this.slidingWindowarray.get(i).get(j);
			}
			int correctlyclassified_c = this.correctlyclassified;
			this.correctlyclassified = 0;
			if (vote.sumOfValues() > 0.0) {
				vote.normalize();
				double acc = (10 * correctlyclassified_c) * this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
				if(!this.disableWeightedVote.isSet() && acc > 0.0) {
					for(int v = 0 ; v < vote.numValues() ; ++v) {
						vote.setValue(v, vote.getValue(v) * acc);
					}
				}
				if(correctlyclassified_c >= this.classification_threshold) {
					combinedVote.addValues(vote);
					n_classifiers += 1;
				}
			}
			
			resultados[i] = Utils.maxIndex(vote.getArrayRef());
		}
		//ATUALIZA A DENSIDADE
		for(int j = 0; j < this.ensembleSizeOption.getValue() ; ++j) {
			int correctlyClassifies = resultados[j] == (int) instance.classValue()? 1 : 0; 
			this.slidingWindowarray.get(j).add(correctlyClassifies);
			if(this.slidingWindowarray.get(j).size() == this.slidingWindowSizeOption.getValue() + 1){	           	
				this.slidingWindowarray.get(j).removeFirst();	
			}

			for(int k = 0; k < this.slidingWindowarray.get(j).size(); ++k) {
				correctlyclassified_at += this.slidingWindowarray.get(j).get(k);
			}
			this.density_control.add(correctlyclassified_at);

			this.density[correctlyclassified_at] += 1;
			if(this.density_control.size() == this.density_control_size.getValue() + 1) {

				int index_decimal = this.density_control.get(0);
				this.density[index_decimal] -= 1;
				this.density_control.removeFirst();
			}
			correctlyclassified_at = 0;
		}
		this.classification_threshold = 0;
		for(int r = 6; r < this.density.length; ++r) {
			int current = this.density[r];
			if(current > 0) {
				this.classification_threshold = 5;
				break;
			}
		}
		
		
		n_classifiers_4 = 0;
		n_classifiers = 0;
		

		return combinedVote.getArrayRef();
	}
}
