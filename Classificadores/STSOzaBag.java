package moa.classifiers.ensemble_selection;

import java.util.LinkedList;

import org.apache.mahout.math.Arrays;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.classifiers.meta.OzaBag;
import moa.core.DoubleVector;
import moa.core.Utils;

public class STSOzaBag extends OzaBag {
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
	int[] resultados = new int[100];
	public LinkedList<Integer> density_control;

	@Override
	public void resetLearningImpl() {
		this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
		Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
		if (this.slidingWindowarray == null) {
			this.slidingWindowarray = new LinkedList<LinkedList<Integer>>();
		}
		if(this.density_control == null) {
			this.density_control = new LinkedList<Integer>();
		}
		baseLearner.resetLearning();
		for (int i = 0; i < this.ensemble.length; i++) {
			this.ensemble[i] = baseLearner.copy();
			this.slidingWindowarray.add(new LinkedList<Integer>());
		}
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		DoubleVector combinedVote = new DoubleVector();
		for (int i = 0; i < this.ensemble.length; i++) {
			//learner accuracy
			for(int j = 0; j < this.slidingWindowarray.get(i).size(); ++j) {
				this.correctlyclassified += this.slidingWindowarray.get(i).get(j);
			}
			int correctlyclassified_c = this.correctlyclassified;
			this.correctlyclassified = 0;
			DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
			if (vote.sumOfValues() > 0.0) {
				vote.normalize();
				if(correctlyclassified_c >= classification_threshold) {
					combinedVote.addValues(vote);
					n_classifiers += 1;
				}
			}
			resultados[i] = Utils.maxIndex(vote.getArrayRef());
		}
		//ATUALIZA A DENSIDADE
		for(int j = 0; j < resultados.length; ++j) {
			int correctlyClassifies = resultados[j] == (int) inst.classValue()? 1 : 0; 
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
		int most_recurrent = 0;
		for(int r = 0; r < this.density.length; ++r) {
			int current = this.density[r];
			if(current >= most_recurrent) {
				this.classification_threshold = r;
				most_recurrent = current;
			}
		}
		return combinedVote.getArrayRef();
	}
}
