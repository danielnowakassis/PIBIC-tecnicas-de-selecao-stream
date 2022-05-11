package moa.classifiers.ensemble_selection;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.meta.StreamingRandomPatches;
import moa.core.DoubleVector;

public class SRPm extends StreamingRandomPatches{
	private static final long serialVersionUID = 1L;
	public int correctlyclassified;
	public int classification_threshold = 0;
	public int n_classifiers;
	public int n_classifiers_4 = 0;
	public int correctlyclassified_at = 0;
	public int media = 0;
	public int acertos = 0;
	public int inst_min = 100 / 100;
	
	public ArrayList<ArrayList<Integer>> slidingWindowarray;
	public ArrayList<Integer> density_control;
	public ArrayList<Integer> current_density;
	public ArrayList<Integer> density;
	public ArrayList<Integer> difference;
	public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'z',
			"Size of the sliding window", 10, 1, Integer.MAX_VALUE);
	public IntOption density_control_size = new IntOption("density_control_size", 'ç',
			"Size of the sliding window", 100, 1, Integer.MAX_VALUE);
	public long n_classifiers_total = 0;
	FileWriter fWriter;
	@Override
	public double[] getVotesForInstance(Instance instance) {
		Instance testInstance = instance.copy();
		testInstance.setMissing(instance.classAttribute());
		testInstance.setClassValue(0.0);
		if(this.ensemble == null)
			initEnsemble(testInstance);
		//add ensemble array        	
		if (this.slidingWindowarray == null) {
			this.slidingWindowarray = new ArrayList<ArrayList<Integer>>();
		}
		if(this.density == null) {
			this.density = new ArrayList<>();
			this.current_density = new ArrayList<>();
			this.difference = new ArrayList<>();
			for (int k = 0; k < slidingWindowSizeOption.getValue() + 1; k++) {
				this.density.add(0);
				this.current_density.add(0);
				this.difference.add(0);
			}
		}
		if(this.density_control == null) {
			this.density_control = new ArrayList<Integer>();
		}
		DoubleVector combinedVote = new DoubleVector();

		for(int i = 0 ; i < this.ensemble.length ; ++i) {
			//add ensemble array        	
    		if (this.slidingWindowarray.size() < this.ensemble.length) {
    			ArrayList<Integer> array = new ArrayList<Integer>();
    			this.slidingWindowarray.add(array);
    		}
    		
        	
        	
        	//learner accuracy
        	for(int j = 0; j < this.slidingWindowarray.get(i).size(); ++j) {
				correctlyclassified += this.slidingWindowarray.get(i).get(j);
			}
        	int correctlyclassified_c = correctlyclassified;
			correctlyclassified = 0;
			
			DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
			if (vote.sumOfValues() > 0.0) {
				vote.normalize();
				double acc = this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
				if(!this.disableWeightedVote.isSet() && acc > 0.0) {
					for(int v = 0 ; v < vote.numValues() ; ++v) {
						vote.setValue(v, vote.getValue(v) * acc);
					}
				}
				 //selecao classificadores
                if(correctlyclassified_c >= classification_threshold) {
                	combinedVote.addValues(vote);
                	n_classifiers += 1;}
			}
			if(this.instancesSeen > inst_min && i == 0) {
				for(int l = 0; l < this.density_control_size.getValue(); l++) {
					acertos += this.density_control.get(l);
				}
				media = acertos/ this.density_control_size.getValue();
				acertos = 0;
			} 
			//atualiza modelo janela deslizante
			boolean correctlyClassifies = this.ensemble[i].classifier.correctlyClassifies(instance);
			int b = correctlyClassifies? 1 : 0;
			this.slidingWindowarray.get(i).add(b); 
			if(this.slidingWindowarray.get(i).size() == this.slidingWindowSizeOption.getValue() + 1){	           	
				this.slidingWindowarray.get(i).remove(0);	
			}
			//atualiza modelo controle densidade
			//learner accuracy
			
        	for(int j = 0; j < this.slidingWindowarray.get(i).size(); ++j) {
				correctlyclassified_at += this.slidingWindowarray.get(i).get(j);
			}
				    
			
			
			
			this.density_control.add(correctlyclassified_at);
			int add_decimal = this.density.get(correctlyclassified_at) + 1;
			this.density.set(correctlyclassified_at, add_decimal);
			if(this.density_control.size() == this.density_control_size.getValue() + 1) {
				int index_decimal = this.density_control.get(0);
				int remove_decimal = this.density.get(index_decimal) - 1;
				this.density.set(index_decimal, remove_decimal);
				this.density_control.remove(0);
			}
			correctlyclassified_at = 0;
		}
		n_classifiers = 0;
		return combinedVote.getArrayRef();
	}

}

