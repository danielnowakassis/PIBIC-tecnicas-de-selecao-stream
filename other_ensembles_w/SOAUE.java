import java.util.ArrayList;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.meta.OnlineAccuracyUpdatedEnsemble;
import moa.core.DoubleVector;

public class SOAUE extends OnlineAccuracyUpdatedEnsemble {
	private static final long serialVersionUID = 1L;
	public ArrayList<ArrayList<Integer>> ensemblearray;
	public static boolean changetobkg = false;
	public static int correctlyclassified;
	public static float accuracy;
	public static int decimal_accuracy;
	public static int classification_threshold = 0;


	public ArrayList<Integer> decimal_accuracy_density;
	public ArrayList<Float> recurrency_control;
	public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'z',
			"Size of the sliding window", 10, 1, Integer.MAX_VALUE);
	public IntOption secondMostRecurrentThreshold = new IntOption("secondMostRecurrentThreshold", 'y',
			"Second most recurrent measure classification threshold", 10, 1, 100);
	public IntOption thirdMostRecurrentThreshold = new IntOption("thirdMostRecurrentThreshold", 'ç',
			"Third most recurrent measure classification threshold", 10, 1, 100);
	
	
	@Override
	public double[] getVotesForInstance(Instance inst) {
		DoubleVector combinedVote = new DoubleVector();

		if (this.trainingWeightSeenByModel > 0.0) {
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
			for (int i = 0; i < this.ensemble.length; i++) {
				if (this.ensemblearray.size() < this.ensemble.length) {
	        		ArrayList<Integer> array = new ArrayList<Integer>();
	        		this.ensemblearray.add(array);
	        	}//slidingwindow -> classifiers accuracy
	        	boolean correctlyClassifies = this.ensemble[i].classifier.correctlyClassifies(inst);
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


	        	int most_recurrent = 0;
	        	int second_recurrent = 0;
	        	int third_recurrent = 0;
	        	int classification_threshold = 0;
	        	
	        	this.recurrency_control.add(accuracy);
	        	int add_decimal = this.decimal_accuracy_density.get(decimal_accuracy) + 1;
	        	this.decimal_accuracy_density.set(decimal_accuracy, add_decimal);

	        	if(this.recurrency_control.size() == this.ensemble.length + 1) {
	        		int index_decimal = (int) (10 * this.recurrency_control.get(0));
	        		int remove_decimal = this.decimal_accuracy_density.get(index_decimal) - 1;
	        		this.decimal_accuracy_density.set(index_decimal, remove_decimal);
	        		this.recurrency_control.remove(0);
	        	}
	        	//most recurring decimal_accuracy			    
	        	for(int k = 0; k < 11; k++) {
	        		int current = this.decimal_accuracy_density.get(k);
	        		if(current > most_recurrent) {
	        			most_recurrent = current;

	        			classification_threshold = k;
	        		}else if(current > second_recurrent && k > classification_threshold ){
	        			second_recurrent = current;
	        			if (second_recurrent >= this.secondMostRecurrentThreshold.getValue()){
	        				classification_threshold = k;
	        			}	
	        		}else if(current >= third_recurrent && k > classification_threshold /*&& (this.third.isSet()*/) {
	        			third_recurrent = current;
	        			if(third_recurrent >= this.secondMostRecurrentThreshold.getValue()) {
	        				classification_threshold = k;
	        			}
	        		}

	        	}
				if (this.weights[i][0] > 0.0) {
					DoubleVector vote = new DoubleVector(this.ensemble[(int) this.weights[i][1]].classifier.getVotesForInstance(inst));

					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						// scale weight and prevent overflow
						vote.scaleValues(this.weights[i][0] / (1.0 * this.ensemble.length + 1.0));
						if(decimal_accuracy >= classification_threshold) {
	    					combinedVote.addValues(vote);}
					}
				}
			}
		}
		
		//combinedVote.normalize();
		return combinedVote.getArrayRef();
	}
}
