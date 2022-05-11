package moa.classifiers.ensemble_selection;



import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.meta.AdaptiveRandomForestRegressor;
import moa.classifiers.meta.LeveragingBag;
import moa.classifiers.meta.OnlineAccuracyUpdatedEnsemble;
import moa.classifiers.meta.OnlineSmoothBoost;
import moa.classifiers.meta.OzaBag;
import moa.classifiers.meta.OzaBoost;
import moa.classifiers.meta.StreamingRandomPatches;
import moa.classifiers.trees.FIMTDD;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.tasks.EvaluatePrequential;
import moa.tasks.EvaluatePrequentialCV;
import moa.tasks.EvaluatePrequentialDelayedCV;
import moa.tasks.EvaluatePrequentialRegression;

public class MainRegression {

	public static void main(String[] args) { 
		FileWriter fWriter;
		AdaptiveRandomForestRegressor learner = new AdaptiveRandomForestRegressor();
		//FIMTDD learner = new FIMTDD();
		//SARFReg learner = new SARFReg();
		

	    
		//learner.slidingWindowSizeOption.setValue(10);
		//learner.density_control_size.setValue(500);
		ArffFileStream stream = new ArffFileStream("path_basededadosB", -1);
		stream.prepareForUse(); 
		BasicRegressionPerformanceEvaluator evaluator = new BasicRegressionPerformanceEvaluator();

		//learner.numberOfJobsOption.setValue(-1);
		
		//learner.baseLearnerOption.setValueViaCLIString("trees.EFDT");

		//learner.ensembleSizeOption.setValue(100);
		//learner.memberCountOption.setValue(100);
	
		
		EvaluatePrequentialRegression task = new EvaluatePrequentialRegression();
		task.sampleFrequencyOption.setValue(1000000000);

		//task.sampleFrequencyOption.setValue(2000);
		
		task.learnerOption.setCurrentObject(learner);
		task.streamOption.setCurrentObject(stream);
		task.evaluatorOption.setCurrentObject(evaluator); 
		task.prepareForUse();
		LearningCurve le = (LearningCurve) task.doTask();
		System.out.println(le.getMeasurementName(4));
		System.out.println(le.getMeasurement(0, 4)); 
		System.out.println(le.getMeasurementName(3));
		System.out.println(le.getMeasurement(0, 3)); 
		//System.out.println(learner.n_classifiers_total);//imprime resultados
		System.out.println(le.getMeasurementName(1));
		System.out.println(le.getMeasurement(0, 1)); 
		try {
			fWriter = new FileWriter("path_arquivo", true);
			BufferedWriter  bw = new BufferedWriter(fWriter);
			PrintWriter pw = new PrintWriter(fWriter);
			pw.println(le.getMeasurementName(4));
			bw.newLine();
			pw.println(le.getMeasurement(0, 4)); 
			bw.newLine();
			pw.println(le.getMeasurementName(3));
			bw.newLine();
			pw.println(le.getMeasurement(0, 3)); 
			bw.newLine();
			pw.println(le.getMeasurementName(1));
			bw.newLine();
			pw.println(le.getMeasurement(0, 1)); 
			bw.newLine(); 
			bw.close();
			pw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}  
} 