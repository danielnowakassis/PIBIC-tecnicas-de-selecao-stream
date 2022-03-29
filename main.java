package moa.classifiers.ensemble_selection;



import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.meta.LeveragingBag;
import moa.classifiers.meta.OnlineAccuracyUpdatedEnsemble;
import moa.classifiers.meta.OnlineSmoothBoost;
import moa.classifiers.meta.OzaBag;
import moa.classifiers.meta.OzaBoost;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.tasks.EvaluatePrequential;

public class Main {

	public static void main(String[] args) { 
		AdaptiveRandomForest learner = new AdaptiveRandomForest();
		//SAdaptiveRandomForest learner = new SAdaptiveRandomForest();
		//OzaBag learner = new OzaBag();
		//SOzaBag learner = new SOzaBag();
		//LeveragingBag learner = new LeveragingBag();
		//SLeveragingBag learner = new SLeveragingBag();
		//OnlineAccuracyUpdatedEnsemble learner = new OnlineAccuracyUpdatedEnsemble();
		//SOAUE learner = new SOAUE();
		//OzaBoost learner = new OzaBoost();
		//SOzaBoost learner = new SOzaBoost();
		//OnlineSmoothBoost learner = new OnlineSmoothBoost();
		//SOnlineSmoothBoost learner = new SOnlineSmoothBoost();
		
		

	    
	    

		ArffFileStream stream = new ArffFileStream("C:/Users/daniel/Desktop/arffReal/ozone.arff", -1);
		stream.prepareForUse(); 
		BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();

		//learner.numberOfJobsOption.setValue(-1);
		/*
		//learner.baseLearnerOption.setValueViaCLIString("trees.HoeffdingTree");
		learner.slidingWindowSizeOption.setValue(1);
		learner.secondMostRecurrentThreshold.setValue(1);
		learner.thirdMostRecurrentThreshold.setValue(1);*/
		learner.ensembleSizeOption.setValue(100);
		//learner.memberCountOption.setValue(100);
	
		
		EvaluatePrequential task = new EvaluatePrequential();
		task.sampleFrequencyOption.setValue(1000000000);

		//task.sampleFrequencyOption.setValue(2000);

		task.learnerOption.setCurrentObject(learner);
		task.streamOption.setCurrentObject(stream);
		task.evaluatorOption.setCurrentObject(evaluator); 
		task.prepareForUse();
		LearningCurve le = (LearningCurve) task.doTask();
		System.out.println(le.getMeasurementName(4));
		System.out.println(le.getMeasurement(0, 4)); //imprime resultados


	}
} 