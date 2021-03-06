package moa.classifiers.ensemble_selection;



import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import moa.classifiers.meta.AdaptiveRandomForest;
import moa.classifiers.meta.LeveragingBag;
import moa.classifiers.meta.OnlineAccuracyUpdatedEnsemble;
import moa.classifiers.meta.OnlineSmoothBoost;
import moa.classifiers.meta.OzaBag;
import moa.classifiers.meta.OzaBoost;
import moa.classifiers.meta.StreamingRandomPatches;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.tasks.EvaluatePrequential;
import moa.tasks.EvaluatePrequentialCV;
import moa.tasks.EvaluatePrequentialDelayedCV;

public class Main {

	public static void main(String[] args) { 
		FileWriter fWriter;
		//MediaSRP learner = new MediaSRP();
		//AdaptiveRandomForest learner = new AdaptiveRandomForest();
		//CertoArf learner = new CertoArf();
		//MediaArf learner = new MediaArf();
		//WARF learner = new WARF();
		//SSRP learner = new SSRP();
		StreamingRandomPatches learner = new StreamingRandomPatches();
		//MediaSRP learner = new MediaSRP();
		//SSRP learner = new SSRP();
		//SAdaptiveRandomForest learner = new SAdaptiveRandomForest();
		//CertoOzaBag learner = new CertoOzaBag();
		//SOzaBag learner = new SOzaBag();
		//OzaBag learner = new OzaBag();
		//LeveragingBag learner = new LeveragingBag();
		//SLeveragingBag learner = new SLeveragingBag();
		//OnlineAccuracyUpdatedEnsemble learner = new OnlineAccuracyUpdatedEnsemble();
		//SOAUE learner = new SOAUE();
		//OzaBoost learner = new OzaBoost();
		//StreamingRandomPatches learner = new StreamingRandomPatches();
		
		//SOzaBoost learner = new SOzaBoost();
		//OnlineSmoothBoost learner = new OnlineSmoothBoost();
		//SOSB learner = new SOSB();
		//SOnlineSmoothBoost learner = new SOnlineSmoothBoost();
		
		

	    
		//learner.slidingWindowSizeOption.setValue(10);
		//learner.density_control_size.setValue(500);
		ArffFileStream stream = new ArffFileStream("path_base_de_dados", -1);
		stream.prepareForUse(); 
		BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();

		//learner.numberOfJobsOption.setValue(-1);
		
		//learner.baseLearnerOption.setValueViaCLIString("trees.EFDT");

		//learner.ensembleSizeOption.setValue(100);
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