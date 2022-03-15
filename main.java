package moa.classifiers.meta;

import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.tasks.EvaluatePrequential;

public class main {

public static void main(String[] args) { 
        
        
        //DSARF  learner= new DSARF();
        AdaptiveRandomForest learner = new AdaptiveRandomForest();
        
        ArffFileStream stream = new ArffFileStream("C:/Users/daniel/Desktop/arffReal/SpamAssassin.arff", -1);
        stream.prepareForUse() ; 
        BasicClassificationPerformanceEvaluator evaluator = new BasicClassificationPerformanceEvaluator();
        
        learner.numberOfJobsOption.setValue(-1);
        //learner.slidingWindowSizeOption.setValue(10);
        EvaluatePrequential task = new EvaluatePrequential();
        task.sampleFrequencyOption.setValue(1000000000);
        
        //task.sampleFrequencyOption.setValue(2000);
        
        task.learnerOption.setCurrentObject(learner);
        task.streamOption.setCurrentObject(stream);
        task.evaluatorOption.setCurrentObject(evaluator); 
        task.prepareForUse();
        LearningCurve le = (LearningCurve) task.doTask();
        System.out.println(le); //imprime resultados
        
        
}
} 
