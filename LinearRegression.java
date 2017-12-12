package algrithms;
import Jama.Matrix;
import java.io.FileReader;
import java.io.Serializable;
import java.io.BufferedReader;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;


public class LinearRegression implements Serializable {
	public Double lambda;
	public String trainingFile;
	public String testingFile;
	public List<EnsembleDataRecord> wholedata;
	public List<EnsembleDataRecord> testing;
	public List<EnsembleDataRecord> training;
	public List<EnsembleFeature> features;
	public double cutoff;
	public int number;
	public String[] actualLabel;
	
	public LinearRegression() {
		lambda = 1.0;
		wholedata = new ArrayList<EnsembleDataRecord>();
		features = new ArrayList<EnsembleFeature>();
		actualLabel = new String[2];
		actualLabel[0] = "-1";
		actualLabel[1] = "-1";
	}
	
    /**
     * Turning the list of EnsembleDataRecord, which is the training, into a large Matrix.
     * @return matrix row * col
     */
	public Matrix readMatrix(){
		int row = training.size();
		int col = features.size();
		Matrix matrix = new Matrix(row, col);
		for (int c = 0; c < col; c++) {
			if(features.get(c).type == 0){
				String colname = features.get(c).col;
				for (int r = 0; r < row; r++) 
					matrix.set(r, c, Double.valueOf(training.get(r).getValue(colname)));
			}
			if(features.get(c).type == 2 || features.get(c).type == 1){
				String colname = features.get(c).col;
				String level = features.get(c).level;
				for (int r = 0; r < row; r++){
					String value = training.get(r).getValue(colname);
					if(level.equals(value))
						matrix.set(r, c,1.0);
					else
						matrix.set(r, c, 0.0);
				}
				
			}
			if(features.get(c).type == 4)
				for (int r = 0; r < row; r++) 
					matrix.set(r, c,0);
		}
		return matrix;
	}
	
    /**
     * Read an EnsembleDataRecord and turn it into a double array.
     * @param record
     * 				data record in the form of EnsembleDataRecord
     * @return double[] 
     * 				data record in the form of double[]
     */
	public Double[] readRecord(EnsembleDataRecord record){
		int col = features.size();
		Double[] matrix = new Double[col];
		for (int c = 0; c < col; ++c) {
			if(features.get(c).type == 0){
				String colname = features.get(c).col;
				matrix[c] =  Double.valueOf(record.getValue(colname));
			}
			if(features.get(c).type == 2 || features.get(c).type == 1){
				String colname = features.get(c).col;
				String level = features.get(c).level;
				String value = record.getValue(colname);
				if(level.equals(value))
					matrix[c] = 1.0;
				else
					matrix[c] = 0.0;
			}
			if(features.get(c).type == 4)
				matrix[c] = 0.0;
		}
		return matrix;
	}
	
    /**
     * Turning the list of EnsembleDataRecord, which is the testing, into a large Matrix.
     * @return matrix row * col
     */
	public Matrix readtestingMatrix(){
		int row = testing.size();
		int col = features.size();
		Matrix matrix = new Matrix(row, col);
		for (int c = 0; c < col; c++) {
			if(features.get(c).type == 0){
				String colname = features.get(c).col;
				for (int r = 0; r < row; ++r) 
					matrix.set(r, c, Double.valueOf(testing.get(r).getValue(colname)));
			}
			if(features.get(c).type == 2 || features.get(c).type == 1){
				String colname = features.get(c).col;
				String level = features.get(c).level;
				for (int r = 0; r < row; r++){
					String value = testing.get(r).getValue(colname);
					if(level.equals(value))
						matrix.set(r, c,1.0);
					else
						matrix.set(r, c, 0.0);
				}
				
			}
			if(features.get(c).type == 4)
				for (int r = 0; r < row; ++r) 
					matrix.set(r, c,0);
		}
		return matrix;
	}
	
    /**
     * Adding data record.
     * @param line
     *            one record of the data
     */
	public void loadData(String line) {
		String[] record = line.split(",");
		EnsembleDataRecord newdata = new EnsembleDataRecord(record,record.length);
		wholedata.add(newdata);	
	}

	  /**
     * Replace labels of different kind of data with 1 and 0.
     * @param number
     *            index of the label
     */
	public void replacelabel(int number){
    	if(actualLabel[0].equals(actualLabel[1])){
			List<String> coldata = getcoldata(String.valueOf(number-1));
			Set<String> level = new HashSet<String>(coldata);
			Object[] label = level.toArray();
			String la = (String) label[0];
			for(int i = 0; i < wholedata.size(); i++){
				EnsembleDataRecord tmp = wholedata.get(i);
				if(tmp.getLabel().equals(la))
					tmp.setLabel("1");
				else
					tmp.setLabel("0");
			}
			actualLabel[0] = la;
			actualLabel[1] = (String) label[1];
		}
		else{
			for(int i = 0; i < wholedata.size(); i++){
				EnsembleDataRecord tmp = wholedata.get(i);
				if(tmp.getLabel().equals(actualLabel[0]))
					tmp.setLabel("1");
				if(tmp.getLabel().equals(actualLabel[1]))
					tmp.setLabel("0");
			}
		}
	}
	
	 /**
     * Analyze gathered data and get column features. 
     */
	public void parsefeature() {
		int number = Integer.valueOf(wholedata.get(0).label);
		for(int i = 0; i < number; i++){
			EnsembleFeature tmp = new EnsembleFeature();
			tmp.col = String.valueOf(i);
			List<String> coldata = getcoldata(tmp.col);
			if(!coldata.get(0).matches("\\d+(\\.\\d+)?")){		
				Set<String> level = new HashSet<String>(coldata);
				if(level.size()>10){
					tmp.type = 4;
					features.add(tmp);
				}
				else{
					Object[] lev = level.toArray();
					for(int j = 1;j<lev.length;j++){
						EnsembleFeature sub = new EnsembleFeature();
						sub.col = String.valueOf(i);
						sub.level = (String)lev[j];
						if(lev.length == 2)
							sub.type = 2;
						else
							sub.type = 1;
						features.add(sub);					
					}
				}
			}
			else{
				Set<String> level = new HashSet<String>(coldata);
				if(level.size()>10){
					tmp.type = 0;
					features.add(tmp);
				}
				else{
					Object[] lev = level.toArray();
					for(int j = 1;j<lev.length;j++){
						EnsembleFeature sub = new EnsembleFeature();
						sub.col = String.valueOf(i);
						sub.level = (String)lev[j];
						if(lev.length == 2)
							sub.type = 2;
						else
							sub.type = 1;
						features.add(sub);					
					}
				}
			}
		}
	}
	
	 /**
     *Return the column data of the whole data set given the index. 
     *@param col
     *				the index of the column.
     *@return List<String> with the number of wholedata.size()
     */
	public List<String> getcoldata(String col){
		List<String> coldata = new ArrayList<String>();
		for(int i = 0 ;i < training.size();i++){
			String tmp = training.get(i).getValue(col);
			coldata.add(tmp);
		}
		return coldata;
	}
	
	 /**
     *Return the testing column data given the index. 
     *@param col
     *				the index of the column.
     *@return List<String> with the number of wholedata.size()
     */
	public List<String> gettestingcoldata(String col){
		List<String> coldata = new ArrayList<String>();
		for(int i = 0 ;i < testing.size();i++){
			String tmp = testing.get(i).getValue(col);
			coldata.add(tmp);
		}
		return coldata;
	}

	 /**
     *Split the given data set into two parts based on a particular proportion . 
     *@param raw
     *				the given data set.
     *@param train
     *				proportion of training part
     */
	 public void split( int train ){
	    	train = train*wholedata.size()/100;
	    	Random rng = new Random();
	    	Set<Integer> generated = new LinkedHashSet<Integer>();
	    	int range = wholedata.size();
	    	while(generated.size()<train){
	    		int next = rng.nextInt(range);
	    		generated.add(next);
	    	}
	    	List<EnsembleDataRecord> training = new ArrayList<EnsembleDataRecord>();
	    	List<EnsembleDataRecord> testing = new ArrayList<EnsembleDataRecord>();
	    	for (int i = 0; i < range; i++)
	    		if(generated.contains(i))
	    			training.add(wholedata.get(i));
	    		else
	    			testing.add(wholedata.get(i));
	    	this.training = training;
	    	this.testing = testing;    	
	    }
	
	/**
	 * train the model using linear regression with L2 regularizer
	 * Linear regression with L2 regularizer has the close form as follows:
	 * w = (x^{T} * X + \lambda * I)^{-1} * X^{T} * t 
	 * 
	 * @param data n * m
	 * @param targets n * 1
	 * @param lambda
	 * @return weight m * 1
	 */
	public Matrix trainLinearRegressionModel(Matrix data, Matrix targets, Double lambda) {
		int row = data.getRowDimension();
		int column = data.getColumnDimension();
		Matrix identity = Matrix.identity(column, column);
		identity.times(lambda);
		Matrix dataCopy = data.copy();
		Matrix transponseData = dataCopy.transpose();
		Matrix norm = transponseData.times(data);
		Matrix circular = norm.plus(identity);
		Matrix circularInverse = circular.inverse();
		Matrix former = circularInverse.times(data.transpose());
		Matrix weight = former.times(targets);
		
	    return weight;
	}
	
	/**
	 * calculate the accuracy under the best cutoff value
	 * 
	 * @param data
	 * @param targets
	 * @param weights
	 * @return accuracy
	 */
	public double accuraccy (Matrix data, Matrix targets, Matrix weights){
		Matrix predictTargets = predict(data, weights);
		int accurate = 0;
		double cutoff = 0;
		Double[] predictClass = new Double[targets.getRowDimension()];
		for(double i = 0.1;i<1;i+=0.1){
			int tmp = 0;
			for(int j = 0 ;j<targets.getRowDimension();j++){
				if(predictTargets.get(j, 0)>i){
					predictClass[j] = 1.0;
					if(targets.get(j, 0)==1.0)
						tmp++;
				}
				else{
					predictClass[j] = 0.0;
					if(targets.get(j, 0)==0.0)
						tmp++;
				}
			}
			if(tmp>accurate){
				accurate = tmp;
				cutoff = i;
			}
		}
		this.cutoff = cutoff;
		return (double)accurate/targets.getRowDimension();
	}
	
	/**
	 * calcualte the predict targests given features and the learned weights
	 * 
	 * @param data a matrix with n * m 
	 * @param weights a matrix with 1 * m
	 * @return predict targets according to the weight
	 */
	private Matrix predict(Matrix data, Matrix weights) {
		int row = data.getRowDimension();
		Matrix predictTargets = new Matrix(row, 1);
		for (int i = 0; i < row; i++) {
			double value = multiply(data.getMatrix(i, i, 0, data.getColumnDimension() -1 ), weights);
			//System.out.println(value);
			predictTargets.set(i, 0, value);
		}
		return predictTargets;
	}
	

	/**
	 * multiply two matrix with just 1 row and seveal columns
	 * 
	 * @param data a matrix with 1 * column 
	 * @param weights a matrix with 1 * column
	 * @return
	 */
	private Double multiply(Matrix data, Matrix weights) {
		Double sum = 0.0;
		int column = data.getColumnDimension();
		for (int i = 0; i <column; i++) {
			sum += data.get(0, i) * weights.get(i, 0);
		}
		return sum;
	}
	

//	public static void main(String[] args) {
//		LinearEnsembleression lr = new LinearEnsembleression();
//		try {
//			lr.loadData("nxd10.txt", 16);
//			lr.parsefeature();
//			Matrix data = lr.readMatrix();
//			data.print(1, 2);
//			List<String> tar = lr.getcoldata("15");
//			Matrix target = new Matrix(tar.size(),1);
//			for (int i = 0; i < tar.size(); i++)
//				target.set(i, 0, tar.get(i).contains("1")?1.0:0.0);
//			Matrix weights = lr.trainLinearEnsembleressionModel(data, target, lr.lambda);
//		    for (int i = 0; i < weights.getRowDimension(); i++) {
//			System.out.println(weights.get(i, 0));
//		    }
//		    double training_error = lr.evaluateLinearEnsembleressionModel(data, target, weights);
//		    System.out.println(training_error);
//		    List<String> testingtar = lr.gettestingcoldata("15");
//			Matrix testingtarget = new Matrix(testingtar.size(),1);
//			for (int i = 0; i < testingtar.size(); i++)
//				testingtarget.set(i, 0, testingtar.get(i).contains("1")?1.0:0.0);
//			Matrix testingdata = lr.readtestingMatrix();
//		    System.out.println(lr.accuraccy(testingdata, testingtarget, weights));
//			/** get the actual features, meanwhile add a N*1 column vector with value being all 1 as the first column of the features */
////			Matrix trainingData = lr.getDataPoints(training);
////			Matrix testingData = lr.getDataPoints(testing);
////			
////			Matrix trainingTargets = lr.getTargets(training);
////		    Matrix testingTargets = lr.getTargets(testing);
////		    
////		    // Train the model.
////		    Matrix weights = lr.trainLinearEnsembleressionModel(trainingData, trainingTargets, lr.lambda);
////		    for (int i = 0; i < weights.getRowDimension(); i++) {
////			System.out.println(weights.get(i, 0));
////		    }
////		    
////		    // Evaluate the model using training and testing data.
////		    double training_error = lr.evaluateLinearEnsembleressionModel(trainingData, trainingTargets, weights);
////		    double testing_error = lr.evaluateLinearEnsembleressionModel(testingData, testingTargets, weights);
////
////		    System.out.println(training_error);
////		    System.out.println(testing_error);
//		} catch (Exception e) {
//			e.printStackTrace();
//			System.exit(1);			
//		}
//	}
	
}