package algrithms;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import Jama.Matrix;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;
import java.util.LinkedHashSet;


public class LogRegression {

	/** the learning rate */
	private double rate;
	public List<EnsembleDataRecord> wholedata;
	public List<EnsembleDataRecord> testing;
	public List<EnsembleDataRecord> training;
	public List<EnsembleFeature> features;
	/** the weight to learn */
	public double[] weights = null;
	public double cutoff = 0;
	/** the number of iterations */
	private int ITERATIONS = 1000;
	public String[] actualLabel;

	public LogRegression() {
		this.rate = 0.0001;
		wholedata = new ArrayList<EnsembleDataRecord>();
		features = new ArrayList<EnsembleFeature>();
		actualLabel = new String[2];
		actualLabel[0] = "-1";
		actualLabel[1] = "-1";
	}

	/**
	 * Compute the sigmoid of a double
	 * @param z 
	 * @return double sigmoid(z)
	 */
	public double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	/**
	 * train the model using gradient descent and the sigmoid is defined as above
	 * @param data row * col
	 */
	public void train(Matrix data) {
		int row = data.getRowDimension();
		int column = data.getColumnDimension();
		for (int n=0; n<ITERATIONS; n++) {
			double lik = 0.0;
			for (int i=0; i<row; i++) {
				double[] x = data.getArray()[i];
				double predicted = classify(x);
				int label = Integer.valueOf(training.get(i).getLabel().replaceAll("\\s", ""));
				for (int j=0; j<weights.length; j++) {
					weights[j] = weights[j] + rate * (label - predicted) * x[j];
				}
				// not necessary for learning
				lik += label * Math.log(classify(x)) + (1-label) * Math.log(1- classify(x));
			}
			//System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
		}
	}
	
	/**
	 * Calculate the possibility that a data record has label 1.
	 * @param x 
	 * 			data record in the form of double
	 * @return double the possibility
	 */
	private double classify(double[] x) {
		double logit = .0;
		for (int i=0; i<weights.length;i++)  {
			logit += weights[i] * x[i];
		}
		return sigmoid(logit);
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
				for (int r = 0; r < row; ++r) 
					matrix.set(r, c,0);
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
				for (int r = 0; r < row; r++) 
					matrix.set(r, c, Double.valueOf(testing.get(r).getValue(colname)));
			}
			if(features.get(c).type == 2 || features.get(c).type == 1){
				String colname = features.get(c).col;
				String level = features.get(c).level;
				for (int r = 0; r < row; r++){
					if(level.equals(testing.get(r).getValue(colname)))
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
			//features.add(tmp);
		}
		weights = new double[features.size()];
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
	 * calculate the accuracy under the best cutoff value
	 * @param data the matrix of the data without the labels
	 * @param targets the actual label of the data
	 * @return accuracy
	 */
	public double accuraccy (Matrix data, int[] targets){
		double[] predictTargets = predict(data);
		int accurate = 0;
		double cutoff = 0;
		Double[] predictClass = new Double[targets.length];
		for(double i = 0.1;i<1;i+=0.1){
			int tmp = 0;
			for(int j = 0 ;j<targets.length;j++){
				if(sigmoid(predictTargets[j])>i){
					predictClass[j] = 1.0;
					if(targets[j]==1.0)
						tmp++;
				}
				else{
					predictClass[j] = 0.0;
					if(targets[j]==0.0)
						tmp++;
				}
			}
			if(tmp>accurate){
				accurate = tmp;
				cutoff = i;
			}
		}
		this.cutoff = cutoff;
		return (double)accurate/targets.length;
	}
	
	/**
	 * calcualte the predict labels given features and the learned weights
	 * @param data a matrix with n * m 
	 * @return predict targets according to the weight
	 */
	private double[] predict(Matrix data) {
		int row = data.getRowDimension();
		int col = data.getColumnDimension();
		double[] predictTargets = new double[row];
		for (int i = 0; i < row; i++) {
			double poss = 0;
			for(int j = 0;j<col;j++){
				poss+=data.get(i, j)*weights[j];
			}
			//System.out.println(value);
			predictTargets[i] = poss;
		}
		return predictTargets;
	}
	
	
//	public static void main(String... args) {
//		LogEnsembleression lr = new LogEnsembleression();
//		lr.loadData("nxd10.txt");
//		lr.parsefeature();
//		Matrix data = lr.readMatrix();
//		//data.print(1, 2);
//		List<String> tar = lr.getcoldata("15");
//		Matrix target = new Matrix(tar.size(),1);
//		for (int i = 0; i < tar.size(); i++)
//			target.set(i, 0, tar.get(i).contains("1")?1.0:0.0);
//		lr.train(data);
//		for (int i = 0; i < lr.weights.length; i++) {
//			System.out.println(lr.weights[i]);
//		}
//		List<String> testingtar = lr.gettestingcoldata("15");
//		int[] testingtarget = new int[testingtar.size()];
//		for(int i = 0;i<testingtar.size();i++)
//	    	testingtarget[i] = Integer.valueOf(testingtar.get(i).replaceAll("\\s", ""));
//		Matrix testingdata = lr.readtestingMatrix();
//	    System.out.println(lr.accuraccy(testingdata, testingtarget));
//	    double[] zero = new double[5];
//	    System.out.println(zero[2]);
//	}

}