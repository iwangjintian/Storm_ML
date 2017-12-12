package algrithms;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import Jama.Matrix;

class Knn {
	public List<EnsembleDataRecord> wholedata;
	public List<EnsembleDataRecord> testing;
	public Matrix trainMatrix;
	public List<EnsembleDataRecord> training;
	public List<EnsembleFeature> EnsembleFeatures;
	public String[] actualLabel;
	
	public Knn(){
		wholedata = new ArrayList<EnsembleDataRecord>();
		testing = new ArrayList<EnsembleDataRecord>();
		training = new ArrayList<EnsembleDataRecord>();
		EnsembleFeatures = new ArrayList<EnsembleFeature>();
		actualLabel = new String[2];
		actualLabel[0] = "-1";
		actualLabel[1] = "-1";
	}

    /**
     * Turning the list of EnsembleDataRecord, which is the training, into a large Matrix.
     */
	public void readMatrix(){
		int row = training.size();
		int col = EnsembleFeatures.size();
		Matrix matrix = new Matrix(row, col);
		for (int c = 0; c < col; ++c) {
			if(EnsembleFeatures.get(c).type == 0){
				String colname = EnsembleFeatures.get(c).col;
				for (int r = 0; r < row; ++r) 
					matrix.set(r, c, Double.valueOf(training.get(r).getValue(colname)));
			}
			if(EnsembleFeatures.get(c).type == 2 || EnsembleFeatures.get(c).type == 1){
				String colname = EnsembleFeatures.get(c).col;
				for (int r = 0; r < row; ++r){
					String value = training.get(r).getValue(colname);
					matrix.set(r, c,value.hashCode());
				}

			}
			if(EnsembleFeatures.get(c).type == 4)
				for (int r = 0; r < row; ++r) 
					matrix.set(r, c,0);
		}
		trainMatrix = matrix;
	}
	
    /**
     * Read an EnsembleDataRecord and turn it into a double array.
     * @param record
     * 				data record in the form of EnsembleDataRecord
     * @return double[] 
     * 				data record in the form of double[]
     */
	public double[] readRecord(EnsembleDataRecord record){
		int col = EnsembleFeatures.size();
		double[] matrix = new double[col];
		for (int c = 0; c < col; ++c) {
			if(EnsembleFeatures.get(c).type == 0){
				String colname = EnsembleFeatures.get(c).col;
				matrix[c] =  Double.valueOf(record.getValue(colname));
			}
			if(EnsembleFeatures.get(c).type == 2 || EnsembleFeatures.get(c).type == 1){
				String colname = EnsembleFeatures.get(c).col;
				String value = record.getValue(colname);
				matrix[c] = (double) value.hashCode();
			}
			if(EnsembleFeatures.get(c).type == 4)
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
		int col = EnsembleFeatures.size();
		Matrix matrix = new Matrix(row, col);
		for (int c = 0; c < col; ++c) {
			if(EnsembleFeatures.get(c).type == 0){
				String colname = EnsembleFeatures.get(c).col;
				for (int r = 0; r < row; ++r) 
					matrix.set(r, c, Double.valueOf(testing.get(r).getValue(colname)));
			}
			if(EnsembleFeatures.get(c).type == 2 || EnsembleFeatures.get(c).type == 1){
				String colname = EnsembleFeatures.get(c).col;
				String level = EnsembleFeatures.get(c).level;
				for (int r = 0; r < row; ++r){
					String value = testing.get(r).getValue(colname);
					matrix.set(r, c,value.hashCode());
				}

			}
			if(EnsembleFeatures.get(c).type == 4)
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
	public void parseEnsembleFeature() {
		int number = Integer.valueOf(wholedata.get(0).label);
		for(int i = 0; i < number; i++){
			EnsembleFeature tmp = new EnsembleFeature();
			tmp.col = String.valueOf(i);
			List<String> coldata = getcoldata(tmp.col);
			if(!coldata.get(0).matches("\\d+(\\.\\d+)?")){		
				Set<String> level = new HashSet<String>(coldata);
				if(level.size()>10){
					tmp.type = 4;
					EnsembleFeatures.add(tmp);
				}
				else{
					if(level.size()==2)
						tmp.type = 2;
					else
						tmp.type = 1;
					EnsembleFeatures.add(tmp);
				}
			}
			else{
				Set<String> level = new HashSet<String>(coldata);
				if(level.size()>10){
					tmp.type = 0;
					EnsembleFeatures.add(tmp);
				}
				else{
					
					if(level.size() == 2)
						tmp.type = 2;
					else
						tmp.type = 1;
					EnsembleFeatures.add(tmp);					
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
		for(int i = 0 ;i < wholedata.size();i++){
			String tmp = wholedata.get(i).getValue(col);
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
	public void split(int train ){
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
     *Compute the distance between the given record with all the records in training. 
     *@param target
     *				the given data record.
     */
	public double[] distance(double[] target) {
		double[] result = new double[training.size()];
		for(int i = 0; i < training.size(); i++) {
			double[] tmp = trainMatrix.getArray()[i];
			double distance = elementDistance(target,tmp);
			result[i] = distance;
		}
		return result; // euclidian distance would be sqrt(sum)...
	}
	
	 /**
     *Compute the distance between two given records. 
     *@param target
     *				the given data record.
     *@param tmp
     *				the other given data record
     */
	public double elementDistance(double[] target, double[] tmp){
		double sum = 0;
		int a = 0,b = 0,c = 0 ,d = 0;
		for(int i = 0; i< target.length;i++){
			if(EnsembleFeatures.get(i).type == 0)
				sum += (target[i]-tmp[i]) * (target[i]-tmp[i]);
			if(EnsembleFeatures.get(i).type == 1)
				if(target[i] != tmp[i])
					sum += 1;
			if(EnsembleFeatures.get(i).type == 2){
				if(target[i]+tmp[i]==0)
					d++;
				if(target[i]+tmp[i]==2)
					a++;
				if(target[i]+tmp[i]==1)
					c++;
			}
		}
		if(a+b+c+d != 0){
			sum += (a+d)/(a+b+c+d);
		}	
		return sum;
	}
	
	 /**
     *A class which is created for sorting and figuring out the k nearest neighbors.
     */
	public class sortelement implements Comparable<sortelement>{
		public double distance;
		public int label;
		public sortelement(double d,int l){
			distance = d;
			label = l;
		}
		public int compareTo(sortelement compare) {
			double comparedistance = ((sortelement)compare).distance;
			return (int) (this.distance-comparedistance);
		}
	}
	
	 /**
     *Classify the given data. 
     *@param data
     *				the given data records in the form of matrix.
     *@param k
     *				k-nearest-neighbor.
     */
	public int[] classify(Matrix data, int k) {
		int row = data.getRowDimension();
		int[] label = new int[row];
		for(int i = 0; i < row; i++) {
			double[] dist = distance(data.getArray()[i]);
			sortelement[] dCollection = new sortelement[dist.length];
			for(int j = 0;j<dCollection.length;j++)
		    	dCollection[j] = new sortelement(dist[j],Integer.valueOf(training.get(j).getLabel().replaceAll("\\s", "")));
			Arrays.sort(dCollection);
			
			int sum = 0;
			k = k>training.size()?training.size():k;
			for(int a = 0; a < k; a ++)
				sum+=dCollection[a].label;
			if(sum > k/2)
				label[i] = 1;
			else
				label[i] = 0;
			
		}
		return label;
	}
	
	 /**
     *Calculate the accuracy with two given lists of labels. 
     *@param actual
     *				list of the actual labels.
     *@param predict
     *				list of the predicted labels
     */
	public double accurate(int[] actual, int[] predict){
		int acc = 0;
		for(int i = 0;i < actual.length;i++)
			if(actual[i] == predict[i])
				acc++;
		return ((double)acc)/actual.length;
	}
}
