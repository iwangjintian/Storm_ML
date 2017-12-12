package algrithms;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import Jama.Matrix;






public class Bagging 
{
	public List<EnsembleDataRecord> wholedata;
	public List<EnsembleDataRecord> testing;
	public List<EnsembleDataRecord> training;
	private static List<EnsembleFeature> features;
	public List<EnsembleDecisionTree> rf = null;
	public String[] actualLabel;

	public Bagging(){
		wholedata = new ArrayList<EnsembleDataRecord>();
		testing = new ArrayList<EnsembleDataRecord>();
		training = new ArrayList<EnsembleDataRecord>();
		features = new ArrayList<EnsembleFeature>();
		rf = new ArrayList<EnsembleDecisionTree>();
		actualLabel = new String[2];
		actualLabel[0] = "-1";
		actualLabel[1] = "-1";
	}

	/**
	 * Create a certain amount of decision trees and train them individually.
	 * @param treenumber
	 * 					the number of trees
	 */
	public void train(int treenumber){
		for(int i = 0; i < treenumber; i++){
			List<EnsembleDataRecord> subset = splitdata(training.size()/treenumber);
			EnsembleDecisionTree tmp = new EnsembleDecisionTree(subset,testing,features);
			rf.add(tmp);
		}	
		for(int i = 0;i < rf.size(); i++)
			rf.get(i).trainB();
	}


	/**
	 *Predict the labels of a list of EnsembleDataRecords which are contained in testing 
	 *@return the predicted labels
	 */
	public List<Boolean> predict(){
		List<List<Boolean>> result = new ArrayList<List<Boolean>>();
		for(int i = 0;i < rf.size(); i++)
			result.add(rf.get(i).predict(testing));
		List<Boolean> prediction = new ArrayList<Boolean>();
		for(int j = 0; j < testing.size(); j++){
			int sum = 0;
			for(int i = 0; i < result.size(); i++){
				if(result.get(i).get(j))
					sum++;
			}
			if(sum > rf.size()/2)
				prediction.add(true);
			else
				prediction.add(false);
		}
		return prediction;
	}

	/**
	 *Predict the label of an EnsembleDataRecords. 
	 *@param data
	 *				the given data
	 *@return the predicted label
	 */
	public Boolean predict(EnsembleDataRecord record){
		int sum = 0;
		for(int i = 0;i < rf.size(); i++){
			if(rf.get(i).predict(record))
				sum++;
		}
		if(sum > rf.size()/2)
			return true;
		else
			return false;

	}

	/**
	 *Calculate the accuracy based on the parameters. 
	 *@param actual
	 *				actual label of the data
	 *@param predict
	 *				predicted label of the data
	 *@return the accuracy
	 */
	public double accuracy (List<Boolean> actual, List<Boolean> predict){
		if(actual.size() != predict.size())
			return -1;
		int count = 0;
		for (int i = 0; i < actual.size(); i++)
			if(actual.get(i) == predict.get(i))
				count++;
		return ((double)(count))/actual.size();
	}


	/**
	 *Split the wholedata set into two parts based on a particular proportion . 
	 *@param train
	 *				proportion of training part
	 *@return the two parts of the wholedata included in a list
	 */
	private List<EnsembleDataRecord> splitdata(int number){
		Random rng = new Random();
		Set<Integer> generated = new LinkedHashSet<Integer>();
		int range = training.size();
		while(generated.size()<number){
			int next = rng.nextInt(range);
			generated.add(next);
		}
		List<EnsembleDataRecord> subdata = new ArrayList<EnsembleDataRecord>();
		Object[] chosen = generated.toArray();
		for (int i = 0; i < chosen.length; i++)
			subdata.add(training.get((int) chosen[i]));
		return subdata;
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
	 *Return the column data of the whole data set given the index. 
	 *@param col
	 *				the index of the column.
	 */
	private List<String> getcoldata(String col){
		List<String> coldata = new ArrayList<String>();
		for(int i = 0 ;i < wholedata.size();i++){
			String tmp = wholedata.get(i).getValue(col);
			coldata.add(tmp);
		}
		return coldata;
	}

	/**
	 * Analyze gathered data and get column features. 
	 */
	public void set_boundary(){
		int number = Integer.valueOf(wholedata.get(0).label);
		for(int i = 0; i < number; i++){
			List<String> coldata = getcoldata(String.valueOf(i));
			if(!coldata.get(0).matches("\\d+(\\.\\d+)?")){		
				Set<String> level = new HashSet<String>(coldata);
				if(level.size()>10){
					EnsembleFeature tmp = new EnsembleFeature();
					tmp.col = String.valueOf(i);
					tmp.type = 4;
					features.add(tmp);
				}
				else{
					if(level.size()==2){
						EnsembleFeature tmp = new EnsembleFeature();
						tmp.col = String.valueOf(i);
						tmp.type = 2;
						tmp.dummy = (String)level.toArray()[0];
						features.add(tmp);
					}
					else{
						Object[] lev = level.toArray();
						for(int l = 0; l < level.size(); l++){
							EnsembleFeature tmp = new EnsembleFeature();
							tmp.col = String.valueOf(i);
							tmp.type = 1;
							tmp.catagory = (String) lev[l];
							features.add(tmp);
						}
					}
				}
			}
			else{
				Set<String> level = new HashSet<String>(coldata);
				if(level.size()>10){
					double mean = getcolmean(coldata);
					double stdev = getcolstdev(coldata);
					double lower_bound = -Double.MAX_VALUE;
					double upper_bound = mean-3*stdev;
					while(upper_bound <= mean+3*stdev){
						EnsembleFeature tmp = new EnsembleFeature();
						tmp.col = String.valueOf(i);
						tmp.type = 0;
						tmp.lower_bound = lower_bound;
						tmp.higher_bound = upper_bound;
						features.add(tmp);
						lower_bound = upper_bound;
						upper_bound +=stdev;
					}
				}
				else{
					if(level.size()==2){
						EnsembleFeature tmp = new EnsembleFeature();
						tmp.col = String.valueOf(i);
						tmp.type = 2;
						tmp.dummy = (String)level.toArray()[0];						
						features.add(tmp);
					}
					else{
						Object[] lev = level.toArray();
						for(int l = 0; l < level.size(); l++){
							EnsembleFeature tmp = new EnsembleFeature();
							tmp.col = String.valueOf(i);
							tmp.type = 1;
							tmp.catagory = (String) lev[l];
							features.add(tmp);
						}
					}					
				}
			}
		}
	}

	/**
	 *Calculate the column data mean given the parameter. 
	 *@param coldata
	 *				the list of the column data
	 */
	private double getcolmean(List<String> coldata){
		double[] tmp = new double[coldata.size()];
		double sum = 0;
		for(int i = 0; i < tmp.length; i++){
			tmp[i] = Double.valueOf(coldata.get(i));
			sum+=tmp[i];
		}
		return sum/tmp.length;
	}

	/**
	 *Calculate the standard deviation of column data mean given the parameter. 
	 *@param coldata
	 *				the list of the column data
	 */
	private double getcolstdev(List<String> coldata){
		double mean = getcolmean(coldata);
		double[] tmp = new double[coldata.size()];
		double sum = 0;
		for(int i = 0; i < tmp.length; i++){
			tmp[i] = Double.valueOf(coldata.get(i));
			sum+=(tmp[i]-mean)*(tmp[i]-mean);
		}
		return Math.sqrt(sum)/tmp.length;
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
	 *Get the actual label of the testing data. 
	 *@param testing
	 *				the given data
	 *@return the list of the actual labels
	 */
	public static List<Boolean> actual(List<EnsembleDataRecord> testing){
		List<Boolean> actual = new ArrayList<Boolean>();
		for (int i = 0; i < testing.size(); i++){
			if(testing.get(i).getLabel().contains("1"))
				actual.add(true);
			else
				actual.add(false);
		}
		return actual;
	}

	//	public static void main( String[] args ) throws IOException
	//    {
	//		Bagging rf = new Bagging();
	//		rf.train(4);
	//		List<Boolean> prediction = rf.predict();
	//		List<Boolean> actual = actual(rf.testing);
	//		System.out.println(rf.accuracy(actual, prediction));
	//    }
}
