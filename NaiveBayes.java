package algrithms;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;


public class NaiveBayes {
	public List<EnsembleDataRecord> wholedata;
	public List<EnsembleDataRecord> testing;
	public List<EnsembleDataRecord> training;
	public List<List<EnsembleFeature>> features;
	public String[] actualLabel;

	public NaiveBayes(){
		wholedata = new ArrayList<EnsembleDataRecord>();
		testing = new ArrayList<EnsembleDataRecord>();
		training = new ArrayList<EnsembleDataRecord>();
		features = new ArrayList<List<EnsembleFeature>>();
		actualLabel = new String[2];
		actualLabel[0] = "-1";
		actualLabel[1] = "-1";
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
	public void set_boundary(){
		int number = Integer.valueOf(wholedata.get(0).label);
		for(int i = 0; i < number; i++){
			features.add(new ArrayList<EnsembleFeature>());
			List<String> coldata = getcoldata(String.valueOf(i));
			if(!coldata.get(0).matches("\\d+(\\.\\d+)?")){		
				Set<String> level = new HashSet<String>(coldata);
				if(level.size()>10){
					EnsembleFeature tmp = new EnsembleFeature();
					tmp.col = String.valueOf(i);
					tmp.type = 4;
					features.get(i).add(tmp);
				}
				else{
					Object[] lev = level.toArray();
					for(int l = 0; l < level.size(); l++){
						EnsembleFeature tmp = new EnsembleFeature();
						tmp.col = String.valueOf(i);
						tmp.type = 1;
						tmp.catagory = (String) lev[l];
						features.get(i).add(tmp);
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
						features.get(i).add(tmp);
						lower_bound = upper_bound;
						upper_bound +=stdev;
					}
				}
				else{
					Object[] lev = level.toArray();
					for(int l = 0; l < level.size(); l++){
						EnsembleFeature tmp = new EnsembleFeature();
						tmp.col = String.valueOf(i);
						tmp.type = 1;
						tmp.catagory = (String) lev[l];
						features.get(i).add(tmp);
					}				
				}
			}
		}
	}
	
	 /**
     * Calculate the conditional possibility of each feature and store them in the EnsembleFeature. 
     * The conditional possibility is the possibility of each feature under each label.    
     */
	public void calculateBasicPossibility(){
		List<String> label = gettraincoldata(wholedata.get(0).label);
		int ones = 0, zeros = 0;
		for(int i = 0; i < label.size(); i++)
			if(label.get(i).contains("1"))
				ones++;
			else
				zeros++;
		for(int i = 0; i < features.size(); i++){
			for(int j = 0; j <features.get(i).size();j++){
				EnsembleFeature tmp = features.get(i).get(j);
				List<String> values = gettraincoldata(String.valueOf(i));
				int pr = 0, fr = 0;
				if(tmp.type == 1 || tmp.type == 2){
					String catagory = tmp.type == 1?tmp.catagory:tmp.dummy;
					for(int l = 0; l < values.size(); l++){
						if(values.get(l).equals(catagory)&&label.get(l).contains("1"))
							pr++;
						if(values.get(l).equals(catagory)&&label.get(l).contains("0"))
							fr++;	
					}
				}
				if(tmp.type == 0){
					double lower = tmp.lower_bound;
					double upper = tmp.higher_bound;
					for(int l = 0; l < values.size(); l++){
						double value = Double.valueOf(values.get(l));
						if(value>=lower&&value<=upper&&label.get(l).contains("1"))
							pr++;
						if(value>=lower&&value<=upper&&label.get(l).contains("0"))
							fr++;
					}
				}
				tmp.pr = ((double) pr) / ones;
				tmp.fr = ((double) fr) / zeros;
			}
		}
	}
	
	 /**
     * Calculate the possibility of each EnsembleDataRecord in wholedata whether it belongs to label 1 or 0.
     * And store them in the EnsembleDataRecord.   
     */
	public void computePossibility(){
		List<String> label = gettraincoldata(wholedata.get(0).label);
		int ones = 0, zeros = 0;
		for(int i = 0; i < label.size(); i++)
			if(label.get(i).contains("1"))
				ones++;
			else
				zeros++;
		double p1 = (double)ones/training.size(), p0 = (double)zeros/training.size();
		for(int i = 0; i < testing.size(); i++){
			EnsembleDataRecord tmp = testing.get(i);
			double pr = 1.0, fr = 1.0;
			for(int j = 0; j < Integer.valueOf(tmp.label); j++){
				List<EnsembleFeature> featurelist = features.get(j);
				if(featurelist.get(0).type == 1){
					for(int l = 0; l < featurelist.size(); l++){
						if(featurelist.get(l).catagory.equals(tmp.getValue(String.valueOf(j)))){
							pr *= featurelist.get(l).pr;
							fr *= featurelist.get(l).fr;
						}	
					}
				}
				if(featurelist.get(0).type == 0){
					double value = Double.valueOf(tmp.getValue(String.valueOf(j)));
					for(int l = 0; l < featurelist.size(); l++){
						if(value>=featurelist.get(l).lower_bound&&value<=featurelist.get(l).higher_bound){
							pr *= featurelist.get(l).pr;
							fr *= featurelist.get(l).fr;
						}	
					}
				}
			}
			tmp.pr = pr;
			tmp.fr = fr;
		}
	}
	
	 /**
     *Predict the label of one EnsembleDataRecord. 
     */
	public boolean predict(EnsembleDataRecord tmp){
		double pr = 1.0, fr = 1.0;
		for(int j = 0; j < Integer.valueOf(tmp.label); j++){
			List<EnsembleFeature> featurelist = features.get(j);
			if(featurelist.get(0).type == 1){
				for(int l = 0; l < featurelist.size(); l++){
					if(featurelist.get(l).catagory.equals(tmp.getValue(String.valueOf(j)))){
						pr *= featurelist.get(l).pr;
						fr *= featurelist.get(l).fr;
					}	
				}
			}
			if(featurelist.get(0).type == 0){
				double value = Double.valueOf(tmp.getValue(String.valueOf(j)));
				for(int l = 0; l < featurelist.size(); l++){
					if(value>=featurelist.get(l).lower_bound&&value<=featurelist.get(l).higher_bound){
						pr *= featurelist.get(l).pr;
						fr *= featurelist.get(l).fr;
					}	
				}
			}
		}
		tmp.pr = pr;
		tmp.fr = fr;
		if(pr>=fr)
			return true;
		else
			return false;
	}
	
	 /**
     *Get the actual label of the testing data. 
     */
	public List<Boolean> actual(){
		List<Boolean> actual = new ArrayList<Boolean>();
		for (int i = 0; i < testing.size(); i++){
			if(testing.get(i).getLabel().contains("1"))
				actual.add(true);
			else
				actual.add(false);
		}
		return actual;
	}

	 /**
     *Predict the label of a list of EnsembleDataRecords. 
     */
	public List<Boolean> prediction(){
		List<Boolean> actual = new ArrayList<Boolean>();
		for (int i = 0; i < testing.size(); i++){
			if(testing.get(i).pr >=  testing.get(i).fr)
				actual.add(true);
			else
				actual.add(false);
		}
		return actual;
	}
	
	 /**
     *Calculate the accuracy based on the parameters. 
     *@param actual
     *				actual label of the data
     *@param predict
     *				predicted label of the data
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
     *Return the training column data given the index. 
     *@param col
     *				the index of the column.
     */
	public List<String> gettraincoldata(String col){
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
     */
	private List<String> gettestingcoldata(String col){
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
	public void split(List<EnsembleDataRecord> raw, int train ){
		train = train*raw.size()/100;
		Random rng = new Random();
		Set<Integer> generated = new LinkedHashSet<Integer>();
		int range = raw.size();
		while(generated.size()<train){
			int next = rng.nextInt(range);
			generated.add(next);
		}
		List<EnsembleDataRecord> training = new ArrayList<EnsembleDataRecord>();
		List<EnsembleDataRecord> testing = new ArrayList<EnsembleDataRecord>();
		for (int i = 0; i < range; i++)
			if(generated.contains(i))
				training.add(raw.get(i));
			else
				testing.add(raw.get(i));
		this.training = training;
		this.testing = testing;
	}

	//---------------------------------------------main
	public static void main(String[] argv){
		NaiveBayes nb = new NaiveBayes();
		nb.loadData("nxd10.txt");
		nb.calculateBasicPossibility();
		nb.computePossibility();
		System.out.println(nb.accuracy(nb.actual(), nb.prediction()));
	}
}
