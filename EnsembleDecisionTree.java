package algrithms;



import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

import java.util.Random;
import java.util.Set;

public class EnsembleDecisionTree {
	public List<EnsembleDataRecord> wholedata;
	public List<EnsembleDataRecord> testing;
	public List<EnsembleDataRecord> training;
	private List<EnsembleFeature> features;
	public EnsembleNode root;
	public String[] actualLabel;
	private int maxDepth = 8;
	
	 /**
     * Used to train decision tree model and bagging model.
     */
	public void trainB(){
		List<EnsembleDataRecord> data = training;
		List<EnsembleFeature> features = this.features;
		root = growTreeB(data,features,1);
	}
	
	 /**
     * Used to train random forest model.
     */
	public void trainRF(){
		List<EnsembleDataRecord> data = training;
		List<EnsembleFeature> features = this.features;
		root = growTreeRF(data,features,1);
	}
	
	 /**
     * Find out the majority label among the given data.
     * @param data
     * 				a list of EnsembleDataRecord
     * @return
     * 				the majority label
     */
	private String getMajorityLabel(List<EnsembleDataRecord> data){
		int count = 0;
		for(int i = 0; i < data.size(); i++){
			if(data.get(i).getLabel().contains("1"))
				count++;
		}

		return count > data.size()/2?"1":"0";
	}
	
	 /**
     * A recursion method used to build one decision tree for itself or bagging.
     * @param data
     * 				the data which is used to grow the tree to the next step
     * @param features
     * 				the features which are available at this step
     * @param depth
     * 				the tree depth at this moment
     * @return		the tree node at this step
     */
	private EnsembleNode growTreeB(List<EnsembleDataRecord> data, List<EnsembleFeature> features, int depth){
		boolean stoppingCriteriaReached = features.isEmpty() || depth >= maxDepth;
		if (stoppingCriteriaReached) {
			String majorityLabel = getMajorityLabel(data);
			EnsembleNode node = new EnsembleNode(majorityLabel);
			node.leaf = true;
			node.pr = Math.floor(computePRate(data)*1000)/1000;
			node.nr = Math.floor((1 - computePRate(data))*1000)/1000;
			return node;
		}

		EnsembleFeature bestSplit = findBestSplitFeatureB(data, features); // get best set of literals
		List<List<EnsembleDataRecord>> splitData = bestSplit.split(data);
		List<EnsembleFeature> newFeatures = new ArrayList<EnsembleFeature>();
		for(int i = 0;i<features.size();i++)
			if(features.get(i)!=bestSplit)
				newFeatures.add(features.get(i));
		EnsembleNode node = new EnsembleNode(bestSplit);
		if(splitData.get(0).isEmpty() || splitData.get(1).isEmpty()){
			node.label = getMajorityLabel(data).contains("1")?true:false;
			node.leaf = true;
			node.pr = Math.floor(computePRate(data)*1000)/1000;
			node.nr = Math.floor((1 - computePRate(data))*1000)/1000;
			return node;
		}
		for (List<EnsembleDataRecord> subsetTrainingData : splitData)  // add children to current node according to split
			node.addChild(growTreeB(subsetTrainingData, newFeatures, depth + 1));

		double gini = computeGINIIndex(data)/data.size();
		node.gini = Math.floor(gini*1000)/1000;
		return node;

	}

	 /**
     * A recursion method used to build one decision tree for random forest.
     * @param data
     * 				the data which is used to grow the tree to the next step
     * @param features
     * 				the features which are available at this step
     * @param depth
     * 				the tree depth at this moment
     * @return		the tree node at this step
     */
	private EnsembleNode growTreeRF(List<EnsembleDataRecord> data, List<EnsembleFeature> features, int depth){
		boolean stoppingCriteriaReached = features.isEmpty() || depth >= maxDepth;
		if (stoppingCriteriaReached) {
			String majorityLabel = getMajorityLabel(data);
			EnsembleNode node = new EnsembleNode(majorityLabel);
			node.leaf = true;
			node.pr = Math.floor(computePRate(data)*1000)/1000;
			node.nr = Math.floor((1 - computePRate(data))*1000)/1000;
			return node;
		}

		EnsembleFeature bestSplit = findBestSplitFeatureRF(data, features); // get best set of literals
		List<List<EnsembleDataRecord>> splitData = bestSplit.split(data);
		List<EnsembleFeature> newFeatures = new ArrayList<EnsembleFeature>();
		for(int i = 0;i<features.size();i++)
			if(features.get(i)!=bestSplit)
				newFeatures.add(features.get(i));
		EnsembleNode node = new EnsembleNode(bestSplit);
		if(splitData.get(0).isEmpty() || splitData.get(1).isEmpty()){
			node.label = getMajorityLabel(data).contains("1")?true:false;
			node.leaf = true;
			node.pr = Math.floor(computePRate(data)*1000)/1000;
			node.nr = Math.floor((1 - computePRate(data))*1000)/1000;
			return node;
		}
		for (List<EnsembleDataRecord> subsetTrainingData : splitData)  
			node.addChild(growTreeRF(subsetTrainingData, newFeatures, depth + 1));

		double gini = computeGINIIndex(data)/data.size();
		node.gini = Math.floor(gini*1000)/1000;
		return node;

	}

	 /**
     * A recursion method used to build one decision tree for boosting.
     * @param data
     * 				the data which is used to grow the tree to the next step
     * @param features
     * 				the features which are available at this step
     * @param depth
     * 				the tree depth at this moment
     * @return		the tree node at this step
     */
	public EnsembleNode Stump(List<EnsembleDataRecord> data, List<EnsembleFeature> features,int depth){
		boolean stoppingCriteriaReached = features.isEmpty() || depth >= 1;
		if (stoppingCriteriaReached) {
			String majorityLabel = getMajorityLabel(data);
			EnsembleNode node = new EnsembleNode(majorityLabel);
			node.leaf = true;
			node.pr = Math.floor(computePRate(data)*1000)/1000;
			node.nr = Math.floor((1 - computePRate(data))*1000)/1000;
			return node;
		}

		EnsembleFeature bestSplit = findBestPenalty(data, features); // get best set of literals
		List<List<EnsembleDataRecord>> splitData = bestSplit.split(data);
		List<EnsembleFeature> newFeatures = new ArrayList<EnsembleFeature>();
		for(int i = 0;i<features.size();i++)
			if(features.get(i)!=bestSplit)
				newFeatures.add(features.get(i));
		EnsembleNode node = new EnsembleNode(bestSplit);
		if(splitData.get(0).isEmpty() || splitData.get(1).isEmpty()){
			node.label = getMajorityLabel(data).contains("1")?true:false;
			node.leaf = true;
			node.pr = Math.floor(computePRate(data)*1000)/1000;
			node.nr = Math.floor((1 - computePRate(data))*1000)/1000;
			return node;
		}
		for (List<EnsembleDataRecord> subsetTrainingData : splitData)  // add children to current node according to split
			node.addChild(Stump(subsetTrainingData, newFeatures, depth + 1));

		double gini = computeGINIIndex(data)/data.size();
		node.gini = Math.floor(gini*1000)/1000;
		return node;
	}

	 /**
     * A method used to find the best split with a certain feature for random forest.
     * @param data
     * 				the data which is would be split
     * @param features
     * 				the features which are available
     * @return		the best feature
     */
	public EnsembleFeature findBestSplitFeatureRF(List<EnsembleDataRecord> data, List<EnsembleFeature> features) {
		double currentImpurity = 1;
		EnsembleFeature bestSplitFeature = null; // rename split to feature
		List<EnsembleFeature> featurelist = splitfeature(10,features);
		for (EnsembleFeature feature : featurelist) {
			List<List<EnsembleDataRecord>> splitData = feature.split(data);
			int datasize = data.size();
			double calculatedSplitImpurity = 0;
			for(int i = 0; i < splitData.size();i++){
				if(!splitData.get(i).isEmpty()){
					double missrate = computeGINIIndex(splitData.get(i))/datasize;
					calculatedSplitImpurity+=missrate;
				}
			}
			if (calculatedSplitImpurity < currentImpurity) {
				currentImpurity = calculatedSplitImpurity;
				bestSplitFeature = feature;
			}
		}
		return bestSplitFeature;
	}
	
	 /**
     * A method used to find the best split with a certain feature for boosting.
     * @param data
     * 				the data which is would be split
     * @param features
     * 				the features which are available
     * @return		the best feature
     */
	public EnsembleFeature findBestPenalty(List<EnsembleDataRecord> data, List<EnsembleFeature> features) {
		double currentImpurity = Double.MAX_VALUE;
		EnsembleFeature bestSplitFeature = null; // rename split to feature
		for (EnsembleFeature feature : features) {
			List<List<EnsembleDataRecord>> splitData = feature.split(data);
			int datasize = data.size();
			double calculatedSplitImpurity = 0;
			for(int i = 0; i < splitData.size();i++){
				if(!splitData.get(i).isEmpty()){
					double penalty = computePenalty(splitData.get(i));
					calculatedSplitImpurity+=penalty;
				}
			}
			if (calculatedSplitImpurity < currentImpurity) {
				currentImpurity = calculatedSplitImpurity;
				bestSplitFeature = feature;
			}
		}
		return bestSplitFeature;
	}
	
	 /**
     *Compute the penalty index within the given data (only used for boosting)
     * @param data
     * 				the given data
     * @return		the penalty index
     */
	public double computePenalty(List<EnsembleDataRecord> data){
		String prediction = getMajorityLabel(data);
		List<String> coldata = new ArrayList<String>();
		String col = data.get(0).label;
		for(int i = 0 ;i < data.size();i++){
			String tmp = data.get(i).getValue(col);
			coldata.add(tmp);
		}
		double penalty = 0;
		for(int i = 0; i < data.size();i++){
			penalty += prediction.equals(coldata.get(i))?0:data.get(i).penalty;
		}
		return penalty;
	}

	 /**
     * Compute the error weight for the training data (only used for boosting)
     * @return		the error weight for this decision stump
     */
	public double getErrorWeight(){
		List<Boolean> prediction = predict(training);
		List<Boolean> actual = actual(training);
		double sum = 0, errorsum = 0;
		for(int i = 0; i < actual.size();i++){
			errorsum += prediction.get(i) == actual.get(i)?0:training.get(i).penalty;
			sum += training.get(i).penalty;
		}
		return errorsum/sum;
	}
	
	 /**
     * A method used to randomly pick up some features among the given ones.
     * @param number
     * 				number of features that are needed
     * @param features
     * 				the features which are available
     * @return		the list of the chosen features
     */
	private List<EnsembleFeature> splitfeature(int number,List<EnsembleFeature> features){
		Random rng = new Random();
		Set<Integer> generated = new LinkedHashSet<Integer>();
		int range = features.size();
		while(generated.size()<number){
			int next = rng.nextInt(range);
			generated.add(next);
		}
		List<EnsembleFeature> subfeatures = new ArrayList<EnsembleFeature>();
		Object[] chosen = generated.toArray();
		for (int i = 0; i < chosen.length; i++)
			subfeatures.add(features.get((int) chosen[i]));
		return subfeatures;
	}

	 /**
     * A method used to find the best split with a certain feature for decision tree and bagging.
     * @param data
     * 				the data which is would be split
     * @param features
     * 				the features which are available
     * @return		the best feature
     */
	public EnsembleFeature findBestSplitFeatureB(List<EnsembleDataRecord> data, List<EnsembleFeature> features) {
		double currentImpurity = 1;
		EnsembleFeature bestSplitFeature = null; // rename split to feature
		for (EnsembleFeature feature : features) {
			List<List<EnsembleDataRecord>> splitData = feature.split(data);
			int datasize = data.size();
			double calculatedSplitImpurity = 0;
			for(int i = 0; i < splitData.size();i++){
				if(!splitData.get(i).isEmpty()){
					double missrate = computeGINIIndex(splitData.get(i))/datasize;
					calculatedSplitImpurity+=missrate;
				}
			}
			if (calculatedSplitImpurity < currentImpurity) {
				currentImpurity = calculatedSplitImpurity;
				bestSplitFeature = feature;
			}
		}
		return bestSplitFeature;
	}

	 /**
     * A method used to compute the GINI Index which represents the purity.
     * @param data
     * 				the given data 
     * @return		the GINI Index
     */
	private double computeGINIIndex(List<EnsembleDataRecord> data){
		double number1 = 0;
		for(int i = 0; i<data.size();i++)
			if(data.get(i).getLabel().equals("1"))
				number1 = number1 + 1;
		number1 = number1 / data.size();
		double GINI = 1.0-number1*number1-(1-number1)*(1-number1);
		return GINI*data.size();
	}
	private double computePRate(List<EnsembleDataRecord> data){
		double number1 = 0;
		for(int i = 0; i<data.size();i++)
			if(data.get(i).getLabel().contains("1"))
				number1 = number1 + 1 ;
		number1 = number1 / data.size();
		return number1;
	}

	 /**
     * The next four functions are used to print out the decision tree
     */
	public void printTree() {
		printSubtree(root);
	}

	public void printSubtree(EnsembleNode node) {
		if (!node.children.isEmpty()) {
			printTree(node.children.get(0), true, "");
		}
		printNodeValue(node);
		if (node.children.size() > 1 ) {
			printTree(node.children.get(1), false, "");
		}
	}

	private void printNodeValue(EnsembleNode node) {
		if (node.leaf) {
			System.out.print(node.getLabel());
		} else {
			System.out.print(node.getName());
		}
		System.out.println();
	}

	private void printTree(EnsembleNode node, boolean isRight, String indent) {
		if (!node.children.isEmpty()) {
			printTree(node.children.get(0), true, indent + (isRight ? "        " : " |      "));
		}
		System.out.print(indent);
		if (isRight) {
			System.out.print(" /");
		} else {
			System.out.print(" \\");
		}
		System.out.print("----- ");
		printNodeValue(node);
		if (node.children.size() > 1) {
			printTree(node.children.get(1), false, indent + (isRight ? " |      " : "        "));
		}
	}
	/*prediction-----------------------------------------*/
	
	 /**
     *Predict the labels of a list of EnsembleDataRecords. 
     *@param test
     *				the given data
     *@return the predicted labels
     */
	public List<Boolean> predict(List<EnsembleDataRecord> test){
		List<Boolean> result = new ArrayList<Boolean>();
		for(int i = 0 ; i < test.size(); i++){
			EnsembleNode level = root;
			EnsembleDataRecord data = test.get(i);
			while (level != null){
				if(level.leaf == true){
					result.add(level.label);
					break;
				}
				String col = level.feature.col;
				EnsembleFeature feature = level.feature;
				if(feature.type ==1){
					if(data.getValue(col).equals(feature.catagory))
						level = level.children.get(0);
					else
						level = level.children.get(1);
					continue;
				}
				if(feature.type == 0){
					if(feature.lower_bound<=Double.valueOf(data.getValue(col))&&Double.valueOf(data.getValue(col))<=feature.higher_bound)
						level = level.children.get(0);
					else
						level = level.children.get(0);
					continue;
				}
				if(data.getValue(col).equals(feature.dummy))
					level = level.children.get(0);
				else 
					level = level.children.get(1);
			}
		}
		return result;
	}

	 /**
     *Predict the label of an EnsembleDataRecords. 
     *@param data
     *				the given data
     *@return the predicted label
     */
	public Boolean predict(EnsembleDataRecord data){
		Boolean result = false;
		EnsembleNode level = root;
		while (level != null){
			if(level.leaf == true){
				result=level.label;
				break;
			}
			String col = level.feature.col;
			EnsembleFeature feature = level.feature;
			if(feature.type == 1){
				if(data.getValue(col).equals(feature.catagory))
					level = level.children.get(0);
				else
					level = level.children.get(1);
				continue;
			}
			if(feature.type == 0){
				if(feature.lower_bound<=Double.valueOf(data.getValue(col))&&Double.valueOf(data.getValue(col))<=feature.higher_bound)
					level = level.children.get(0);
				else
					level = level.children.get(0);
				continue;
			}
			if(data.getValue(col).equals(feature.dummy))
				level = level.children.get(0);
			else 
				level = level.children.get(1);
		}

		return result;
	}
	/*----------------accuracy-----------------------------------------*/
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
     *Get the actual label of the testing data. 
     *@return the list of the actual labels
     */
	public List<Boolean> actual(List<EnsembleDataRecord> testing){
		List<Boolean> actual = new ArrayList<Boolean>();
		for (int i = 0; i < testing.size(); i++){
			if(testing.get(i).getLabel().contains("1"))
				actual.add(true);
			else
				actual.add(false);
		}
		return actual;
	}

	public EnsembleDecisionTree(){
		wholedata = new ArrayList<EnsembleDataRecord>();
		testing = new ArrayList<EnsembleDataRecord>();
		training = new ArrayList<EnsembleDataRecord>();
		features = new ArrayList<EnsembleFeature>();
		actualLabel = new String[2];
		actualLabel[0] = "-1";
		actualLabel[1] = "-1";
	}
	public EnsembleDecisionTree(List<EnsembleDataRecord> training, List<EnsembleDataRecord> testing,List<EnsembleFeature> subfeature){

		this.testing = testing;
		this.training = training;;
		this.features =subfeature;
		actualLabel = new String[2];
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
//	public static void main(String[] argv){
//		EnsembleDecisionTree dt = new EnsembleDecisionTree();
//		dt.loadData("kdd cup.txt");
//		System.out.println();
//		dt.trainB();
//		dt.printTree();
//		List<Boolean> prediction = dt.predict(dt.testing);
//		List<Boolean> actual = dt.actual(dt.testing);
//		System.out.println(dt.accuracy(actual, prediction));
//
//	}

}
