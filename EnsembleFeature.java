package algrithms;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**this EnsembleFeature class is the foundation of all the algorithms.
 * It represents the column information for the coming data.
 * It contains the data type of that column, the value of this column and so on.*/
public class EnsembleFeature {
	/**the name of the feature*/
	public String col;
	/**the upper and lower bound of this feature (only used when the feature type is 0)*/
	public double lower_bound = -2;
	public double higher_bound = -1;
	/**the level of this feature (only used when the type is 1)*/
	public String catagory;
	/**the value of the dummy feature (only used when the type is 2)*/
	public String dummy = "false";
	/**the type of the feature (decided when parsing the feature)*/
	public int type;
	/**the conditional possibility under 1 label (only used when it is naive bayes)*/
	public double pr;
	/**the conditional possibility under 0 label (only used when it is naive bayes)*/
	public double fr;
	public String level;

	/**Initialize the type 0 feature */
	public EnsembleFeature(String col, int lower, int high){
		this.col = col;
		this.lower_bound = lower;
		this.higher_bound = high;
	};

	public EnsembleFeature(){};

	/**Initialize the type 2 feature */
	public EnsembleFeature(String col, String dummy){
		this.col = col;
		this.dummy = dummy;
	};
	
	public static EnsembleFeature newFeature(){
		return new EnsembleFeature();	
	}

	/**split the given data based on this feature
	 * @param data
	 * 				the given data in the form of list of EnsembleDataRecord 
	 * @return the list of list EnsembleDataRecord which have been separated*/
	public List<List<EnsembleDataRecord>> split(List<EnsembleDataRecord> data) {
		List<List<EnsembleDataRecord>> result = new ArrayList<List<EnsembleDataRecord>>();
		Map<Boolean, List<EnsembleDataRecord>> split = new HashMap<Boolean,List<EnsembleDataRecord>>(); 
		List<EnsembleDataRecord> t = new ArrayList<EnsembleDataRecord>();
		List<EnsembleDataRecord> f = new ArrayList<EnsembleDataRecord>();
		for(int i = 0; i < data.size(); i++)
			if(belongsTo(data.get(i)))
				t.add(data.get(i));
			else
				f.add(data.get(i));
		split.put(true, t);
		split.put(false, f);
		if (split.get(true).size() > 0) {
			result.add(split.get(true));
		} else {
			result.add(new ArrayList<EnsembleDataRecord>());
		}
		if (split.get(false).size() > 0) {
			result.add(split.get(false));
		} else {
			result.add(new ArrayList<EnsembleDataRecord>());
		}
		return result;
	}
	
	/**decide whether this data record has the same feature like this
	 * @param sample 
	 * 				the given data
	 * @return the result */
	private boolean belongsTo(EnsembleDataRecord sample){
		if(type == 1){
			if(sample.getValue(col).equals(catagory))
				return true;
			else
				return false;
		}
		if(type == 0)
			if(lower_bound<=Double.valueOf(sample.getValue(col))&&Double.valueOf(sample.getValue(col))<=higher_bound)
				return true;
			else
				return false;
		if(sample.getValue(col).equals(dummy))
			return true;
		else 
			return false;
	}
}
