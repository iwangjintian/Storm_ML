package algrithms;



import java.util.HashMap;
import java.util.Map;


/**this EnsembleDataRecord class is the foundation of all the algorithms.
 * It represents the data records when new data come.
 * It contains all the information of the new data.*/
public class EnsembleDataRecord {
	/** the initialization of the data record
	 * @param value
	 * 				array contains the information
	 * @param number
	 * 				number of features*/
	public EnsembleDataRecord(String[] value,int number){
		this.label = String.valueOf(number-1);
		for(int i = 0; i < number; i++){
			values.put(String.valueOf(i), value[i].replaceAll("\\s",""));
		}
	}

	/**return the value in the hashmap
	 * @param column
	 * 				the index
	 * @return the value getting from that index*/
	public String getValue(String column){
		return values.get(column);
	}

	/**return the label in the hashmap
	 * @return the label getting from that hashmap*/
	public String getLabel(){
		return values.get(label);
	}

	/**set the label in the hashmap (when replacing the label)
	 * @param label
	 * 				the value of the label
	 */
	public void setLabel(String label){
		values.put(this.label, label);
	}




	/**the data type that store the record*/
	private Map<String,String> values = new HashMap();
	/**the label of this record*/
	public String label;
	/**the possibility that this record has 1 label (only for naive bayes)*/
	public double pr;
	/**the possibility that this record has 0 label (only for naive bayes)*/
	public double fr;
	/**the penalty index of this record (only for boosting)*/
	public double penalty;
}
