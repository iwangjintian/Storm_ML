package algrithms;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.StormSubmitter;
import org.apache.storm.generated.AlreadyAliveException;
import org.apache.storm.generated.AuthorizationException;
import org.apache.storm.generated.InvalidTopologyException;
import org.apache.storm.generated.StormTopology;
import org.apache.storm.kafka.BrokerHosts;
import org.apache.storm.kafka.KafkaSpout;
import org.apache.storm.kafka.SpoutConfig;
import org.apache.storm.kafka.StringScheme;
import org.apache.storm.kafka.ZkHosts;
import org.apache.storm.spout.SchemeAsMultiScheme;
import org.apache.storm.task.OutputCollector;
import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.BasicOutputCollector;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.topology.base.BaseBasicBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;
import org.apache.storm.utils.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import Jama.Matrix;
import algrithms.Knn.sortelement;






public class KnnTopology implements Serializable{

	private static final Logger LOG = LoggerFactory.getLogger(KnnTopology.class);
	private static Knn knn;
	private static int k;
	private static int number = 0;
	
	/** TrainBolt is used to handle the training data sent from the Slinkbolt.
	 * Once the amount of the training data records reaches a certain number, it would start to build the model.
	 * Finishing the model, it would print out the accuracy of this model. */
	public static class TrainBolt extends BaseBasicBolt {
		private int cnt = 0;
		public TrainBolt(){
		}

		@Override
		public void prepare(java.util.Map stormConf, TopologyContext context){
		knn = new Knn();
		k = 5;
		}
		@Override
		public void execute(Tuple tuple, BasicOutputCollector collector) {
			// TODO Auto-generated method stub

			String str = (String)tuple.getValueByField("field_train");
			knn.loadData(str);
			cnt++;
			System.out.println("c n t:"+cnt);
			System.out.println("DATA REC "+knn.wholedata.size()+": "+str);
			if(knn.wholedata.size() % 600 == 0){				//LR.weights = null;
				System.out.println("finish collecting training data. data record read "+ knn.wholedata.size());
				knn.split(80); //split the data when finishing collecting
				knn.replacelabel(number); //unify the label 
				knn.parseEnsembleFeature(); //parse the data to get feature information
				knn.readMatrix(); //transfer the training data into the form of matrix
				System.out.println("training regression model . . . done!");
				Matrix testing = knn.readtestingMatrix(); // transfer the testing data into the form of matrix
				int[] predict = knn.classify(testing, 5); //classify the testing data with k equal to 5
				List<String> testingtar = knn.gettestingcoldata(String.valueOf(number-1)); // get the actual labels of testing data
				int[] testingtarget = new int[testingtar.size()];
				for(int i = 0;i<testingtar.size();i++)
			    	testingtarget[i] = Integer.valueOf(testingtar.get(i).replaceAll("\\s", ""));
				System.out.println("The Accuracy is ...: "+knn.accurate(testingtarget,predict)*100); // print out the accuracy
				System.out.println(knn.training.size() + "+" +knn.testing.size());				
			}
			collector.emit(new Values(str));
		}
		
		
		@Override
		public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {


		}

	}

	/** TestBolt is used to handle the testing data sent from the Slinkbolt.
	 * If the model is ready, TestBolt would predict the label of the coming record according to the model.
	 * And if the model is not ready, TestBolt would start to train the model with current data. Then give out the prediction
	 */
	public static class TestBolt extends BaseBasicBolt {

		@Override
		public void execute(Tuple input, BasicOutputCollector collector) {
			// TODO Auto-generated method stub
			String str = (String)input.getValueByField("field_validation");
			String seperate = ",";
			String[] value = str.split(seperate);
			if(value.length == number - 1){
				if(knn.actualLabel[0].equals(knn.actualLabel[1])){
					System.out.println("finish collecting training data. data record read "+ knn.wholedata.size());
					knn.split(80); //split the data when finishing collecting
					knn.replacelabel(number); //unify the label 
					knn.parseEnsembleFeature(); //parse the data to get feature information
					knn.readMatrix(); //transfer the training data into the form of matrix
					System.out.println("training regression model . . . done!");
					Matrix testing = knn.readtestingMatrix(); // transfer the testing data into the form of matrix
					int[] predict = knn.classify(testing, 5); //classify the testing data with k equal to 5
					List<String> testingtar = knn.gettestingcoldata(String.valueOf(number-1)); // get the actual labels of testing data
					int[] testingtarget = new int[testingtar.size()];
					for(int i = 0;i<testingtar.size();i++)
				    	testingtarget[i] = Integer.valueOf(testingtar.get(i).replaceAll("\\s", ""));
					System.out.println("The Accuracy is ...: "+knn.accurate(testingtarget,predict)*100); // print out the accuracy
					System.out.println(knn.training.size() + "+" +knn.testing.size());	
				}
				System.out.println("we got the test request!");
				EnsembleDataRecord record = new EnsembleDataRecord(value,value.length);// load the coming data record
				double[] dtype = knn.readRecord(record); // parsing it into the form of array
				double[] dist = knn.distance(dtype); // calculate the distance of this new coming data record with all the data records in training
				sortelement[] dCollection = new sortelement[dist.length];
				for(int j = 0;j<dCollection.length;j++)
			    	dCollection[j] = knn.new sortelement(dist[j],Integer.valueOf(knn.training.get(j).getLabel().replaceAll("\\s", "")));
				Arrays.sort(dCollection); // get the 5-nearest neighbors
				int sum = 0;
				int index = k>knn.training.size()?knn.training.size():k;
				for(int i = 0; i < index; i++ )
					sum+=dCollection[i].label;
				String ll = (sum > k/2)?knn.actualLabel[0]:knn.actualLabel[1]; // take the majority vote
				System.out.println("predicted result:"+ ll);


			}
		}

		@Override
		public void declareOutputFields(OutputFieldsDeclarer declarer) {
			// TODO Auto-generated method stub

		}

	}

	/** SlinkBolt is used to distribute the training and testing data.
	 * It would get the number of the features of the data record and send it to the corresponding bolt.
	 */
	public static class SlinkBolt extends BaseBasicBolt {		
		public SlinkBolt(){}

		@Override
		public void execute(Tuple tuple, BasicOutputCollector collector) {
			// TODO Auto-generated method stub
			String str = tuple.getString(0);
			String seperate = ",";
			String[] value = str.split(seperate);
			if(value.length > number)
				number = value.length;
			if(value.length == number)
				collector.emit("train",new Values(str));
			else
				collector.emit("validation", new Values(str));
		}
		@Override
		public void declareOutputFields(OutputFieldsDeclarer outputFieldsDeclarer) {
			outputFieldsDeclarer.declareStream("train", new Fields("field_train"));
			outputFieldsDeclarer.declareStream("validation", new Fields("field_validation"));

		}

	}
	public static void main(String[] args) throws AlreadyAliveException, InvalidTopologyException, AuthorizationException{
		BrokerHosts hosts = new ZkHosts("ubuntu:2181");
		String topic="0807";
		SpoutConfig kafkaConf= new SpoutConfig(hosts, topic,"/"+topic,UUID.randomUUID().toString());
		kafkaConf.zkRoot="/"+topic;
		kafkaConf.scheme = new SchemeAsMultiScheme(new StringScheme());

		KafkaSpout kafkaspout = new KafkaSpout(kafkaConf);
		TopologyBuilder builder = new TopologyBuilder();
		builder.setSpout("kafkaspout", kafkaspout,1);
		builder.setBolt("slink", new SlinkBolt(),1).shuffleGrouping("kafkaspout");
		builder.setBolt("train", new TrainBolt(),1).shuffleGrouping("slink","train");
		builder.setBolt("test", new TestBolt(),1).shuffleGrouping("slink","validation");



		Config conf = new Config();
		conf.setDebug(true);
		if (args != null && args.length > 0) {
			conf.setNumWorkers(1);
			StormSubmitter.submitTopologyWithProgressBar(args[0], conf, builder.createTopology());
		} else {
			LocalCluster cluster = new LocalCluster();
			StormTopology topology = builder.createTopology();
			cluster.submitTopology("test", conf, topology);
			Utils.sleep(40000);
			cluster.killTopology("test");
			cluster.shutdown();
		}
	}

}
