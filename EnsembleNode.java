package algrithms;

import java.util.ArrayList;
import java.util.List;

/** EnsembleNode is the node of the tree. It is used to build a decision tree.*/
public class EnsembleNode {
	/**true positive rate*/
	public double pr;
	/**true negative rate*/
	public double nr;
	/**whether this node is a leaf node*/
	public boolean leaf;
	/**the label of this node*/
	public boolean label;
	/**the gini index of this node representing the purity*/
	public double gini;
	/**the feature that creates this node*/
	public EnsembleFeature feature;
	/**the children of the node*/
	public List<EnsembleNode> children = new ArrayList<EnsembleNode>();
	public EnsembleNode(EnsembleFeature f, double pr, double nr){
		this.feature = f;
		this.pr = pr;
		this.nr = nr;
	}
	public EnsembleNode(EnsembleFeature f){
		this.feature = f;
		this.pr = 0;
		this.nr = 0;
	}
	public EnsembleNode(String label){
		if(label.contains("1"))
			this.label = true;
		else
			this.label = false;
	}
	public static EnsembleNode newNode(){
		return new EnsembleNode(EnsembleFeature.newFeature());
	}
	
	/**add a children*/
	public void addChild(EnsembleNode child){
		this.children.add(child);
	}
	
	/**get the label (this function is only used in the printing tree process)*/
	public String getLabel(){
		
		return "["+this.label+"]"+"  "+ pr+" "+nr;
	}
	
	/**get the full name of this node (this function is only used in the printing tree process)*/
	public String getName(){
		if(feature.type == 1)
			 return feature.col + " " + "=" +" " + feature.catagory + "  "+gini;
		 if(feature.type == 0)
			return feature.lower_bound + " " + "<"+ " "+feature.col + " " + "<" +" " + (int)(feature.higher_bound*100)/((double)100)+ "  "+gini;
		return feature.col+ " "+"="+" "+feature.dummy+ "  "+gini;
		
	}
}
