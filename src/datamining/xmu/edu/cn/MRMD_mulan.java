package datamining.xmu.edu.cn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.j48.*;

public class MRMD_mulan {

	static String [] classAttr;
	static double [][] feaData;
	static String [][] labelData;
	static List<Map.Entry<String, Double>> mrmrList;
	static String inputFile;
	static String outoputFile;
	static int insNum = 0;
	static int feaNum = 0;
	static int labNum = 0;
	static double bestRate=0;
	static double auc=0;
	private static int optNum;
	private static String model="";
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
	// 测试命令	
	//	String x="-i D://mulan.arff -o D://gjs1.arff -sn 2 -ln 1 -df 1 -a D://gjs.arff -model bagging";
	//	args=x.split(" ");
		
	
		if(args.length == 0 ||args[0].equals("-help"))
		{
			System.out.println("Usage: java -jar MRMD.jar -i inputFile -o outputFile -sn seleFeaNum -ln lableNum -df disFunc -a arff -model rf");
			System.exit(0);
		}
		if(args.length != 14)
		{
			System.out.println("\r\nThe number of parameters are not enough  !!!\r\n");
			System.out.println("Usage: java -jar MRMD.jar -i inputFile -o outputFile -sn seleFeaNum -ln lableNum -df disFunc -a arff -model rf");
			System.exit(0);
		}
		inputFile = args[1];          
		outoputFile = args[3];
		model = args[13];
		
		try {
			labNum = Integer.parseInt(args[7]);
		} catch (Exception e) {
			// TODO: handle exception
			System.out.println("\r\nThe parameter of -ln is not a integer !!\r\n");
			System.out.println("Usage: java -jar MRMD.jar -i inputFile -o outputFile -sn seleFeaNum -ln lableNum -df disFunc -a arff -model rf");
			System.exit(0);
		}
		if(labNum < 0)
		{
			System.out.println("\r\nThe parameter of -ln is not available !!\r\n");
			System.out.println("Usage: java -jar MRMD.jar -i inputFile -o outputFile -sn seleFeaNum -ln lableNum -df disFunc -a arff -model rf");
			System.exit(0);
		}
		int seleFeaNum = 0;
		try {
			seleFeaNum = Integer.parseInt(args[5]);
		} catch (Exception e) {
			// TODO: handle exception
			System.out.println("\r\nThe parameter of -sn is error !!\r\n");
			System.out.println("Usage: java -jar MRMD.jar -i inputFile -o outputFile -sn seleFeaNum -ln lableNum -df disFunc -a arff -model rf");
			System.exit(0);
		}
		

		int disFunc = 0;
		try {
			disFunc = Integer.parseInt(args[9]);
		} catch (Exception e) {
			// TODO: handle exception
			System.out.println("\r\nThe parameter of -df is not a integer !!\r\n");
			System.out.println("Usage: java -jar MRMD.jar -i inputFile -o outputFile -sn seleFeaNum -ln lableNum -df disFunc -a arff -model rf");
			System.exit(0);
		}
		
		if(disFunc < 1 || disFunc > 4)
		{
			System.out.println("\r\nThe parameter of -df is error !! df={1, 2, 3, 4}\r\n");
			System.out.println("Usage: java -jar MRMD.jar -i inputFile -o outputFile -sn seleFeaNum -ln lableNum -df disFunc -a arff -model rf");
			System.exit(0);
		}
		
		File InputFile = new File(inputFile);
		if(!InputFile.exists())
		{
			System.out.println("Can't find input file: " + InputFile);
			System.exit(0);
		}
		BufferedReader InputBR = new BufferedReader(new InputStreamReader(new FileInputStream(InputFile), "utf-8"));
		String InputLine = InputBR.readLine();
		String[] dataString;
		while(!InputLine.contains("@DATA") && !InputLine.contains("@data"))
		{
			InputLine = InputBR.readLine();
		}
		InputLine = InputBR.readLine();
		while(InputLine != null)
		{
			if(insNum==1)
				feaNum = InputLine.split(",").length;
			insNum ++;
			InputLine = InputBR.readLine();
		}
		InputBR.close();
		feaNum=feaNum-labNum;
		//insNum--;
		if(seleFeaNum < 0 || seleFeaNum > feaNum)
		{
			System.out.println("\r\nThe parameter of -sn is not available !!\r\n");
			System.out.println("Usage: java -jar MRMD.jar -i inputFile -o outputFile -sn seleFeaNum -ln lableNum -df disFunc -a arff -model rf");
			System.exit(0);
		}
		
		//System.out.println(insNum);
		String [][] inputData = new String[insNum][feaNum + labNum];
		getData gd = new getData();
		gd.setData(inputData);
		gd.setFeaNum(feaNum);
		gd.setInsNum(labNum);
		gd.setLabNum(labNum);
	
		gd.run(inputFile);
		
		double[] PearsonValue = new double[feaNum];
		Pearson pd = new Pearson(feaNum, labNum);
		pd.setInsNum(insNum);
		pd.setFeaNum(feaNum);
		pd.setLabNum(labNum);
		pd.setData(inputData);
		pd.setPearsonValue(PearsonValue);
		pd.run();
		
		labelData = new String[insNum][labNum];
		getLabel gl = new getLabel();
		gl.setData(labelData);
		gl.setFeaNum(feaNum);
		gl.setInsNum(insNum);
		gl.setLabNum(labNum);
		gl.run(inputData);
		
		feaData = new double[insNum][feaNum];
		getFeaData gfd = new getFeaData();
		gfd.setData(feaData);
		gfd.setFeaNum(feaNum);
		gfd.setInsNum(insNum);
		gfd.setLabNum(labNum);
		gfd.run(inputData);
		
		inputData = null;
		
		double[] CosineValue = new double[feaNum];
		Cosine cd = new Cosine(feaNum);
		cd.setCosineValue(CosineValue);
		cd.setData(feaData);
		cd.setFeaNum(feaNum);
		cd.setInsNum(insNum);
		cd.setLabNum(labNum);
//		cd.run();
		
		double[] EuclideanValue = new double[feaNum];
		Euclidean ed = new Euclidean(feaNum);
		ed.setData(feaData);
		ed.setEuclideanValue(EuclideanValue);
		ed.setFeaNum(feaNum);
		ed.setInsNum(insNum);
//		ed.setLabNum(labNum);
		
		double[] TanimotoValue = new double[feaNum];
		Tanimoto td = new Tanimoto(feaNum);
		td.setData(feaData);
		td.setTanimotoValue(TanimotoValue);;
		td.setFeaNum(feaNum);
		td.setInsNum(insNum);
//		td.setLabNum(labNum);
			
		double[] mrmrValue = new double[feaNum];
		switch (disFunc)
		{
			case 1:
				ed.run();
				for(int i = 0; i < feaNum; ++ i)
				{
					mrmrValue[i] = EuclideanValue[i] + PearsonValue[i];
				}
				break;
			case 2:
				cd.run();
				for(int i = 0; i < feaNum; ++ i)
				{
					mrmrValue[i] = CosineValue[i] + PearsonValue[i];
				}
				break;
			case 3:
				td.run();
				for(int i = 0; i < feaNum; ++ i)
				{
					mrmrValue[i] = TanimotoValue[i] + PearsonValue[i];
				}
				break;
			case 4:
				ed.run();
				cd.run();
				td.run();
				for(int i = 0; i < feaNum; ++ i)
				{
					mrmrValue[i] = (PearsonValue[i] * 3 + EuclideanValue[i] + CosineValue[i] + TanimotoValue[i])/3;
				}
				break;
			default:
				break;
		}
		
		
		double max=0;
		for (int i=0;i<mrmrValue.length;i++){
			if (mrmrValue[i]>max){
				max=mrmrValue[i];
			}
		}
		
		for (int i=0;i<mrmrValue.length;i++){
			mrmrValue[i]=mrmrValue[i]/max;
		}
		
		PearsonValue = null;
		EuclideanValue = null;
		CosineValue = null;
		TanimotoValue = null;
		
		Map<String, Double> mrmrMap = new HashMap<String, Double>(); 
		mrmrList = new ArrayList<Map.Entry<String, Double>>(mrmrMap.entrySet());
		mrmrList = initialHashMap(mrmrValue, feaNum);
		BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(outoputFile), false), "utf-8"));
		bufferedWriter.write("The number of selected features is: " + seleFeaNum + "\r\n\r\n");
		bufferedWriter.write("The index of selected features start from 0" + "\r\n\r\n");
		bufferedWriter.write("NO." + "		" + "FeaName" + "		" + "Score" + "\r\n\r\n");
		int line = 1;
		for(int i = 0; i < seleFeaNum; ++ i)
		{
			bufferedWriter.write(line + "		" + mrmrList.get(i).getKey() + "		" + mrmrList.get(i).getValue() + "\r\n");
			line ++;
		}
		bufferedWriter.flush();
		bufferedWriter.close();
		
		
		
		classAttr = new String[labNum];
		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(inputFile)), "utf-8"));
		String lineString = bufferedReader.readLine();
		while(!lineString.contains("@relation") && !lineString.contains("@Relation"))
		{
			lineString = bufferedReader.readLine();
		}
		lineString = bufferedReader.readLine();
		while(lineString.length() == 0)
		{
			lineString = bufferedReader.readLine();
		}
		int count = 0;
		while(count < feaNum)
		{
			if(lineString.length() != 0)
			{
				count ++;
				lineString = bufferedReader.readLine();
			}
			else 
			{
				lineString = bufferedReader.readLine();
			}
		}
		while(lineString.length() == 0)
		{
			lineString = bufferedReader.readLine();
		}
		count = 0;
		while(count < labNum)
		{
			if(lineString.length() != 0)
			{
				
				classAttr[count] = lineString;
				count ++;
				lineString = bufferedReader.readLine();
			}
			else 
			{
				lineString = bufferedReader.readLine();
			}
		}
		bufferedReader.close();
		
		

	
		
		String arff = args[11];
		BufferedWriter arffWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(arff), false), "utf-8"));
		arffWriter.write("@relation " + (new File(inputFile)).getName() + "_feaSele");
		arffWriter.newLine();
		arffWriter.newLine();
		
		for(int i = 1; i <= seleFeaNum; i ++)
		{
			arffWriter.write("@attribute Fea" + i + " numeric");
			arffWriter.newLine();
		}
		for(int i = 0; i < labNum; i ++)
		{
			arffWriter.write(classAttr[i]);
			arffWriter.newLine();
		}
		arffWriter.write("\r\n@data\r\n\r\n");
		
		System.out.println("insNum" + insNum);
		
		for(int i = 0; i < insNum; i ++)
		{
			
			for(int j = 0; j < seleFeaNum; j ++)
			{
//				System.out.println(i);
				arffWriter.write(feaData[i][Integer.parseInt(mrmrList.get(j).getKey().substring(3, mrmrList.get(j).getKey().length()))] + ",");
			}
//			arffWriter.write(feaData[i][Integer.parseInt(mrmrList.get(seleFeaNum - 1).getKey().substring(3, mrmrList.get(seleFeaNum - 1).getKey().length()))] + "\r\n");
			for(int j = 0; j < labNum - 1; j ++)
			{
				arffWriter.write(labelData[i][j] + ",");
			}
			arffWriter.write(labelData[i][labNum - 1]);
			arffWriter.newLine();
		}
		
		arffWriter.flush();
		arffWriter.close();
		System.out.println("MRMD over.");	
		System.out.println("Feature selction optimation begin");	
		System.out.println("model:"+model);
		int num = optSelect();
		writeFeature(num);
		System.out.println("The best feature number: "+String.valueOf(optNum));	
		System.out.println("The best rate: "+String.valueOf(bestRate));	
		System.out.println("Feature selction optimation end,the best arff save in opt.arff");	
		
	}
	
   public static int optSelect() throws Exception{
	   double max=0;
	   optNum=1;
	   double temp=0;
	   for(int i=1;i<= feaNum;i++){
		   try {
			writeFeature(i);
			temp = featureRate();
			System.out.println("feature num: "+i+" rate: "+temp);
			if (temp>max){
				max=temp;
				optNum=i;
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	   }
	   bestRate=max; 
	   writeFeature(optNum);
	   return optNum;
   }	
	
	public static void writeFeature(int seleFeaNum) throws IOException{
		BufferedWriter arffWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File("opt.arff"), false), "utf-8"));
		arffWriter.write("@relation " + (new File(inputFile)).getName() + "_feaSele");
		arffWriter.newLine();
		arffWriter.newLine();
		
		for(int i = 0; i < seleFeaNum; i ++)
		{
			arffWriter.write("@attribute Fea" + i + " numeric");
			arffWriter.newLine();
		}
		for(int i = 0; i < labNum; i ++)
		{
			arffWriter.write(classAttr[i]);
			arffWriter.newLine();
		}
		arffWriter.write("\r\n@data\r\n\r\n");
		
	//	System.out.println("insNum" + insNum);
		
		for(int i = 0; i < insNum; i ++)
		{
			
			for(int j = 0; j < seleFeaNum; j ++)
			{
//				System.out.println(i);
				arffWriter.write(feaData[i][Integer.parseInt(mrmrList.get(j).getKey().substring(3, mrmrList.get(j).getKey().length()))] + ",");
			}
//			arffWriter.write(feaData[i][Integer.parseInt(mrmrList.get(seleFeaNum - 1).getKey().substring(3, mrmrList.get(seleFeaNum - 1).getKey().length()))] + "\r\n");
			for(int j = 0; j < labNum - 1; j ++)
			{
				arffWriter.write(labelData[i][j] + ",");
			}
			arffWriter.write(labelData[i][labNum - 1]);
			arffWriter.newLine();
		}
		
		arffWriter.flush();
		arffWriter.close();
	}
	
	
	public static Classifier getClassifier(String name){
		Classifier classify = null;
		switch (name){
			case "rf":
				 classify=new RandomForest();
				 break;
			case "svm":
				 classify=new LibSVM();
				 break;
			case "bagging":
				 classify=new Bagging();
				 break;
			default:
				System.out.println("目前只支持rf(randomForst),svm,bagging");
		}
		return classify;
	}
	
	
	public static double featureRate() throws Exception{
		FileReader reader=new FileReader("opt.arff");
		Instances  data=new Instances(reader);
		data.setClassIndex(data.numAttributes()-1);//设置训练集中，target的index  
		Classifier classify=getClassifier(model);
		classify.buildClassifier(data);
		Evaluation eval=new Evaluation(data);
		
		if(insNum>10){
			eval.crossValidateModel(classify, data,10, new Random());
		}
		else{
			eval.crossValidateModel(classify, data,insNum, new Random());
		}
		eval.evaluateModel(classify, data);
		return 1-eval.errorRate();
		//System.out.println(String.valueOf(1-eval.errorRate()));
		//System.out.println(eval.toSummaryString());
		//System.out.println(eval.toMatrixString());
	}

	public static List initialHashMap(double data[], int feaNum)
	{
		Map<String, Double> mrmrMap = new HashMap<String, Double>(); 
		for(int i = 0; i < feaNum; ++ i)
		{
			mrmrMap.put("Fea" + i, data[i]);
		}
		
		List<Map.Entry<String, Double>> mrmrList = new ArrayList<Map.Entry<String, Double>>(mrmrMap.entrySet());
		Collections.sort(mrmrList, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2)
			{
//				return (o1.getValue()).compareTo(o2.getValue());
				if(Double.parseDouble(o1.getValue().toString()) < Double.parseDouble(o2.getValue().toString()))
				{
					return 1;
				}
				else if(Double.parseDouble(o1.getValue().toString()) == Double.parseDouble(o2.getValue().toString()))
				{
					return 0;
				}
				else {
					return -1;
				}
			}
			
		});
		return mrmrList;
	}
}
