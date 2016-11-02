package datamining.xmu.edu.cn;

public class Pearson 
{
	private  int insNum;
	private  int feaNum;
	private  int labNum;
	private  String inputData[][];
	private  double[][] PearsonData;
	private  double[] PearsonValue;
	
	public Pearson(int feaNum, int labNum)
	{
		PearsonData = new double[feaNum][labNum];
	}
	
	public void setInsNum(int insNum)
	{
		this.insNum = insNum;
	}
	
	public void setFeaNum(int feaNum)
	{
		this.feaNum = feaNum;
	}
	
	public void setLabNum(int labNum)
	{
		this.labNum = labNum;
	}
	
	public void setData(String[][] inputData)
	{
		this.inputData = inputData;
	}
	
//	public void setPearsonData(double[][] PearsonData)
//	{
//		this.PearsonData = PearsonData;
//	}
//	
	public void setPearsonValue(double[] PearsonValue)
	{
		this.PearsonValue = PearsonValue;
	}
	
	public double average(String[][] data, int col)
	{
		double sum = 0.0;
		int num = insNum;
		for(int i = 0; i < insNum; ++ i)
		{
			if(data[i][col].equals("?"))
			{
				num --;
				continue;
			}
			sum += Double.parseDouble(data[i][col]);
		}
		return sum / num;
	}
	
	public double varience(String[][] data, double average1, int col1, double average2, int col2)
	{
		double sum = 0.0;
		int num = insNum;
		for(int i = 0; i < insNum; ++ i)
		{
			if(data[i][col1].equals("?") || data[i][col2].equals("?"))
			{
				num --;
				continue;
			}
			sum += (Double.parseDouble(data[i][col1]) - average1) * (Double.parseDouble(data[i][col2]) - average2);
		}
		return sum / num;
	}
	
	public double[] run()
	{
		System.out.println("Start to calculate Pearson's correlation coefficient...");
		double numerator, denominator;
		
	
		
		for(int i = 0; i < feaNum; ++ i)
		{
			//double r = 0;
//			int k = 0;
			for(int j = feaNum; j < labNum + feaNum; ++ j)
			{
				//System.out.println( average(data, i));
				//System.out.println( average(data, j));

				double average1 = average(inputData, i);
				
				double average2 = average(inputData, j);
				numerator = varience(inputData, average1, i, average2, j);
				denominator = Math.sqrt(varience(inputData, average1, i, average1, i) * varience(inputData, average2, j, average2, j));
				if(Math.abs(denominator) < (1E-10))
				{
					this.PearsonData[i][j - feaNum] = 0.0;
					continue;
				}
				this.PearsonData[i][j - feaNum] = Math.abs(numerator/denominator);
				//System.out.println(PearsonValue[i][k]);
//				k ++;
			}	
		}
		double sum;
		for(int i  = 0; i < feaNum; ++ i)
		{
			sum = 0.0;
			for(int j = 0; j < labNum; ++ j)
			{
				sum += PearsonData[i][j];
			}
			PearsonValue[i] = sum / labNum;
		}
		PearsonData = null;
		inputData = null;
		System.out.println("Calculating Pearson's correlation coefficient over!!!");
		return PearsonValue;
	}
}
