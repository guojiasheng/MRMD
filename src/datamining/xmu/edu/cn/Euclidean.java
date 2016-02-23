package datamining.xmu.edu.cn;

public class Euclidean 
{
	private  int insNum;
	private  int feaNum;
	private  int labNum;
	private  double[][] data;
	private  double[][] EuclideanData;
	private  double[] EuclideanValue;
	
	public Euclidean(int feaNum)
	{
		EuclideanData = new double[feaNum][feaNum];
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
	
	public void setData(double[][] data)
	{
		this.data = data;
	}
	
	public void setEuclideanValue(double[] EuclideanValue)
	{
		this.EuclideanValue = EuclideanValue;
	}
	
	public double calc(double data[][], int col1, int col2)
	{
		double sum = 0.0;
		for(int i = 0; i < insNum; ++ i)
		{
			sum += (data[i][col1] - data[i][col2]) * (data[i][col1] - data[i][col2]);
		}
		return Math.sqrt(sum);
	}
	
	public double[] run()
	{
		System.out.println("Start to calculate Euclidean Distance...");
		for(int i = 0; i < feaNum; ++ i)
		{
			for(int j = i + 1; j < feaNum; ++ j)
			{
				EuclideanData[i][j] =  calc(data, i, j);
				EuclideanData[j][i] = EuclideanData[i][j];
			}
		}
		double sum;
		for(int i = 0; i < feaNum; ++ i)
		{
			sum = 0.0;
			for(int j = 0; j < feaNum; ++ j)
			{
				sum += EuclideanData[i][j];
			}
			EuclideanValue[i] = sum / feaNum;
		}
		data = null;
		EuclideanData = null;
		System.out.println("Calculating Euclidean Distance over!!!");
		return EuclideanValue;
	}
}
