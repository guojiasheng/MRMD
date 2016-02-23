package datamining.xmu.edu.cn;

public class Cosine 
{
	private  int insNum;
	private  int feaNum;
	private  int labNum;
	private  double[] dot;
	private  double[][] data;
	private  double[][] CosineData;
	private  double[] CosineValue;
	
	public Cosine(int feaNum)
	{
		dot = new double[feaNum];
		CosineData = new double[feaNum][feaNum];
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
	
	public void setCosineValue(double[] CosineValue)
	{
		this.CosineValue = CosineValue;
	}
	
//	public double length(double data[][], int col)
//	{
//		double sum = 0.0;
//		for(int i = 0; i < insNum; ++ i)
//		{
//			sum += data[i][col] * data[i][col];
//		}
//		return Math.sqrt(sum);
//	}
	
	public double DotMul(double data[][], int col1, int col2)
	{
		double sum = 0.0;
		for(int i = 0 ; i < insNum; ++ i)
		{
			sum += data[i][col1] * data[i][col2]; 
		}
		return sum;
	}
	
	public double[] run()
	{
		System.out.println("Start to calculate Cosine Distance...");
		double numerator, denominator;
		for(int i = 0; i < feaNum; ++ i)
		{
			dot[i] = DotMul(data, i, i);
		}
		for(int i = 0; i < feaNum; ++ i)
		{
			for(int j = i + 1; j < feaNum; j ++)
			{
				numerator = DotMul(data, i, j);
				denominator = Math.sqrt(dot[i]) * Math.sqrt(dot[j]);
				if(Math.abs(denominator) < (1E-10))
				{
					CosineData[i][j] = CosineData[j][i] = 0.0;
					continue;
				}
				CosineData[i][j] = CosineData[j][i] = numerator/denominator;
			}
		}
		double sum;
		for(int i = 0; i < feaNum; ++ i)
		{
			sum = 0.0;
			for(int j = 0; j < feaNum; ++ j)
			{
				sum += CosineData[i][j];
			}
			CosineValue[i] = sum / feaNum;
		}
		dot = null;
		data = null;
		CosineData = null;
		System.out.println("Calculating Cosine Distance over!!!");
		return CosineValue;
	}

}
