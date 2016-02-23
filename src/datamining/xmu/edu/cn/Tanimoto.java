package datamining.xmu.edu.cn;

public class Tanimoto 
{
	private  int insNum;
	private  int feaNum;
	private  int labNum;
	private  double[] dot;
	private  double[][] data;
	private  double[][] TanimotoData;
	private  double[] TanimotoValue;
	
	public Tanimoto(int feaNum)
	{
		dot = new double[feaNum];
		TanimotoData = new double[feaNum][feaNum];
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
		
	public void setTanimotoValue(double[] TanimotoValue)
	{
		this.TanimotoValue = TanimotoValue;
	}
	
	public double DotMul(double data[][], int col1, int col2)
	{
		double sum = 0.0;
		for(int i = 0 ; i < insNum; ++ i)
		{
			sum += data[i][col1] * data[i][col2]; 
		}
		return sum;
	}
	
//	public double length(double data[][], int col)
//	{
//		double sum = 0.0;
//		for(int i = 0; i < insNum; ++ i)
//		{
//			sum += data[i][col] * data[i][col];
//		}
//		return sum;
//	}
	
	public double[] run()
	{
		System.out.println("Start to calculate Tanimoto Coefficient...");
		double numerator, denominator;
		for(int i = 0; i < feaNum; ++ i)
		{
			dot[i] = DotMul(data, i, i);
		}
		for(int i = 0; i < feaNum; ++ i)
		{
			for(int j = i + 1; j < feaNum; ++ j)
			{
				numerator = DotMul(data, i, j);
				denominator = (dot[i] + dot[j] - numerator);
				if(Math.abs(denominator) < (1E-10))
				{
					TanimotoData[i][j] = TanimotoData[j][i] = 0.0;
					continue;
				}
				TanimotoData[i][j] = TanimotoData[j][i] =  numerator/denominator; 
			}
		}
		double sum;
		for(int i = 0; i < feaNum; ++ i)
		{
			sum = 0.0;
			for(int j = 0; j < feaNum; ++ j)
			{
				sum += TanimotoData[i][j];
			}
			TanimotoValue[i] = sum / feaNum;
		}
		data = null;
		dot = null;
		TanimotoData = null;
		System.out.println("Calculating Tanimoto Coefficient over!!!");
		return TanimotoValue;
	}	
}
