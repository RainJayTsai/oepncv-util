package org.rainjay.opencv.util;

import java.util.ArrayList;
import java.util.List;

import org.opencv.contrib.Contrib;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;


public class Util {
	public static Mat imagesc( Mat input){
		MinMaxLocResult tmp = Core.minMaxLoc(input);
		System.out.println(input.dump());
		Double scale = 255.0 / (tmp.maxVal - tmp.minVal);
		input.convertTo(input, CvType.CV_8UC1, scale, (0-tmp.minVal)*scale );
		Contrib.applyColorMap(input, input, Contrib.COLORMAP_JET);
		return input;
	}
	
	public static Mat ToColVtr(Mat input){
		return ToColVtr(input,1);
	}
	
	public static Mat ToColVtr(Mat input, int cn){
		input = input.t(); //transpose
		input = input.reshape(cn,1);
		return input;
	}
	
	public static Mat addRow(Mat input, Mat rslt) {
		input = Util.ToColVtr(input);
		rslt.push_back(input);
		return rslt;
	}
	public static Mat dotDivide( Mat dividend, Mat divisor){
		try
		{
			if( dividend.rows() != divisor.rows() || dividend.cols() != divisor.cols())
			{
				throw new Exception("Matrix must be same size!!");
			}
			double[] a,b;
			for( int i = 0; i < dividend.rows(); i++ )
			{
				for( int j = 0; j < divisor.cols(); j++ )
				{
					a = dividend.get(i, j);
					b = divisor.get(i, j);
					dividend.put(i, j, a[0]/b[0] );
				}
			}
		} catch (Exception e)
		{
			// TODO: handle exception
			e.printStackTrace();
		}
		return dividend;
	}
	public static Mat dotDivide(Mat dividend, double divisor){
		double[] a;
		
		for( int i = 0; i < dividend.rows(); i++ )
		{
			for( int j = 0; j < dividend.cols(); j++ )
			{
				a = dividend.get(i, j);
				dividend.put(i, j, a[0]/divisor );
			}
		}
		return dividend;
	}
	
	public static void divide( Mat dividend, double divisor, Mat rslt){
		Mat tmp =  new Mat( new Size(dividend.cols(), dividend.rows()), dividend.type(), new Scalar(divisor));
		Core.divide(dividend, tmp, rslt);
	}
	public static void divide( Mat dividend, double divisor)
	{
		Util.divide(dividend, divisor, dividend);
	}
	
	public static List<Mat> fft(Mat m){
		Mat iMat = new Mat( new Size( m.cols(), m.rows()), m.type(), new Scalar(0) ); //size( col, row )
		List<Mat> list = new ArrayList<Mat>();
		list.add(m);
		list.add(iMat);

		Core.merge(list, m);
		Core.dft(m, m);
		Core.split(m, list);
		return list;
	}
	
	public static Mat nfft(Mat m){
		Mat iMat = new Mat( new Size( m.cols(), m.rows()), m.type(), new Scalar(0) ); //size( col, row )
		List<Mat> list = new ArrayList<Mat>();
		list.add(m);
		list.add(iMat);

		Core.merge(list, iMat);
		Core.dft(iMat, iMat);
		return iMat;
	}
	
	public static void conjTransp(Mat m, Mat dst){
		List<Mat> list = new ArrayList<Mat>();
		list.add( new Mat());
		list.add( new Mat());
		
		Core.split(m, list);
		Mat tmpMat = new Mat(list.get(1).size(), list.get(1).type(), new Scalar(0));
		Core.subtract( tmpMat, list.get(1), tmpMat);
		
		list.set(1, tmpMat);
		Core.merge(list, dst);
		
		dst.t().convertTo(dst, dst.type());
		
	}
	
	public static void invCmplex( Mat m, Mat inverse){
		List<Mat> list = new ArrayList<Mat>();
		list.add(new Mat());
		list.add(new Mat());
		Core.split(m, list);
		
		Mat realMat = list.get(0);
		Mat imagMat = list.get(1);
		
		Mat twicMat = new Mat( m.rows()*2, m.cols()*2, CvType.CV_32FC1 );
		
		if( m.type() == CvType.CV_32FC2 )
		{
			float[] tmp = new float[1];
			
			for( int i = 0; i < m.rows(); i++ ){
				for( int j = 0; j < m.cols(); j++ ){
					realMat.get(i, j,tmp);
					twicMat.put(i, j, tmp);
					twicMat.put(i+m.rows(), j+m.cols(), tmp);
					imagMat.get(i, j,tmp);
					twicMat.put(i, j+m.cols(), tmp);
					twicMat.put(i+m.rows(), j, tmp);
				}
			}
			
			//twicMat = twicMat.inv();
			Core.invert(twicMat, twicMat);
			//realMat = new Mat();
			for( int i = 0; i < m.rows();i++){
				for( int j = 0; j < m.cols(); j++ )
				{
					twicMat.get(i, j,tmp);
					realMat.put(i, j, tmp);
					twicMat.get(i, j+m.cols(),tmp);
					imagMat.put(i,j,tmp);
				}
			}
			Core.subtract(new Mat(m.rows(),m.cols(),CvType.CV_32F,new Scalar(0)), imagMat, imagMat);
			Core.merge(list, inverse);
		}
		else {
			double[] tmp = new double[1];
			for( int i = 0; i < m.rows(); i++ ){
				for( int j = 0; j < m.cols(); j++ ){
					realMat.get(i, j,tmp);
					twicMat.put(i, j, tmp);
					twicMat.put(i+m.rows(), j+m.cols(), tmp);
					imagMat.get(i, j,tmp);
					twicMat.put(i, j+m.cols(), tmp);
					twicMat.put(i+m.rows(), j, tmp);
				}
			}
			
			//twicMat = twicMat.inv();
			Core.invert(twicMat, twicMat);
			//realMat = new Mat();
			for( int i = 0; i < m.rows();i++){
				for( int j = 0; j < m.cols(); j++ )
				{
					twicMat.get(i, j,tmp);
					realMat.put(i, j, tmp);
					twicMat.get(i, j+m.cols(),tmp);
					imagMat.put(i,j,tmp);
				}
			}
			Core.subtract(new Mat(m.rows(),m.cols(),CvType.CV_32F,new Scalar(0)), imagMat, imagMat);
			Core.merge(list, inverse);
		}
		
	}
}
