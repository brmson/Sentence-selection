/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package yodaqass;

//import jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jblas.DoubleMatrix;

/**
 *
 * @author silvicek
 */

class Mb{
    private final DoubleMatrix M;
    private final double b;
    private final String path="data/Mb.txt";
    
    public Mb(){
        DoubleMatrix M=DoubleMatrix.zeros(50,50);
        double b=0;
        File f=new File(path);
        FileReader fr = null;
        try {
            fr = new FileReader(f);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(YodaqaSS.class.getName()).log(Level.SEVERE, null, ex);
        }
        BufferedReader br=new BufferedReader(fr);
        String line;
        try {
            for(int j=0;j<50;j++){
                line=br.readLine();
                String [] numbers=line.split(" ");
                for(int i=0;i<50;i++){
                    M.put(j,i, Double.parseDouble(numbers[i]));
                }
            }
            line=br.readLine();
            b=Double.parseDouble(line);
        } catch (IOException ex) {
            Logger.getLogger(YodaqaSS.class.getName()).log(Level.SEVERE, null, ex);
        }
        this.M=M;
        this.b=b;
    }
    
    public DoubleMatrix getM() {
        return M;
    }

    public double getB() {
        return b;
    }
    
}


class Dictionary{
    private final Map<String,double[]> dictionary;
    private final String path="data/glovewiki.txt";
    
    public Dictionary(){
        Map<String,double[]> dictionary=new HashMap<>();
        FileReader fr=null;
        try {
            File f=new File(path);
            fr = new FileReader(f);
            BufferedReader br=new BufferedReader(fr);
            
            String line;
            while((line=br.readLine())!=null){
                String[] s=line.split(" ");
                String word=s[0];
                double[] gword=new double[s.length-1];
                for(int i=1;i<s.length;i++){
                    gword[i-1]=Double.parseDouble(s[i]);
                }
                dictionary.put(word, gword);
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(YodaqaSS.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(YodaqaSS.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                fr.close();
            } catch (IOException ex) {
                Logger.getLogger(YodaqaSS.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        System.out.println("GloVe dictionary created");
        this.dictionary=dictionary;
    }

    public double[] get(String key){
        return dictionary.get(key);
    }
}


class W{
    private final DoubleMatrix w;
    private final String path="data/weights.txt";
    
    public W(){
        DoubleMatrix w=DoubleMatrix.zeros(4,1);
        File f=new File(path);
        FileReader fr = null;
        try {
            fr = new FileReader(f);
        } catch (FileNotFoundException ex) {
            Logger.getLogger(YodaqaSS.class.getName()).log(Level.SEVERE, null, ex);
        }
        BufferedReader br=new BufferedReader(fr);
        String line;
        try {
            int i=0;
            while((line=br.readLine())!=null){
                    w.put(i, Double.parseDouble(line));
                    i++;
            }
        } catch (IOException ex) {
            Logger.getLogger(YodaqaSS.class.getName()).log(Level.SEVERE, null, ex);
        }
        this.w=w;
    }
    
    public DoubleMatrix getW(){
        return this.w;
    }
}

class Relatedness{
    
    private Mb mb;
    private Dictionary dict;
    
    public Relatedness(){
        this.mb=new Mb();
        this.dict=new Dictionary();
    }
    
    public double probability(List<String> q,List<String> a){
        return probability(gloveBOW(q,this.dict),gloveBOW(a,this.dict));   
    }
    
    /** Counts probability from q,a glove vectors. */
    private double probability(DoubleMatrix q,DoubleMatrix a){
        DoubleMatrix M=this.mb.getM();
        double b=this.mb.getB();
        return 1/(1+Math.exp(-z(q,M,a,b)));
    }
    
    /** Returns glove vectors from input sentence, dictionary. Uses box-of-words approach. */
    private DoubleMatrix gloveBOW(List<String> words,Dictionary dict){
        DoubleMatrix bow = DoubleMatrix.zeros(50,1);
        int w=0;
        for(int i=0;i<words.size();i++){
            double[] x=dict.get(words.get(i));
            if(x!=null){
                DoubleMatrix glove=new DoubleMatrix(x);
                w++;
                bow=bow.add(glove);
            }
        }
        if(w!=0)bow=bow.div(w);
//        System.out.println(words+" = "+bow);
        return bow;
    }
     /** qTMa+b. */
     private double z(DoubleMatrix q,DoubleMatrix M,DoubleMatrix a,double b){
        return q.transpose().mmul(M).mmul(a).get(0)+b;
    }
    
}


class Probability{
    private W w;
    private Relatedness r;
    private Map<String,Double> idf;
    
    public Probability(){
        this.w=new W();
        this.r=new Relatedness();
    }
    
        /** Counts probability using word counts. */
    public double probability(List<String> qtext,List<String> atext){
        double p1=r.probability(qtext, atext);
        System.out.println("p1="+p1);
        double count=0;
        double idfcount=0;
        for(String s:atext){
            double f=Collections.frequency(qtext, s);
            count+=f;
            if(f>0)idfcount+=Math.log(idf.get(s)/f);
        }
        DoubleMatrix x=new DoubleMatrix(new double[][] {{p1,count,idfcount,1}});
        return 1/(1+Math.exp(-x.mmul(w.getW()).get(0)));
    }
    
    
    /** Returns map of word counts. */  ////=null?
    public void setidf(List<String> words){
        Map<String,Double> idf=new HashMap<>();
        for(String word:words){
            if(idf.containsKey(word)){
                idf.put(word, idf.get(word)+1);
            }else{
                idf.put(word, 1.0);
            }
        }
        this.idf=idf;
    }
    
}

public class YodaqaSS {
    
    static List<String> split(String sentence){
        String[]x=sentence.toLowerCase().split(" ");
        List<String> a=new ArrayList<>();
        for(String s:x){
            a.add(s);
        }
        return a;
    }
    
    
    
    public static void main(String[] args) {
            List<String> q=new ArrayList<>();
            q.add("what");
            q.add("do");
            q.add("practitioners");
            q.add("of");
            q.add("wicca");
            q.add("worship");
            q.add("?");
            
            List<String> a=split("An estimated 50,000 Americans practice Wicca , a form of polytheistic nature worship");
            Probability p=new Probability();
            p.setidf(a);
            System.out.println(p.probability(q, a));
    }
    
}
