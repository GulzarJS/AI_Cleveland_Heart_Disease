/*
 *  Created by Gulzar Safar on 12/17/2020
 */

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class DataLib {


    private static ArrayList<ArrayList<String>> data = new ArrayList<ArrayList<String>>();

    /**
     * Read data from csv file
     * @param filename
     * @param separator
     * @return 2D arraylist
     */
    private static ArrayList<ArrayList<String>> importCSV(String filename, String separator) {
        String line = "";
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            while ((line = br.readLine()) != null){
                ArrayList<String> dataAsString = new ArrayList<String>(Arrays.asList(line.split(separator)));
                data.add(dataAsString);
            }
            br.close();
        }catch(IOException e) {e.printStackTrace();}
        return data;
    }

    /**
     * Read data from CSV file and return them, shuffled, in an 2D array
     * @param fileName
     * @param separator
     * @return 2D array
     */
    public static float[][] copyDataToArray(String fileName, String separator){
        importCSV(fileName, separator);
        return shuffleData();
    }

    /**
     * Shuffle data in 2D array
     * @return 2D array
     */
    private static float[][] shuffleData(){
        int nbInstances = data.size();
        int nbFeatures  = data.get(0).size();
        // Shuffle the data
        Collections.shuffle(data);
        float[][] arrayData = new float[nbInstances][nbFeatures];
        for (int i = 0; i < nbInstances; i++) {
            for (int j = 0; j < nbFeatures; j++)
                arrayData[i][j] = Float.parseFloat(data.get(i).get(j));
        }
        return arrayData;
    }

    public static float[][] tanh(float[][] Z){
        float[][] activations = new float[Z.length][Z[0].length];
        for (int i = 0; i < Z.length; i++)
            for (int j = 0; j < Z[0].length; j++)
                activations[i][j] = (float) (2.f/(1+Math.exp(-2.f*Z[i][j])) - 1);
        return activations;
    }

    /**
     * Print data
     */
    public static void printData(){
        int i = 0;
        for (ArrayList<String> entry : data){
            System.out.println("["+ i++ +"]"+entry);
        }
    }

    /**
     * Write data to csv file
     * @param filename
     * @param data
     */
    public static void exportDataToCSV(String filename, String data) {
        try {
            Writer out = new BufferedWriter(new FileWriter(filename));
            out.write(data);
            out.close();
        } catch (IOException e) { e.printStackTrace(); }
    }

}
