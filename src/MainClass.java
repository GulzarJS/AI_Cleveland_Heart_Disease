/*
 *  Created by Gulzar Safar on 12/17/2020
 */

public class MainClass {


    public static void main(String[] args) {

        // Copy data from file, shuffle them and write them in 2D array
        float[][] data = DataLib.copyDataToArray("heart_disease_dataset.csv" /*filename*/, ";" /*separator*/);


        NeuralNet nn = new NeuralNet(data, 100 /*batchSize*/, 1 /*nb classes*/,5 /*nb neurons in h1 */, false /* has 1 hidden layer */);


        nn.train(400/*nb of epochs*/);
    }
}
