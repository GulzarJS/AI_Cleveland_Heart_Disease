/*
 *  Created by Gulzar Safar on 12/17/2020
 */

import java.util.Arrays;

public class NeuralNet {

    private float[][] trainingData;
    private float[][] testingData;
    private int nbInstances;
    private int nbFeatures;
    private int nbClasses;
    private int batchSize;

    private float[][] X_train, Y_train, X_test, Y_test, W1, W2, b1, b2, Wmid, bmid;
    private float eta = 0.01f;

    // has 1 hidden layer or not
    private boolean Has1HLayer;


    /**
     * Class constructor
     *
     * @param data       data read from input file
     * @param batchSize  number of instances in each batch for training
     * @param K          number of output class (e.g. Iris dataset => 3)
     * @param h1Size     number of neurons in hidden layer 1
     * @param Has1HLayer has 1 hidden layer or not
     */
    public NeuralNet(float[][] data, int batchSize, int K /*nb classes*/, int h1Size, boolean Has1HLayer) {

        nbInstances = data.length;
        nbFeatures = data[0].length;
        this.batchSize = batchSize;
        this.nbClasses = K;
        this.Has1HLayer = Has1HLayer;
        int h1S = h1Size;

        // age, resting_blood_pressure, cholesterol and max_heart_rate_achieved age, resting_blood_pressure, cholesterol and max_heart_rate_achieved => non categorical attributes
        int[] indices = {0, 3, 4, 7, 9};

        int trainingSize = (int) (nbInstances * 0.75);
        int testingSize = nbInstances - trainingSize;
        trainingData = new float[trainingSize][nbFeatures];
        testingData = new float[testingSize][nbFeatures];

        // Copy data into training & testing set
        for (int i = 0; i < trainingSize; i++)
            for (int j = 0; j < data[0].length; j++)
                trainingData[i][j] = data[i][j];

        for (int i = 0; i < testingSize; i++)
            for (int j = 0; j < data[0].length; j++)
                testingData[i][j] = data[i + trainingSize][j];

        // Normalizing the data sets
        NNLib.normalize(trainingData, indices);
        NNLib.normalize(testingData, indices);

        X_train = new float[this.nbFeatures - 1][this.batchSize];  // -1 because the last column is the label
        Y_train = new float[this.nbClasses][this.batchSize];

        X_test = new float[this.nbFeatures - 1][this.batchSize];
        Y_test = new float[this.nbClasses][this.batchSize];
        W1 = new float[h1S][13];
        NNLib.initMatrix(W1);
        b1 = new float[h1S][1];
        for (int i = 0; i < b1.length; i++) Arrays.fill(b1[i], 0.f);
        W2 = new float[1][h1S];
        NNLib.initMatrix(W2);
        b2 = new float[1][1];
        for (int i = 0; i < b2.length; i++) Arrays.fill(b2[i], 0.f);

        // In do not have 1 hidden layer
        if (!Has1HLayer) {
            Wmid = new float[h1S][h1S];
            NNLib.initMatrix(Wmid);
            bmid = new float[h1S][1];
            W2 = new float[1][h1S];
            NNLib.initMatrix(W2);
            b2 = new float[1][1];
            for (int i = 0; i < b2.length; i++) Arrays.fill(b2[i], 0.f);
        }
    }

    @SuppressWarnings("unused")
    private void printData(float[][] data) {
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++)
                System.out.print(data[i][j] + " ");
            System.out.println();
        }
    }

    /**
     * Create one-hots vector for classification
     *
     * @param k target      class to set to 1
     * @param Y matrix of one-hots
     */
    private void createOneHot(int k, int indexInBatch, float[][] Y) {
        if (k > Y.length  ) throw new RuntimeException("There is only " + Y.length + " possible classes");

        Y[0][indexInBatch] = k;

    }

    /**
     * Load one instance of data
     * Separate features and labels #dataIndex in X and Y
     *
     * @param dataSet   training or testing
     * @param X         matrix for inputs
     * @param Y         matrix for outputs
     * @param dataIndex index in dataSet of data to copy
     * @param batchSize number of instances in each batch
     */
    private void loadAttributesAndLabels(float[][] dataSet, float[][] X, float[][] Y, int dataIndex, int batchSize) {
        // 1. Load data attributes in X
        int k = 0; // index to load the features (data[ex][i] => feature #i of instance #ex)
        int indexInBatch = dataIndex % batchSize;
        for (k = 0; k < dataSet[indexInBatch].length - 1; k++) {
            X[k][indexInBatch] = dataSet[dataIndex][k];
        }

        // 2. Load data labels in Y (create a one-hot for class prediction)
        createOneHot((int) dataSet[dataIndex][k], indexInBatch, Y);
    }

    /**
     * Shuffle the set of training data
     * Fisher–Yates shuffle :
     * https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
     * Note : NNLib.rnd.nextInt(max) returns an integer in [0;max[
     */
    private void shuffleTrainingData() {
        int j;
        float[] t;
        for (int i = 0; i < trainingData.length; i++) {
            j = (int) Math.random() * i;
            t = new float[trainingData.length];
            t = trainingData[j];
            trainingData[j] = trainingData[i];
            trainingData[i] = t;
        }
    }


    /**
     * Feed the testing set to the model and measure error / accuracy
     *
     * @returns cost on testing set
     */
    public float testPrediction() {
        //int currentDataIndex = 0;
        float cost = 0.f;
        int countPredictions = 0, TP = 0, FP = 0, FN = 0, TN = 0;

        for (int i = 0; i < testingData.length; i++)
            loadAttributesAndLabels(testingData, X_test, Y_test, i, batchSize);

        // Feed-Forward
        float[][] Z1, A1, Z2, A2, Zmid, Amid;
        Z1 = NNLib.addVec(NNLib.mult(W1, X_test), b1);
        A1 = NNLib.relu(Z1);
        if (this.Has1HLayer) {
            Z2 = NNLib.addVec(NNLib.mult(W2, A1), b2);
        } else {
            Zmid = NNLib.addVec(NNLib.mult(Wmid, A1), bmid);
            Amid = NNLib.relu(Zmid);
            Z2 = NNLib.addVec(NNLib.mult(W2, Amid), b2);
        }
        A2 = NNLib.sigmoid(Z2);

        // Calculations
        int batchSize = A2[0].length;
        int K = Y_test.length; // # classes
        for (int k = 0; k < K; k++)
            for (int c = 0; c < batchSize; c++) {
                cost += Y_test[k][c] * Math.log(A2[k][c]);
                if (NNLib.checkPrediction(A2, Y_test, c)) {
                    countPredictions++;

                }
            }
        cost = -(1.f / batchSize) * cost;
        float accuracy = 100.f * countPredictions / (K * batchSize);
        System.out.println("  Cost of test data: "  + cost);
        System.out.println("  Accuracy of test data: " + accuracy);

        return cost;
    }


    /**
     * Perform 1 epoch of training
     *  1/ load a minibatch of data into X_train and Y_train
     *  2/ forward pass
     *  3/ compute the gradient of the loss
     *  4/ update the parameters
     */
    private float trainingEpoch(){
        int seenTrainingData = 0;
        shuffleTrainingData(); // shuffle the data before training
        float [][] Z1,A1,Z2,A2,Amid = null,Zmid,delta1,delta2,deltamid,dW2,dW1,dWmid = null,db1,db2,dbmid = null;


        for(;seenTrainingData<trainingData.length;seenTrainingData++) {
            for(int i=0;i<trainingData.length;i++) {
                loadAttributesAndLabels(trainingData, this.X_train, this.Y_train,i ,this.batchSize);
            }

            //Feed-Forward propagation

            Z1=NNLib.addVec(NNLib.mult(W1, X_train), b1);
            A1=NNLib.relu(Z1);
            if(this.Has1HLayer) {
                Z2=NNLib.addVec(NNLib.mult(W2, A1), b2);
            }else {
                Zmid=NNLib.addVec(NNLib.mult(Wmid, A1), bmid);
                Amid=NNLib.relu(Zmid);
                Z2=NNLib.addVec(NNLib.mult(W2, Amid), b2);
            }

            A2=NNLib.sigmoid(Z2);

            //Backward Propagation
            delta2=NNLib.subtract(A2,Y_train);
            if(this.Has1HLayer) {
                dW2=NNLib.mult(NNLib.mult(delta2,NNLib.transpose(A1)), 1.0f/(float)trainingData.length);
                delta1=NNLib.hadamard(NNLib.mult(NNLib.transpose(W2), delta2), NNLib.reluDeriv(A1));
            }else {
                dW2=NNLib.mult(NNLib.mult(delta2,NNLib.transpose(Amid)), 1.0f/(float)trainingData.length);
                deltamid=NNLib.hadamard(NNLib.mult(NNLib.transpose(W2), delta2), NNLib.reluDeriv(Amid));
                dWmid=NNLib.mult(NNLib.mult(deltamid,NNLib.transpose(A1)), 1.0f/(float)trainingData.length);
                dbmid=deltamid;
                delta1=NNLib.hadamard(NNLib.mult(NNLib.transpose(Wmid), deltamid), NNLib.reluDeriv(A1));
            }
            db2=delta2;
            dW1=NNLib.mult(NNLib.mult(delta1,NNLib.transpose(X_train)), 1.0f/(float)trainingData.length);
            db1=delta1;
            if(seenTrainingData==0) {
                int t=0;
                for(int i=0;i<Y_train[0].length;i++) {
                    if(NNLib.checkPrediction(A2,Y_train,i))
                        t++;
                }
            }


            // Updating Variables
            W2=NNLib.subtract(W2, NNLib.mult(dW2, this.eta));
            b2=NNLib.subtract(b2, NNLib.mult(db2, this.eta));
            W1=NNLib.subtract(W1, NNLib.mult(dW1, this.eta));
            b1=NNLib.subtract(b1, NNLib.mult(db1, this.eta));

            // If there is hidden layer, update middle variables
            if(!this.Has1HLayer) {
                Wmid=NNLib.subtract(Wmid, NNLib.mult(dWmid, this.eta));
                bmid=NNLib.subtract(bmid, NNLib.mult(dbmid, this.eta));
            }
        }
        return testPrediction();
    }

    /**
     * Train the neural network
     * @param nbEpoch number of epoch to run
     */
    public void train(int nbEpoch){
        String trainingProgress = "";
        for (int e = 0; e < nbEpoch; e++){

            System.out.println(" [ Epoch " +e+ "]");
            trainingProgress += e + "," + trainingEpoch() + "\n";
        }

        // Export the error per epoch in output file
        DataLib.exportDataToCSV("training.out", trainingProgress);
    }
}