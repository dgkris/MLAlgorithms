package com.dgkris.mlalgorithms;

import Jama.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.Random;


public class MultiLayerPerceptron implements Serializable {

    private static final long serialVersionUID = 4067489398414382568L;
    private static final String LINEAR = "linear";
    private static final String LOGISTIC = "logistic";
    private static final String SOFTMAX = "softmax";

    private final Logger logger = LoggerFactory.getLogger(MultiLayerPerceptron.class);

    private int nInputs;
    private int nOutputs;
    private long nTrainingRows;
    private int hiddenLayerSize;

    private int beta;
    private float momentum;
    private String outputType;

    private Matrix inputs;
    private Matrix targets;
    private Matrix weights1;
    private Matrix weights2;
    private Matrix hidden;

    /**
     * @param inputs   : Input array
     * @param targets  : Target Outputs
     * @param nhidden  : Number of hidden layers
     * @param beta     : Hidden layer parameter
     * @param momentum : Momentum for delta error
     * @param outtype  : Type of output (linear,softype,logistic)
     */
    public MultiLayerPerceptron(double inputs[][], double targets[][], int nhidden, int beta, float momentum, String outtype) {
        this.nInputs = getNumberOfColumns(inputs);
        this.nOutputs = getNumberOfColumns(targets);
        this.nTrainingRows = getNumberOfRows(inputs);
        this.hiddenLayerSize = nhidden;
        this.beta = beta;
        this.momentum = momentum;
        this.outputType = outtype;

        this.inputs = new Matrix(inputs);
        this.targets = new Matrix(targets);

        initializeWeights();

    }

    /**
     * <blockquote> Initialize weights to values between -1/sqrt(n) to 1/sqrt(n). Does a random-0.5/(sqrt(n)/2)<blockquote>
     */
    private void initializeWeights() {
        weights1 = (Matrix.random(nInputs + 1, hiddenLayerSize).
                minus(new Matrix(nInputs + 1, hiddenLayerSize, 0.5))).times(2.0d / Math.sqrt(nInputs));
        weights2 = (Matrix.random(hiddenLayerSize + 1, nOutputs).
                minus(new Matrix(hiddenLayerSize + 1, nOutputs, 0.5))).times(2.0d / Math.sqrt(hiddenLayerSize));
    }

    public Matrix getInputs() {
        return inputs;
    }

    public void setInputs(Matrix inputs) {
        this.inputs = inputs;
    }

    /**
     * @param eta            : Learning Rate. Small values preferred as large values of eta may lead to giant steps and thus a
     *                       minima might be over-stepped.
     * @param iterationCount : Number of training loops.
     *                       <blockquote>Updates the weights by computing delta=Computed Output-Expected Output<blockquote>
     */
    public void train(float eta, int iterationCount) {

        //Add bias node layer inputs to input values
        inputs = addBiasNodeInputs(inputs, 1.0d);

        Matrix updatew1 = new Matrix(weights1.getRowDimension(), weights1.getColumnDimension(), 0.0d);
        Matrix updatew2 = new Matrix(weights2.getRowDimension(), weights2.getColumnDimension(), 0.0d);

        for (int iterationNumber = 0; iterationNumber < iterationCount; iterationNumber++) {

            //Compute Current Output=Inputs X Weights
            Matrix outputs = forwardPropogate(inputs);

            //Total induced error= sum of squares of error values.
            Double totalInducedError = computeTotalInducedError(targets, outputs);
            logger.debug("Iteration : {} , Total induced error : {}", iterationNumber, totalInducedError);

            Matrix deltaOutput = null;

            if (outputType.equals(LINEAR)) {
                deltaOutput = (targets.minus(outputs)).times(1.0d / nTrainingRows);
            } else if (outputType.equals(LOGISTIC)) {
                deltaOutput = (targets.minus(outputs))
                        .times((outputs.times(new Jama.Matrix(outputs.getRowDimension(), outputs.getColumnDimension(), 1.0d).minus(outputs))));
            } else if (outputType.equals(SOFTMAX)) {
                deltaOutput = (targets.minus(outputs)).times(1.0d / nTrainingRows);
            }

            Matrix deltaHidden;

            deltaHidden = hidden.arrayTimes(new Matrix(hidden.getRowDimension(), hidden.getColumnDimension(), 1.0d).minus(hidden)).arrayTimes(
                    (deltaOutput.times(weights2.transpose())));

            updatew1 = (inputs.transpose().times(deltaHidden.getMatrix(0, deltaHidden.getRowDimension() - 1, 0, deltaHidden.getColumnDimension() - 2))).times(eta).plus(
                    updatew1.times(momentum));
            updatew2 = (hidden.transpose().times(deltaOutput)).times(eta).plus(updatew2.times(momentum));

            weights1.plusEquals(updatew1);
            weights2.plusEquals(updatew2);

            shuffleInputsandTargets();

        }
    }

    /**
     * Shuffle the inputs and the corresponding targets to ensure uniform distribution of weights
     */
    private void shuffleInputsandTargets() {
        int j = -1;
        int rows = (int) nTrainingRows;
        Random rand = new Random();
        double inp[][] = inputs.getArray();
        double targ[][] = targets.getArray();
        for (int i = rows - 1; i > 0; i--) {
            j = rand.nextInt(i);
            swap(inp[i], inp[j]);
            swap(targ[i], targ[j]);
        }
        inputs = new Matrix(inp);
        targets = new Matrix(targ);
    }

    private void swap(double[] item1, double[] item2) {
        double holder[] = item1;
        item2 = item1;
        item1 = holder;
    }

    /**
     * @param targetMatrix
     * @param currentOutput
     * @return total error induced  : 0.5*(sum of squares of all matrix values)
     */
    private Double computeTotalInducedError(Matrix targetMatrix, Matrix currentOutput) {
        Matrix error = targets.minus(currentOutput);

        for (int rowHandle = 0; rowHandle < error.getRowDimension(); rowHandle++) {
            for (int columnHandle = 0; columnHandle < error.getColumnDimension(); columnHandle++) {
                error.set(rowHandle, columnHandle, Math.pow(error.get(rowHandle, columnHandle), 2));
            }
        }
        return 0.5d * error.norm1();
    }

    /**
     * Does a forward propogation step with the inputs passed
     *
     * @param inputs
     * @return output = biasedInput X Weights
     */
    public Matrix forwardPropogate(Matrix inputs) {

        hidden = inputs.times(weights1);
        for (int rowHandle = 0; rowHandle < hidden.getRowDimension(); rowHandle++)
            for (int columnHandle = 0; columnHandle < hidden.getColumnDimension(); columnHandle++) {
                hidden.set(rowHandle, columnHandle, 1.0d / (1.0d + Math.exp(-beta * hidden.get(rowHandle, columnHandle))));
            }

        hidden = addBiasNodeInputs(hidden, 1.0d);

        Matrix outputs = hidden.times(weights2);

        if (outputType.equals("linear")) {
            return outputs;
        } else if (outputType.equals("logistic")) {
            for (int rowHandle = 0; rowHandle < outputs.getRowDimension(); rowHandle++)
                for (int columnHandle = 0; columnHandle < outputs.getColumnDimension(); columnHandle++) {
                    hidden.set(rowHandle, columnHandle, 1.0d / (1.0d + Math.exp(-beta * outputs.get(rowHandle, columnHandle))));
                }
        }
        return null;

    }

    /**
     * Predicts the output for the input matrix
     *
     * @param input
     * @return Predicted output matrix for the inputs
     */
    public Matrix predict(Matrix input) {
        input = addBiasNodeInputs(input, 1.0d);

        Matrix outputs = forwardPropogate(input);
        int nClasses = targets.getColumnDimension();
        if (nClasses == 1) {
            nClasses = 2;
            for (int rowHandle = 0; rowHandle < outputs.getRowDimension(); rowHandle++)
                for (int columnHandle = 0; columnHandle < outputs.getColumnDimension(); columnHandle++) {
                    if (outputs.get(rowHandle, columnHandle) > 0.5d)
                        outputs.set(rowHandle, columnHandle, 1.0d);
                    else
                        outputs.set(rowHandle, columnHandle, 0.0d);
                }
            return outputs;
        } else {
            return null;
        }
    }

    /**
     * Adds a bias node to handle the zero inputs
     *
     * @param matrix
     * @param biasValue
     * @return
     */
    private Matrix addBiasNodeInputs(Matrix matrix, double biasValue) {
        Matrix biasConcatMatrix = new Matrix(matrix.getRowDimension(), matrix.getColumnDimension() + 1);
        biasConcatMatrix.setMatrix(0, matrix.getRowDimension() - 1, 0, matrix.getColumnDimension() - 1, matrix);
        biasConcatMatrix.setMatrix(0, matrix.getRowDimension() - 1, biasConcatMatrix.getColumnDimension() - 1, biasConcatMatrix.getColumnDimension() - 1,
                new Matrix(biasConcatMatrix.getRowDimension(), 1, biasValue));
        return biasConcatMatrix;
    }

    private int getNumberOfColumns(double[][] array) {
        return array[0].length;
    }

    private long getNumberOfRows(double[][] array) {
        return array.length;
    }

    private double[][] normalize(double[][] denormalizedInput) {
        int nRow = (int) getNumberOfRows(denormalizedInput);
        int nCol = getNumberOfColumns(denormalizedInput);
        double normalizedInput[][] = new double[nRow][nCol];
        double lowerRange = 0.0d, upperRange = 1.0d;
        double max[] = new double[nCol];
        double min[] = new double[nCol];

        for (int col = 0; col < nCol; col++) {
            max[col] = Double.MIN_VALUE;
            min[col] = Double.MAX_VALUE;
        }

        for (int col = 0; col < nCol; col++) {
            for (int row = 0; row < nRow; row++) {
                if (min[col] > denormalizedInput[row][col])
                    min[col] = denormalizedInput[row][col];
                if (max[col] < denormalizedInput[row][col])
                    max[col] = denormalizedInput[row][col];
            }
        }
        for (int col = 0; col < nCol; col++) {
            for (int row = 0; row < nRow; row++) {
                normalizedInput[row][col] = ((upperRange - lowerRange) *
                        (denormalizedInput[row][col] - min[col])) / (max[col] - min[col]);
            }
        }
        return normalizedInput;
    }

    public static void main(String args[]) {
        //Input vector
        double[][] inputs = {{0., 0}, {0., 1.}, {1., 0.}, {1., 1.}};

        //Output vector : Input output form a AND GATE here
        double[][] targets = {{0}, {0}, {0}, {1}};

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(inputs, targets, 2, 3, 0.9f, LINEAR);

        mlp.train(0.5f, 5000);

        //Prediction for an input vector
        double[][] input = {{1., 1}};
        Matrix output = mlp.predict(new Matrix(input));

        //Print the output
        output.print(output.getRowDimension(), output.getColumnDimension());
    }
}
