using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// This class represents an Artificial Neural Network. The network contained here
/// can be trained using either the standard vanilla backpropagation, or the newly
/// proposed partical swarm optimization method. Parameters for the networks are
/// currently hardcoded for those that were determined to be effective values
/// for the given training style and dataset.
/// 
/// The possible datasets for training are handwritten digit recognition, which
/// contains a feature set of 64 values, and wine data which is 13 features in size.
/// The digit recognition training set is significantly more complex and will take
/// substantially longer (computationally) to train.
/// 
/// The training method and data set to be used are selected when the program is run.
/// 
/// @author:    Terrence Knox - 4347860
/// @author:    W. Michael Young - 4245718
/// @date:      April, 2013
/// @version:   1.0
/// </summary>

public class Network
{
    private int _hiddenDims = 15;        // Number of hidden neurons.
    private int _inputDims = 64;        // Number of input neurons.
    private int _outputDims = 10;       //Number of output neurons.
    private int _iteration;            // Current training iteration.
    private int _maxIteration = 500;          //maximum number of iterations.
    private int swarmProp;                  //if its swarmprop
    private Layer _hidden;              // Collection of hidden neurons.
    private Layer _inputs;              // Collection of input neurons.
    private Layer _outputs;             //Collection of output neurons
    private List<Pattern> _patterns;    // Collection of training patterns.
    private double[] _scales;       //scaling factors for use with the wine data
    private Random _rnd = new Random(); // Global random number generator.
    private int dataset;                //state variable for dataset; 0 = digits, 1 = wine.
    //SWARM VARIABLES
    private int swarmSize;              //the size of the swarm
    private int dimensions;             //size of a particles vector
    private int upper;                  //upper bounds for position value
    private int lower;                  //lower bounds for position value
    private List<int> psoIndexes;          //randomized list for which error index order to use

    ParticleSwarm pso;

    [STAThread]
    static void Main()
    {
        new Network();
    }//Main

    /// <summary>
    /// Creates the network and calls the functions required to perform training on
    /// the selected data sets using the chosen method
    /// </summary>
    public Network()
    {
        GetIfSwarm();
        GetDataset();
        LoadPatterns();
        Initialise();
        Train();
        Test();
    }//constructor

    /// <summary>
    /// Prompts the user to select a dataset (digit recognition, wine) and initializes
    /// the network parameters to the values deemed efficient through testing.
    /// 0 is for digit recognition, 1 is for wine
    /// </summary>
    private void GetDataset()
    {
        Console.WriteLine("Select Dataset: 0 = Digit Recognition, 1 = Wine");
        string strVal = Console.ReadLine();
        dataset = Int32.Parse(strVal);
        switch (swarmProp)
        {
            case 0:
                //using backprop, set backprop network params
                switch (dataset)
                {
                    case 0:
                        _outputDims = 10;
                        _hiddenDims = 18;
                        _inputDims = 64;
                        _maxIteration = 200;
                        break;
                    case 1:
                        _outputDims = 3;
                        _hiddenDims = 5;
                        _inputDims = 13;
                        _maxIteration = 100;
                        break;
                }
                break;
            case 1:
                //using pso, set pso network params
                switch (dataset)
                {
                    case 0:
                        _outputDims = 10;
                        _hiddenDims = 10;
                        _inputDims = 64;
                        _maxIteration = 50;
                        break;
                    case 1:
                        _outputDims = 3;
                        _hiddenDims = 10;
                        _inputDims = 13;
                        _maxIteration = 40;
                        break;
                }
                break;
        }
    }//GetDataset

    /// <summary>
    /// Prompts the user to select a training algorithm in order to train the network.
    /// 0 is for vanilla backpropagation, 1 is for particle swarm optimization
    /// </summary>
    private void GetIfSwarm()
    {
        Console.WriteLine("Select Training Algorithm: 0 = Backpropagation, 1 = Particle Swarm Optimization");
        string strVal = Console.ReadLine();
        swarmProp = Int32.Parse(strVal);
        if (swarmProp == 1)
        {
            switch (dataset)
            {
                case 0:
                    swarmSize = 100;
                    dimensions = 1332;
                    upper = 1;
                    lower = -1;
                    break;
                case 1:
                    swarmSize = 50;
                    dimensions = 160;
                    upper = 1;
                    lower = -1;
                    break;
            }
            pso = new ParticleSwarm(swarmSize, dimensions, upper, lower);
            //psoIndexes = new List<int>();
        }
    }//GetIfSwarm

    /// <summary>
    /// Performs the necessary actions to train the network using the selected dataset and
    /// training algorithm. Output is displayed each epoch which consists of the epoch
    /// number, the error for the epoch and in the case of pso the global best misclassifications.
    /// A timer is run at the start as well to determine total computational time for a given training
    /// style.
    /// </summary>
    private void Train()
    {
        //start the timer
        System.Diagnostics.Stopwatch timer = System.Diagnostics.Stopwatch.StartNew();
        String errorVals = "Epoch, Error Value, Validation Rate" + Environment.NewLine;
        double error;
        do
        {
            error = 0;
            if (swarmProp == 1)
            {
                for (int i = 0; i < swarmSize; i++)
                {
                    //reset fitness of each particle in the swarm to 0 for this iteration
                    pso.setFitness(i, 0);
                }
            }
            foreach (Pattern pattern in _patterns)
            {
                double[] predics;
                double[] delta;
                switch (swarmProp)
                {
                    case 0:
                        //-----BACKPROPAGATION-----
                        //retrieve predicted values
                        predics = Activate(pattern);
                        //calculate the error of the network
                        delta = new double[predics.Length];
                        for (int i = 0; i < predics.Length; i++)
                        {
                            delta[i] = pattern.Output[i] - predics[i];
                            error += Math.Pow(delta[i], 2);
                        }
                        //adjust the network weights
                        AdjustWeights(delta);
                        break;
                    case 1:
                        //-----PARTICLE SWARM OPTIMIZATION-----
                        for (int i = 0; i < swarmSize; i++)
                        {
                            //set the weights of the ANN to that of the current particle
                            setWeightsForSwarm(i);
                            //retrieve predicted values
                            predics = Activate(pattern);
                            delta = new double[predics.Length];
                            for (int j = 0; j < predics.Length; j++)
                            {
                                //calculate error, entirely for printing purposes
                                delta[j] = pattern.Output[j] - predics[j];
                                error += Math.Pow(delta[j], 2);
                            }
                            //if the pattern was classified properly, do nothing
                            //if it was classified wrong, increase error by 1
                            pso.setFitness(i, pso.Swarm[i].Fitness + swarmPredictVal(pattern));
                        }
                        break;
                }
            }
            if (swarmProp == 1)
            {
                //perform one iteration of the swarm
                pso.iterateSwarm(_iteration);
                //set weights of network to the global best, for validation and output purposes
                setWeightsForSwarm(pso.findGlobalBest());
            }
            double validate = getValidationRate();
            if (swarmProp == 1)
            {
                Console.WriteLine("Iteration {0}\tError {1:0.000}\tValidation {2:0.000}\tGB {3:0}", _iteration, error, validate, pso.Swarm[pso.findGlobalBest()].Fitness);
                errorVals += _iteration.ToString() + ", " + error.ToString() + ", " + validate.ToString() + ", " + pso.Swarm[pso.findGlobalBest()].Fitness + Environment.NewLine;
            }
            else
            {
                Console.WriteLine("Iteration {0}\tError {1:0.000}\tValidation {2:0.000}", _iteration, error, validate);
                errorVals += _iteration.ToString() + ", " + error.ToString() + ", " + validate.ToString() + Environment.NewLine;
            }
            _iteration++;
            //if (_iteration > _restartAfter) Initialise();
        } while (_iteration <= _maxIteration);
        if (swarmProp == 1)
        {
            pso.printBest();
        }
        timer.Stop();
        Console.WriteLine("Elapsed Time: " + timer.Elapsed.ToString());
        printToFile(errorVals);
    }//Train

    /// <summary>
    /// Sets the weights of the network to those contained within a given particle.
    /// Weights are set on a per layer basis starting from the top, and begin with 
    /// the input layer. Weight setting continues in this fashion until the lower most
    /// weights between hidden and output are set.
    /// </summary>
    /// <param name="index">the index of the particle within the swarm whose weights will be used</param>
    private void setWeightsForSwarm(int index)
    {
        double[] newWeights = pso.Swarm[index].Position.Vector;
        int count = 0;
        foreach (Neuron hidden in _hidden)
        {
            foreach (Neuron input in _inputs)
            {
                Weight currWeight = hidden.Weights.Find(delegate(Weight t) { return t.Input == input; });
                currWeight.Value = newWeights[count];
                count++;
            }
        }

        foreach (Neuron output in _outputs)
        {
            foreach (Neuron hidden in _hidden)
            {
                Weight currWeight = output.Weights.Find(delegate(Weight t) { return t.Input == hidden; });
                currWeight.Value = newWeights[count];
                count++;
            }
        }
    }//setWeightsForSwarm

    /// <summary>
    /// Determines if the particle predicted the proper value with a given
    /// pattern. If it has, it will return 0 (and thus add 0 to the overall
    /// error) otherwise it will increase the error by 1
    /// </summary>
    /// <param name="pattern">the pattern being examined</param>
    /// <returns>the increase in error for the particle</returns>
    private double swarmPredictVal(Pattern pattern)
    {
        int best = 0;
        double[] predics = Activate(pattern);
        double bestCalc = predics[0];
        for (int i = 1; i < predics.Length; i++)
        {
            if (predics[i] > bestCalc)
            {
                best = i;
                bestCalc = predics[i];
            }
        }
        if (pattern.Output[best] == 1.0)
        {
            //classified properly, no error
            return 0.0;
        }
        else
        {
            //misclassification, error
            return 1.0;
        }
    }//swarmPredictVal

    /// <summary>
    /// Determines the validation rate of the networks current weight structure. The testing files
    /// contained with the application will be used for the selected dataset. The return value is a
    /// percentage based on the number of correctly classified patterns and the total number of patterns.
    /// </summary>
    /// <returns>the validation rate</returns>
    private double getValidationRate()
    {
        double rate = 0.0;
        double totalPatterns = 0.0;
        double correctPatterns = 0.0;
        switch (dataset)
        {
            case 0:
                //digit recognition dataset
                for (int i = 0; i < 10; i++)
                {
                    StreamReader file = new StreamReader(@"Digit_Recog\digit_test_" + i.ToString() + ".txt");
                    String line = "";
                    while ((line = file.ReadLine()) != null)
                    {
                        double[] predics = Activate(new Pattern(line, _inputDims, _outputDims, i));
                        int best = 0;
                        double bestCalc = predics[0];
                        //determine which was classified
                        for (int j = 1; j < predics.Length; j++)
                        {
                            if (predics[j] > bestCalc)
                            {
                                best = j;
                                bestCalc = predics[j];
                            }
                        }
                        if (best == i)
                        {
                            //properly classified, increase correctness
                            correctPatterns += 1.0;
                        }
                        totalPatterns += 1.0;
                    }
                }
                break;
            case 1:
                //wine
                StreamReader fileWine = new StreamReader(@"Wine\wine_test.txt");
                String lineWine = "";
                while ((lineWine = fileWine.ReadLine()) != null)
                {
                    String[] lineVals = lineWine.Split(',');
                    Pattern currPat = new Pattern(lineWine.Substring(2), _inputDims, _outputDims, Int32.Parse(lineVals[0]));
                    for (int i = 0; i < currPat.Inputs.Length; i++)
                    {
                        currPat.setInput(i, currPat.Inputs[i] / _scales[i]);
                    }
                    double[] predics = Activate(currPat);
                    int best = 0;
                    double bestCalc = predics[0];
                    //determine which was classified
                    for (int j = 1; j < predics.Length; j++)
                    {
                        if (predics[j] > bestCalc)
                        {
                            best = j;
                            bestCalc = predics[j];
                        }
                    }
                    if (best == (Int32.Parse(lineVals[0]) - 1))
                    {
                        //proper classification, increase correctness
                        correctPatterns += 1.0;
                    }
                    totalPatterns += 1.0;
                }
                break;
        }
        rate = (correctPatterns / totalPatterns) * 100;
        return rate;
    }//getValidationRate

    /// <summary>
    /// Prints a string to an output CSV (comma separated values) file. This is used after
    /// training has been completed to generated a file which may be opened in Excel
    /// to easily create graphs and view the data.
    /// </summary>
    /// <param name="errorVals">the string to be written</param>
    private void printToFile(String errorVals)
    {
        StreamWriter sw = new StreamWriter("output.csv");
        sw.Write(errorVals);
        sw.Close();
    }//printToFile

    /// <summary>
    /// Prompts the user for a filename (those contained with the application, or which
    /// are in the folder locally with the exe) and reads it. The output of the network
    /// will be printed. Activates after training has been completed.
    /// </summary>
    private void Test()
    {
        Console.WriteLine("\nBegin network testing\nPress Ctrl C to exit\n");
        while (1 == 1)
        {
            try
            {
                Console.WriteLine("Digit path to validate");
                string path = Console.ReadLine();
                Console.WriteLine("Enter the expected value:");
                string expectedValue = Console.ReadLine();
                System.IO.StreamReader file = new System.IO.StreamReader(path);
                String line;
                while ((line = file.ReadLine()) != null)//keep reading in new lines
                {
                    double[] predics = Activate(new Pattern(line, _inputDims, _outputDims, Int32.Parse(expectedValue)));
                    int best = 0;
                    double bestCalc = predics[0];
                    for (int i = 1; i < predics.Length; i++)
                    {
                        if (predics[i] > bestCalc)
                        {
                            best = i;
                            bestCalc = predics[i];
                        }
                    }
                    Console.WriteLine("{0:0}\n", best);
                }
                file.Close();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }
    }//Test

    /// <summary>
    /// Activates a given input pattern and determines the output of the network
    /// at each output node based on the current weight structure.
    /// </summary>
    /// <param name="pattern">the pattern to be activated</param>
    /// <returns>an array of outputs which correspond to the output nodes</returns>
    private double[] Activate(Pattern pattern)
    {
        double[] outputs = new double[_outputs.Count];
        int counter = 0;
        for (int i = 0; i < pattern.Inputs.Length; i++)
        {
            _inputs[i].Output = pattern.Inputs[i];
        }
        foreach (Neuron neuron in _hidden)
        {
            neuron.Activate();
        }
        foreach (Neuron neuron in _outputs)
        {
            neuron.Activate();
            outputs[counter] = neuron.Output;
            counter++;
        }
        return outputs;
    }//Activate

    /// <summary>
    /// Adjusts the weights of the network for backpropagation. Using the error found
    /// at the output node (expected - actual), the weights between hidden and output
    /// are updated. Then the error is propagated backwards and used to update the weights
    /// between hidden and input.
    /// </summary>
    /// <param name="delta">the error given by each output node</param>
    private void AdjustWeights(double[] delta)
    {
        int counter = 0;
        foreach (Neuron neuron in _outputs)
        {
            neuron.AdjustWeights(delta[counter]);
            counter++;
        }
        foreach (Neuron neuron in _hidden)
        {
            double errorSum = 0.0;
            foreach (Neuron outNeuron in _outputs)
            {
                errorSum += outNeuron.ErrorFeedback(neuron);
            }
            neuron.AdjustWeights(errorSum);
        }
    }//AdjustWeights

    /// <summary>
    /// Initializes the input, hidden and output layers of the ANN as well as
    /// the nodes they contain
    /// </summary>
    private void Initialise()
    {
        _inputs = new Layer(_inputDims);
        _hidden = new Layer(_hiddenDims, _inputs, _rnd);
        _outputs = new Layer(_outputDims, _hidden, _rnd);
        _iteration = 0;
        Console.WriteLine("Network Initialised");
    }//Initialise

    /// <summary>
    /// Reads the input file based on the selected dataset and creates the training List
    /// which is used to train the network. The list is randomized before use, and in the case
    /// of wine data it is also normalized to make all values between 0 and 1. The normalization
    /// is done by dividing each feature in a given training example by the largest value of that
    /// specific feature in the entire set. Thus it would be as such:
    /// v1 : {1, 2, 3}
    /// v2 : {3, 1, 2}
    /// then the scale would be {3, 2, 3}
    /// new v1 : {1/3, 2/2, 3/3}
    /// new v2 : {3/3, 1/2, 2/3}
    /// </summary>
    private void LoadPatterns()
    {

        Pattern example;
        string line;
        _patterns = new List<Pattern>();
        switch (dataset)
        {
            case 0:
                //digit recognition
                for (int i = 0; i < 10; i++)//for each digit 0-9
                {
                    System.IO.StreamReader file = new System.IO.StreamReader(@"Digit_Recog\digit_train_" + i + ".txt");
                    while ((line = file.ReadLine()) != null)//keep reading in new lines
                    {
                        example = new Pattern(line, _inputDims, _outputDims, i);
                        _patterns.Add(example);
                    }
                    file.Close();
                }
                break;
            case 1:
                //wine data
                _scales = new double[13];
                System.IO.StreamReader fileWine = new System.IO.StreamReader(@"Wine\wine_train.txt");
                while ((line = fileWine.ReadLine()) != null)
                {
                    string[] lineVals = line.Split(',');
                    int expected = Int32.Parse(lineVals[0]);
                    example = new Pattern(line.Substring(2), _inputDims, _outputDims, expected);
                    _patterns.Add(example);
                    if (_patterns.Count == 1)
                    {
                        for (int i = 0; i < _scales.Length; i++)
                        {
                            _scales[i] = _patterns[0].Inputs[i];
                        }
                    }
                    for (int i = 0; i < _scales.Length; i++)
                    {
                        _scales[i] = Math.Max(_patterns[_patterns.Count - 1].Inputs[i], _scales[i]);
                    }
                }
                normalizePatterns();
                break;
        }
        randomizePatterns();
    }//LoadPatterns

    /// <summary>
    /// Normalizes the list based on the found scale. For use with wine data. Example
    /// is shown in previous function 'Load Pattern' comments. Scale factors
    /// are also obtained in this function.
    /// </summary>
    private void normalizePatterns()
    {
        foreach (Pattern pattern in _patterns)
        {
            double[] inputs = pattern.Inputs;
            for (int i = 0; i < inputs.Length; i++)
            {
                pattern.setInput(i, inputs[i] / _scales[i]);
            }
        }
    }//normalizePatterns

    /// <summary>
    /// Randomizes the list which contains the training patterns to prevent the
    /// network from being trained on only the last value in the file.
    /// </summary>
    private void randomizePatterns()
    {
        Random rdm = new Random();
        int n = _patterns.Count;
        while (n > 1)
        {
            n--;
            int k = rdm.Next(n + 1);
            //swap training candidates
            Pattern temp = _patterns[k];
            _patterns[k] = _patterns[n];
            _patterns[n] = temp;
        }
    }//randomizePatterns
}//Network

/// <summary>
/// Represents a layer within an artifical neural network. Layers contain
/// Neurons, and have no other function besides organization.
/// </summary>
public class Layer : List<Neuron>
{
    public Layer(int size)
    {
        for (int i = 0; i < size; i++)
            base.Add(new Neuron());
    }//contructor

    public Layer(int size, Layer layer, Random rnd)
    {
        for (int i = 0; i < size; i++)
            base.Add(new Neuron(layer, rnd));
    }//contructor
}//Layer

/// <summary>
/// Represents a Neuron contained within a layer of an artifical neural network.
/// Neurons in one layer are only connected to neurons in the layer immediately
/// before and/or after it. This means the input neurons are connected to every
/// neuron in the hidden layer, the hidden is connected to input and output, and
/// the output is connected to the hidden. Weight values are initially randomly
/// set. Methods for altering the weights when using backpropagation are
/// contained here. For the PSO method, the weight update work is contained within
/// ParticleSwarm.
/// </summary>
public class Neuron
{
    private double _bias;                       // Bias value.
    private double _error;                      // Sum of error.
    private double _input;                      // Sum of inputs.
    private double _lambda = 6;                // Steepness of sigmoid curve.
    private double _learnRate = 0.7;            // Learning rate.
    private double _output = double.MinValue;   // Preset value of neuron.
    private List<Weight> _weights;              // Collection of weights to inputs.

    public Neuron() { }//empty constructor

    public Neuron(Layer inputs, Random rnd)
    {
        _weights = new List<Weight>();
        foreach (Neuron input in inputs)
        {
            Weight w = new Weight();
            w.Input = input;
            w.Value = rnd.NextDouble() * 2 - 1;
            _weights.Add(w);
        }
    }//constructor

    /// <summary>
    /// Sums the input values multiplied by their associated weight values
    /// to the neuron
    /// </summary>
    public void Activate()
    {
        _input = 0;
        foreach (Weight w in _weights)
        {
            _input += w.Value * w.Input.Output;
        }
    }//Activate

    /// <summary>
    /// gets the list of weights
    /// </summary>
    public List<Weight> Weights
    {
        get
        {
            return _weights;
        }
    }//Weights

    /// <summary>
    /// returns the error present in a neuron which is input to the current, to allow
    /// for backpropagation
    /// </summary>
    /// <param name="input">the input neuron in question</param>
    /// <returns>the error feedback</returns>
    public double ErrorFeedback(Neuron input)
    {
        Weight w = _weights.Find(delegate(Weight t) { return t.Input == input; });
        return _error * Derivative * w.Value;
    }//ErrorFeedback

    /// <summary>
    /// Adjusts the weights of the neuron based on the error and learning rate
    /// </summary>
    /// <param name="value">the error value</param>
    public void AdjustWeights(double value)
    {
        _error = value;
        for (int i = 0; i < _weights.Count; i++)
        {
            _weights[i].Value += _error * Derivative * _learnRate * _weights[i].Input.Output;
        }
        _bias += _error * Derivative * _learnRate;
    }//AdjustWeights

    /// <summary>
    /// Calculates the derivative of the activation function
    /// </summary>
    private double Derivative
    {
        get
        {
            double activation = Output;
            return activation * (1 - activation);
        }
    }//Derivative

    /// <summary>
    /// Gets the output of the neuron
    /// </summary>
    public double Output
    {
        get
        {
            if (_output != double.MinValue)
            {
                return _output;
            }
            return 1 / (1 + Math.Exp(-_lambda * (_input + _bias)));
        }
        set
        {
            _output = value;
        }
    }//Output
}//Neuron

/// <summary>
/// Represents a single training pattern. The pattern contains input values
/// as well as expected outputs. In the case of digit recognition, there are
/// 64 inputs and 10 outputs (outputs are 0, 1, 2, 3, ... 9) and for wine
/// there are 13 inputs and 3 outputs (outputs are 1, 2, 3 - network sees them
/// as 0, 1, 2 however to correspond with array indexes).
/// </summary>
public class Pattern
{
    private double[] _inputs;
    private double[] _output;

    public Pattern(string value, int inputSize, int outputSize, int expectedValue)
    {
        _output = new double[outputSize];
        for (int i = 0; i < _output.Length; i++)
        {
            _output[i] = 0;
        }
        switch (outputSize)
        {
            case 3:
                _output[expectedValue - 1] = 1;
                break;
            case 10:
                _output[expectedValue] = 1;
                break;
        }
        _inputs = new double[inputSize];
        String[] line = value.Split(',');
        for (int i = 0; i < inputSize; i++)
        {
            _inputs[i] = Double.Parse(line[i]);
        }
    }//constructor

    /// <summary>
    /// gets the list of input values
    /// </summary>
    public double[] Inputs
    {
        get { return _inputs; }
    }//Inputs

    /// <summary>
    /// sets a specific input value. For use with wine data in order to
    /// normalize the pattern inputs
    /// </summary>
    /// <param name="index">the index of the input to change</param>
    /// <param name="value">the new input value</param>
    public void setInput(int index, double value)
    {
        _inputs[index] = value;
    }//setInput

    /// <summary>
    /// gets the expected outputs of the pattern
    /// </summary>
    public double[] Output
    {
        get { return _output; }
    }//Output

    /// <summary>
    /// gets a single value which is the index of the expected value. Differs
    /// from Output, as Output returns the full array
    /// </summary>
    /// <returns>the index of the expected value</returns>
    public int getExpected()
    {
        for (int i = 0; i < _output.Length; i++)
        {
            if (_output[i] == 1.0)
            {
                return i;
            }
        }
        return 0;
    }//getExpected
}//Pattern

/// <summary>
/// Represents a weight between two neurons. Weights contain an input
/// value and a weight, nothing else.
/// </summary>
public class Weight
{
    public Neuron Input;
    public double Value;
}//Weight
