using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

/// <summary>
/// Represents the position vector of a particle, which is the weights of
/// a neural network structure in this case
/// 
/// @author:    Michael Young - 4245718
/// @author:    Terrence Knox - 4347860
/// @date:      April 2013
/// @version:   1.0
/// </summary>

    public class Position
    {
        private double[] vector; //position of particle in search space

        public Position(int size)
        {
            this.vector = new double[size];
        }

        public double[] Vector
        {
            set { this.vector = value; }
            get { return this.vector; }
        }

        public double getValue(int index)
        {
            return this.vector[index];
        }

        public void setValue(int index, double value)
        {
            this.vector[index] = value;
        }

        //Randomly initializes the position vector with random values
        public void randInt()
        {
            for (int i = 0; i < vector.Length; i++)
            {

            }
        }//randInit

        public int getSize()
        {
            return vector.Length;
        }

        public override string ToString()
        {
            string text = "Position Vector: \n";
            text += "[";
            for (int i = 0; i < vector.Length; i++)
            {
                text += vector[i] + ",";
            }
            text += "]";
            return text;
        }
    }//Position
