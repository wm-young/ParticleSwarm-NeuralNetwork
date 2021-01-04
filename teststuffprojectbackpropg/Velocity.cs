using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

/// <summary>
/// Represents the velocity of a particle
/// 
/// @author:    Michael Young - 4245718
/// @author:    Terrence Knox - 4347860
/// @date:      April 2013
/// @version:   1.0
/// </summary>

    public class Velocity
    {
        private double[] vector;//velocity of particle in search space

        public Velocity(int size)
        {
            vector = new double[size];
        }


        public double [] Vector{
            set { vector = value; }
            get { return vector; }
        }

        public int getSize()
        {
            return vector.Length;
        }

        public double getValue(int i)
        {
            return vector[i];
        }

        public override string ToString()
        {
            string text = "Velocity Vector: \n";
            text += "[";
            for (int i = 0; i < vector.Length; i++)
            {
                text += vector[i] + ",";
            }
            text += "]";
            return text;
        }
    }//Velocity
