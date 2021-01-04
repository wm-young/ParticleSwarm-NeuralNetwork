using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

/// <summary>
/// Represents a particle contained within the swarm. The particles position corresponds
/// to the weights of the network, and thus one particle is one entire neural network weight
/// structure. The fitness of a particle is then the number of misclassified patterns, as
/// when this value is reduced the network will be performing better.
/// 
/// @author:    Michael Young - 4245718
/// @author:    Terrence Knox - 4347860
/// @date:      April 2013
/// @version:   1.0
/// </summary>

    public class Particle
    {

        private Position position;
        private Velocity velocity;
        private double fitness;

        public Particle(int positionSize,int velocitySize)
        {
            position = new Position(positionSize);
            velocity = new Velocity(velocitySize);
        }

        public double Fitness
        {
            set { fitness = value; }
            get { return fitness; }
        }

        public Position Position
        {
            set { position = value; }
            get { return position; }
        }

        public Velocity Velocity
        {
            set { velocity = value; }
            get { return velocity; }
        }
    }//Particle