using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

/// <summary>
/// Represents the swarm of Particle objects. The particles will work towards minimizing their
/// error, which in the case of training a neural network is determined by the number of
/// misclassified patterns using a given particles position vector as the weights. As the error
/// is minimized it will signify that the particle is classifying more patterns properly
/// and is thus learning the data.
/// 
/// @author:    Michael Young - 4245718
/// @author:    Terrence Knox - 4347860
/// @date:      April 2013
/// @version:   1.0
/// </summary>

    public class ParticleSwarm
    {

        /**
         * @param args
         */
        private int swarm_size;
        private Particle[] swarm;
        private Particle[] pBest;//best particle solutions found so far
        private Particle gBest;//global best - best solution overall found so far
        private int max_iterations;
        private int xMax; //min and max value the function can take
        private int xMin;
        private double c1;
        private double c2;
        private double inertia;
        double vMax; //maximum velocity of a particle to avoid over accelerating passed world bounds
        int posSize,velSize;
        private int position_fault;

        public ParticleSwarm(int swarmSize, int dimensions, int upperBound, int lowerBound)
        {
            swarm_size = swarmSize;
            posSize = dimensions;
            velSize = dimensions;
            xMin = lowerBound;
            xMax = upperBound;

            initializeParameters();
            initializeSwarm();
        }//constructor

        public Particle[] Swarm
        {
            set { this.swarm = value; }
            get { return this.swarm; }
        }

        private void initializeParameters()
        {
            position_fault = 0;
            pBest = new Particle[swarm_size];
            gBest = new Particle(posSize, velSize);
            c1 = 1.496180;
            c2 = 1.496180;
            inertia = 0.729844;
            max_iterations = 300;
            vMax = (0.2) * (xMax - xMin / 2);
        }

        private void initializeSwarm()
        {
            swarm = new Particle[swarm_size];
            Particle particle;
            Random rand = new Random();

            for (int i = 0; i < swarm_size; i++)
            {
                particle = new Particle(posSize, velSize);
                pBest[i] = new Particle(posSize, velSize);
                //Randomly initialize position
                for (int j = 0; j < particle.Position.getSize(); j++)
                {
                    double pos = rand.NextDouble() * (xMax - xMin) + xMin;
                    particle.Position.Vector[j] = pos;
                }

                //Randomly initialize velocity
                for (int j = 0; j < particle.Velocity.getSize(); j++)
                {
                    double vel = rand.NextDouble() * 2.0 - 1.0;
                    particle.Velocity.Vector[j] = vel;
                }
                swarm[i] = particle;
            }
        }

        /// <summary>
        /// Does one iteration through the whole swarm
        /// </summary>
        /// <param name="epoch"></param>
        public void iterateSwarm(int epoch)
        {
            Random rand = new Random();

            //update the pBest positions
            if (epoch == 0)
            {
                for (int i = 0; i < swarm_size; i++)
                {
                    pBest[i] = new Particle(2, 2);
                    pBest[i].Position.Vector = (double[])swarm[i].Position.Vector.Clone();
                    pBest[i].Velocity.Vector = (double[])swarm[i].Velocity.Vector.Clone();
                    //pBest[i].Fitness = calculateFitness(swarm[i]);
                    pBest[i].Fitness = swarm[i].Fitness;
                }
            }
            else
            {

                for (int i = 0; i < swarm_size; i++)
                {
                    //double fitness = calculateFitness(swarm[i]);
                    double fitness = swarm[i].Fitness;
                    //if fitness is better, update personal best for that particle
                    if (fitness < pBest[i].Fitness)
                    {
                        pBest[i] = new Particle(2, 2);
                        pBest[i].Position.Vector = (double[])swarm[i].Position.Vector.Clone();
                        pBest[i].Velocity.Vector = (double[])swarm[i].Velocity.Vector.Clone();
                        //pBest[i].Fitness = calculateFitness(swarm[i]);
                        pBest[i].Fitness = swarm[i].Fitness;
                    }
                }
            }

            //update the global best(gBest)
            int bestIndex = findGlobalBest();

            if (epoch == 0)
            {
                gBest = new Particle(posSize, velSize);
                gBest.Position.Vector = (double[])swarm[bestIndex].Position.Vector.Clone();
                gBest.Velocity.Vector = (double[])swarm[bestIndex].Velocity.Vector.Clone();
                 //gBest.Fitness = calculateFitness(swarm[bestIndex]);
                gBest.Fitness = swarm[bestIndex].Fitness;
            }
            else
            {
                if (swarm[bestIndex].Fitness < gBest.Fitness)
                {
                    gBest = new Particle(posSize, velSize);
                    gBest.Position.Vector = (double[])swarm[bestIndex].Position.Vector.Clone();
                    gBest.Velocity.Vector = (double[])swarm[bestIndex].Velocity.Vector.Clone();
                    //gBest.Fitness = calculateFitness(swarm[bestIndex]);
                    gBest.Fitness = swarm[bestIndex].Fitness;
                }
            }

            Velocity newVelocity;

            /* update velocities and position according to the following equations
             * NewVel = oldVel + c1r1(pbest-current) + c2r2(gbest-current)
             * Where c1 and c2 are acceleration constants, r1 and r2 are random values within (0,1)
             * 
             * newPos = oldPos + newVel
             * The new position is equal to the old position plus the velocity.
             */
            for (int i = 0; i < swarm_size; i++)
            {
                Position currPosition = swarm[i].Position;
                Velocity currVelocity = swarm[i].Velocity;
                newVelocity = new Velocity(velSize);

                for (int j = 0; j < swarm[i].Position.getSize(); j++)
                {
                    double r1 = rand.NextDouble();
                    double r2 = rand.NextDouble();
                    newVelocity.Vector[j] = inertia * currVelocity.Vector[j] + c1 * r1 * (pBest[i].Position.Vector[j] - currPosition.Vector[j]) + c2 * r2 * (gBest.Position.Vector[j] - currPosition.Vector[j]);
                }

                updateVelocity(newVelocity, swarm[i]);
                updatePosition(swarm[i]);
            }
            verifyPositions(); //verify that all the new positions are in the world
        }//iterateSwarm

        /*Updates the velocity of the given particle
         * Check to make sure the new velocity isn't above our threshold of vMax.
         * If it is, update accordingly depending on direction for each x and y value.*/
        private void updateVelocity(Velocity newVelocity, Particle particle)
        {
            for (int i = 0; i < newVelocity.getSize(); i++)
            {
                if (Math.Abs(newVelocity.Vector[i]) > vMax)
                {
                    if (newVelocity.Vector[i] < 0)
                    {
                        newVelocity.Vector[i] = -vMax;
                    }
                    else
                    {
                        newVelocity.Vector[i] = vMax;
                    }
                     
                }
                particle.Velocity.Vector[i] = newVelocity.Vector[i];
            }
        }

        /* Verifies that all the positions of the particles are within the world bounds.
         * If not then resets the position to a random location within the world. 
         */
        private void verifyPositions()
	    {
		    Random rand = new Random();
		    for(int i=0;i<swarm_size;i++)
		    {
                //Each particles position
                for (int j = 0; j < swarm[i].Position.getSize(); j++)
                {
                    if (swarm[i].Position.Vector[j] > xMax || swarm[i].Position.Vector[j] < xMin)
                    {
                        position_fault++;
                        Console.WriteLine("---Particle " + i + " position reset---");
                       Console.WriteLine("Position(" + swarm[i].Position.getValue(0) + "," + swarm[i].Position.getValue(1) + ")");
                        double pos = rand.NextDouble()*(xMax-xMin) + xMin;
                        swarm[i].Position.Vector[j] = pos;
                    }
                }
		    }
	    }

        //Updates the particles position according to the boundaries of the solution space
        private void updatePosition(Particle particle)
        {
            //Check boundaries to ensure particle stays within the limits
            for (int i = 0; i < particle.Position.getSize(); i++)
            {
                if ((particle.Position.Vector[i] + particle.Velocity.Vector[i]) > xMax || (particle.Position.Vector[i] + particle.Velocity.Vector[i]) < xMin)
                {
                    particle.Velocity.Vector[i] = -particle.Velocity.Vector[i]; //move in opposite direction to avoid steping out of bounds
                }
                particle.Position.Vector[i] = particle.Position.Vector[i] + particle.Velocity.Vector[i];
            }
        }


        //Returns the index of the best particle in the fitness list
        public int findGlobalBest()
        {
            double bestValue;
            int bestIndex;
            bestValue = swarm[0].Fitness;
            bestIndex = 0;
            for (int i = 0; i < swarm_size; i++)
            {
                if (swarm[i].Fitness < bestValue)
                {
                    bestValue = swarm[i].Fitness;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        //Calculates the fitness of all the particles in the swarm
        private double calculateFitness(Particle particle)
        {
            double x;
            double y;
            x = particle.Position.getValue(0);
            y = particle.Position.getValue(1);
            particle.Fitness = (Math.Pow((2.8125 - x + x * Math.Pow(y, 4)), 2) +
                      Math.Pow((2.25 - x + x * Math.Pow(y, 2)), 2) +
                      Math.Pow((1.5 - x + x * y), 2));
           // particle.Fitness = (Math.Pow(x,2) + Math.Pow(y,2));


            return particle.Fitness;
        }

        public void setFitness(int index, double fitness)
        {
            this.swarm[index].Fitness = fitness;
        }

        /// <summary>
        /// Returns the position of the particle at the given index value
        /// </summary>
        /// <param name="particle"></param>
        /// <returns></returns>
        public Position getVectorValues(int index)
        {
            return this.swarm[index].Position;
        }


        public void printSwarm()
	    {
		    for(int i=0;i<swarm_size;i++)
		    {
			    Console.WriteLine("Particle: "+ i);

			    Console.WriteLine("Pos: ("+swarm[i].Position.getValue(0)+","+swarm[i].Position.getValue(1)+")");
                Console.WriteLine("Vel: (" + swarm[i].Velocity.getValue(0) + "," + swarm[i].Velocity.getValue(1) + ")");
                Console.WriteLine("Fit: (" + swarm[i].Fitness+")");
			    Console.WriteLine("--------------");
		    }
	    }

        public void printBest()
	    {
		    for(int i=0;i<swarm_size;i++)
		    {
			    Console.WriteLine("pBest[" + i + "]: "+pBest[i].Fitness + "; (" + pBest[i].Position.getValue(0) + "," + pBest[i].Position.getValue(1)+ ")");
			
		    }

            Console.WriteLine("gBest: " + gBest.Fitness);

		    Console.WriteLine("Number of Position Faults:" + position_fault);
	    }
    }//ParticleSwarm
