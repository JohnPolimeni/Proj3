
/*
 * Created By: John Polimeni, Isaac Wong, Luke Gerstner
 * Created On: November 5, 2017
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

public class MyInferencer {

	/*
	 * enumerationAsk algorithm comes from AIMA Figure 14.9 (pg. 525)
	 */
	public Distribution enumerationAsk(BayesianNetwork bn, RandomVariable X, Assignment e) {
		Distribution d = new Distribution();
		// Go through each value of X
		for (int i = 0; i < X.getDomain().size(); i++) {
			Assignment a = e.copy();
			// set the assignment with a random variable as its key and an
			// element from the domain of the random variable to its value.
			a.set(X, X.getDomain().get(i));
			d.put(X.getDomain().get(i), enumerateAll(bn, bn.getVariableListTopologicallySorted(), a));
		}

		// return the normalized distribution
		d.normalize();
		return d;

	}

	public Double enumerateAll(BayesianNetwork bn, List<RandomVariable> list, Assignment e) {
		// Check if list is empty
		if (list.size() == 0) {
			return 1.0;
		}
		// Assign Y as the first variable from the Bayesian Network's list
		RandomVariable Y = list.get(0);

		// Create a new list with everything, but the first element in it
		ArrayList<RandomVariable> rest = new ArrayList<RandomVariable>(list.subList(1, list.size()));

		// Check to see if e has the value Y in it
		if (e.containsKey(Y)) {

			double q = bn.getProb(Y, e);
			double w = enumerateAll(bn, rest, e);
			return q * w;

		} else {
			double sum = 0.0;
			for (int i = 0; i < Y.getDomain().size(); i++) {

				e.put(Y, Y.getDomain().get(i));

				Assignment a = e.copy();
				a.put(Y, Y.getDomain().get(i));

				sum += bn.getProb(Y, a) * enumerateAll(bn, rest, a);
			}
			return sum;
		}

	}

	/*
	 * rejectionSampling algorithm comes from AIMA Figure 14.14 (pg. 533)
	 */
	public Distribution rejectionSampling(RandomVariable X, Assignment e, BayesianNetwork bn, int n) {
		Distribution d = new Distribution();
		for (Object o : X.getDomain()) {
			d.put(o, 0);
		}

		for (int i = 0; i < n; i++) {
			Assignment x = priorSample(bn);
			// reset the boolean variable each time through the loop, very
			// important!!!!!
			boolean accept = true;
			// Check if the assignment is consistent with the evidence
			for (Entry<RandomVariable, Object> ev : e.entrySet()) {
				if (!ev.getValue().equals(x.get(ev.getKey()))) {
					// reject if it does not agree with the evidence
					accept = false;
					break;
				}
			}
			if (accept) {
				d.put(x.get(X), d.get(x.get(X)) + 1);
			}
		}
		// normalize the function then return it
		d.normalize();
		return d;
	}

	// Helper method to be used in rejectionSampling
	public Assignment priorSample(BayesianNetwork bn) {
		Assignment x = new Assignment();
		List<RandomVariable> list = bn.getVariableListTopologicallySorted();
		for (int i = 0; i < bn.size(); i++) {
			// Choose variable from the Bayesian Network
			RandomVariable random = list.get(i);
			ArrayList<Double> weight = new ArrayList<>();
			for (int j = 0; j < random.getDomain().size(); j++) {
				// Set the assignment to the random variable with a value of
				// true or false
				x.set(random, random.getDomain().get(j));
				// Holds conditional probability table
				weight.add(bn.getNodeForVariable(random).cpt.get(x));

			}
			// Create a random number
			double rand = Math.random();
			double sum = 0.0;
			for (int k = 0; k < weight.size(); k++) {
				sum += weight.get(k);
				if (rand <= sum) {
					x.put(random, random.getDomain().get(k));
					break;
				}
			}
		}
		return x;
	}

	/*
	 * likelihoodWeighting algorithm comes from AIMA Figure 14.15 (pg. 534)
	 */

	// instance variables for the likelihoodWeighting algorithm
	public static Assignment globalAssignment = new Assignment();
	public static double globalWeight = 1.0;

	/*
	 * likelihoodWeighting algorithm comes from AIMA Figure 14.15 (pg. 534)
	 * 
	 * inputs: X, the query variable e, observed values for variables in E bn, a
	 * Bayesian network specifying joint distribution P(X1, ... , Xn)
	 */
	public Distribution likelihoodWeighting(RandomVariable X, Assignment e, BayesianNetwork bn, int numSamples) {

		// W from book
		// a vector of weighted counts for each value of X, intitally set to
		// zero
		Distribution d = new Distribution();

		// set all the values in the distribution to zero
		for (int i = 0; i < X.getDomain().size(); i++) {
			d.put(X.getDomain().get(i), 0);
		}

		// iterate through number of samples
		for (int i = 0; i < numSamples; i++) {

			// change the global field for globalAssignment and globalWeight
			// this calculates the event and weight
			weightedSample(bn, e);

			// get the value of the global assignemnt
			double curr = d.get(globalAssignment.get(X));

			// for the given variable X for the sampled assignment, make its
			// weight equal to the sum of its current weight plus the newly
			// calcualted weight
			// W[X] = W[X] + weight
			d.put(globalAssignment.get(X), curr + globalWeight);
		}

		// normalize the distribution
		d.normalize();
		return d;
	}

	// Helper method for likelihoodWeighting
	// sets the assignment event and it's weight
	public void weightedSample(BayesianNetwork bn, Assignment e) {

		// reset the values of the global assignment and weight fields to
		// initial values
		globalAssignment = new Assignment();
		globalWeight = 1.0;

		// for each randomVariable in the bayseian network, traversing the
		// network topologically
		for (RandomVariable rv : bn.getVariableListTopologicallySorted()) {

			// if the assignemnt contains the variable in question
			if (e.containsKey(rv)) {

				// Xi is an evidence variable with the value xi in e
				// set the globalAssignment assignment by mapping the random
				// variable to the value of that random variable in the original
				// assignment
				globalAssignment.put(rv, e.get(rv));

				// weight = weight * P(Xi == xi | Parents(Xi))
				globalWeight *= bn.getProb(rv, globalAssignment);

				// evidence variable does not have value xi in e
			} else {

				// create a new temporary distribution object to store the
				// random sample from P(Xi | parents(Xi))
				Distribution d = new Distribution();
				// for every domain in the random variable rv
				for (int i = 0; i < rv.domain.size(); i++) {

					// create a sample assignment, initially with values set to
					// the global assignment
					Assignment sample = globalAssignment.copy();

					// set it's values to equal that of the random variable
					sample.set(rv, rv.getDomain().get(i));

					// get the probability of the random variable from the
					// generated assignment
					double prob = bn.getProb(rv, sample);

					// map the random variable's values to the generated
					// probability
					// value is stored in the generated distribution
					d.put(rv.getDomain().get(i), prob);

				}

				// normalize the distribution
				d.normalize();

				// create a sum variable, initially set to zero
				double sum = 0;

				// get a random number between 0 and 1
				double rand = Math.random();

				// iterate through every value of the random domain
				for (int i = 0; i < rv.getDomain().size(); i++) {

					// increase the sum by the probability of getting that
					// variable
					sum += d.get(rv.getDomain().get(i));

					// sum will increasingly get bigger
					// as it gets bigger, there is a greater chance that it will
					// catch at this if statement
					if (rand <= sum) {

						// the sum is sufficiently large enough based on the
						// randomly generated number
						// set the global assignment's value at the random
						// variable to what it is at this iteration of the for
						// loop
						globalAssignment.set(rv, rv.getDomain().get(i));
						break;
					}
				}
			}
		}
		// System.out.println(" 'returned' " + globalAssignment.toString());
		// doesn't return anything , instead stores it's result in the global
		// assignment
		return;
	}

	public Distribution Gibbs(RandomVariable X, Assignment e, BayesianNetwork bn, int numSamples) {

		// N, a vector of counts for each value of X , initially zero
		Distribution N = new Distribution();
		for (int i = 0; i < X.getDomain().size(); i++) {
			N.put(X.getDomain().get(i), 0);
		}

		// Z, the nonevidence variables in bn
		ArrayList<RandomVariable> Z = new ArrayList<RandomVariable>(bn.getVariableListTopologicallySorted());
		for (RandomVariable rv : e.keySet()) {
			Z.remove(rv);
		}

		// x, the current state of the network, initially copied from e
		Assignment x = e.copy();

		// initialize x with random values for the variables in Z
		for (RandomVariable Zi : Z) {

			x.put(Zi, Zi.getDomain().get(new Random().nextInt(Zi.getDomain().size())));
			// x.put(Zi, rand);
		}

		for (int i = 0; i < numSamples; i++) {
			for (RandomVariable Zi : Z) {
				Distribution d = new Distribution();
				double sum = 0.0;
				for (int j = 0; j < X.getDomain().size(); j++) {
					d.put(X.getDomain().get(j), 0);
				}
				for (Object o : Zi.getDomain()) {
					// make parents of Zi
					x.set(Zi, o);

					// calculate P(Zi|mb(Zi))
					double product = bn.getProb(Zi, x);

					// ArrayList<RandomVariable> yj = new
					// ArrayList<RandomVariable>(x.keySet());
					for (BayesianNetwork.Node n : bn.getNodeForVariable(Zi).children) {
						product *= bn.getProb(n.variable, x);
					}
					d.put(o, product);
				}
				d.normalize();
				Object value = null;
				for (Object o : Zi.getDomain()) {
					x.put(Zi, o);
					sum += d.get(o);

					if (Math.random() <= sum) {
						value = o;
						x.put(Zi, value);
						break;
					}
				}

				N.put(x.get(Zi), N.get(x.get(Zi)) + 1);

			}
		}

		N.normalize();
		return N;

	}

}
