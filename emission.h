/*
 * emission.h
 * Supervised training of the emission WFST
 *
 *  Created on: Oct 23, 2019
 *      Author: Maria Ryskina
 */

#ifndef EMISSION_H_
#define EMISSION_H_

#include <list>
#include <math.h>

#include <fst/fstlib.h>

#include "base_fst.h"
#include "fst_utils.h"
#include "data_utils.h"

using namespace fst;

// Trainer for the emission model in the expectation semiring
class EmissionLogExpSemiring {
public:
	EmissionFst<ExpVecArc> fst;
	int targetAlphSize;
	int sourceAlphSize;
	int max_delay;

	// Initializing emission parameters with uniform + random noise (from seed)
	EmissionLogExpSemiring(int md, size_t trg_a, size_t src_a, int seed) :
		fst(md, trg_a, src_a, VecWeight::One()) {

		targetAlphSize = trg_a;
		sourceAlphSize = src_a;
		max_delay = md;
//		if (md == 0) {
//			EpsilonTotalFilter<ExpVecArc, NUM_EPS_TOTAL> epsFilterInput(targetAlphSize, fst.target_epsilon);
//			EpsilonTotalFilter<ExpVecArc, NUM_EPS_TOTAL> epsFilterOutput(sourceAlphSize, fst.source_epsilon);
//			Compose(epsFilterInput, fst, &fst);
//			Compose(fst, epsFilterOutput, &fst);
//		}

		VecWeight initLp = addNoise(fst.arcIndexer.size(), fst.getLogProbs(), seed);
		fst = EmissionFst<ExpVecArc>(md, trg_a, src_a, initLp);
	};

	// Initializing emission parameters with fixed values (e.g. from prior)
	EmissionLogExpSemiring(int md, size_t trg_a, size_t src_a, VecWeight lp = VecWeight::One()) :
		fst(md, trg_a, src_a, lp) {

		targetAlphSize = trg_a;
		sourceAlphSize = src_a;
		max_delay = md;
	};

	VecWeight train(std::vector<std::vector<int>> targetIndicesVector, std::vector<std::vector<int>> sourceIndicesVector,
			bool verbose = false, int max_iter = 1) {
		std::clock_t start;
		double elapsed;
		start = std::clock();

		std::cout << "Supervised model will be trained for " << max_iter << " iteration(s)" << std::endl;
		VecWeight emProbs;
		float mll = 0;
		float prevMll = INFINITY;
		int iter = 1;
		float convergenceThreshold = 1.05;
		int step = div(targetIndicesVector.size(), 10).quot;

		while (true) {
			std::cout << "ITERATION " << iter << std::endl;

			float numTokens = 0;
			int skipCount = 0;
			int diffSkipCount = 0;

			Adder<VecWeight> final;
			for (int i = 0; i < targetIndicesVector.size(); i++) {
				std::vector<int> targetIndices = targetIndicesVector[i];
				std::vector<int> sourceIndices = sourceIndicesVector[i];

				VectorFst<ExpVecArc> input = constructInput<ExpVecArc>(targetIndices, fst.target_epsilon);
				VectorFst<ExpVecArc> output = constructOutput<ExpVecArc>(sourceIndices, fst.source_epsilon);

				VectorFst<ExpVecArc> lattice;
				Compose<ExpVecArc>(input, (VectorFst<ExpVecArc>)fst, &lattice);
				Compose<ExpVecArc>(lattice, output, &lattice);

				if (lattice.NumStates() == 0) {
					int diff = sourceIndices.size() - targetIndices.size();
					if (diff > max_delay) diffSkipCount++;
					skipCount++;
					continue;
				}

				numTokens += sourceIndices.size();

				VecWeight out;
				LogWeight ll;

				// Collecting expected counts
				std::vector<ExpVecWeight> dist;
				ShortestDistance(lattice, &dist, true);
				out = dist[0].Value2();
				ll = dist[0].Value1();

				mll += ll.Value();
				final.Add(Divide(out, ll));

				if (((i+1) % step == 0 || i == targetIndicesVector.size() - 1) && verbose) {
					elapsed = (std::clock() - start) / (double) CLOCKS_PER_SEC;
					std::cout << "String pairs processed: " << (i + 1) << "; of them skipped: " <<
							skipCount << "; time elapsed: " << elapsed << std::endl;
				}
			}

			std::cout<<"Log-likelihood of training data: "<< mll << std::endl;
			if (skipCount > 0) {
				std::cout << "Skipped due to composition failure: " << skipCount <<
						" out of " << targetIndicesVector.size();
				std::cout << "; of them " << diffSkipCount << " due to delay over "
						<< max_delay << std::endl;
			}

			if (verbose) {
				elapsed = (std::clock() - start) / (double) CLOCKS_PER_SEC;
				std::cout << "Time elapsed: "<< elapsed << "; tokens per second: " <<
						numTokens / elapsed << std::endl;
			}

			// Normalizing expected counts collected over entire corpus
			emProbs = fst.normalize(final.Sum());
			fst = EmissionFst<ExpVecArc>(max_delay, targetAlphSize, sourceAlphSize, emProbs);
			if (prevMll - mll <= log(convergenceThreshold) || iter == max_iter) break;
			prevMll = mll;
			mll = 0;
			iter++;
		}

		return emProbs;
	}

protected:
	VecWeight addNoise(int numArcs, VecWeight lp, int seed) {
		srand(seed);
		double delta = 1e-2;

		for (int i = 0; i < numArcs; i++) {
			double f = (double)rand() / RAND_MAX;
			if (lp.Value(i) == LogWeight::One()) {
				lp.SetValue(i, - f * delta); // using Taylor approximation
			} else {
				LogWeight noise = -log(f * delta);
				lp.SetValue(i, Plus(lp.Value(i), noise));
			}
		}
		return fst.normalize(lp);
	}
};

class EmissionTropicalSemiring {
public:
	EmissionFst<StdArc> fst;
	int targetAlphSize;
	int sourceAlphSize;
	int max_delay;

	EmissionTropicalSemiring(int md, size_t trg_a, size_t la, VecWeight lp) :
		fst(md, trg_a, la, lp) {

		targetAlphSize = trg_a;
		sourceAlphSize = la;
		max_delay = md;
//		if (md == 0) {
//			EpsilonTotalFilter<StdArc, NUM_EPS_TOTAL> epsFilterInput(targetAlphSize, fst.target_epsilon);
//			EpsilonTotalFilter<StdArc, NUM_EPS_TOTAL> epsFilterOutput(sourceAlphSize, fst.source_epsilon);
//			Compose(epsFilterInput, fst, &fst);
//			Compose(fst, epsFilterOutput, &fst);
//		}
	}

	// Collecting a set of all allowed emission labels
	std::vector<bool> getOIndices() {
		std::vector<bool> res(sourceAlphSize + 1, false);
		for (ArcIterator<VectorFst<StdArc>> aiter(fst, fst.Start()); !aiter.Done(); aiter.Next()) {
			const StdArc &arc = aiter.Value();
			res[arc.olabel] = true;
		}
		return res;
	}
};

// Training the emission WFST on supervised data
EmissionTropicalSemiring trainEmission(IndexedStrings data, int max_delay, int targetAlphSize, int sourceAlphSize,
		int seed, std::string output_dir, bool no_save = false, int max_iter = 1) {

	EmissionLogExpSemiring logExpEm(max_delay, targetAlphSize, sourceAlphSize, seed);
	VecWeight emProbs = logExpEm.train(data.targetIndices, data.sourceIndices, true, max_iter);

	std::cout << "Emission model (expectation semiring): " << logExpEm.fst.NumStates() << " states, "
			<< logExpEm.fst.getNumArcs() << " arcs\n";
	EmissionTropicalSemiring tropicalEm(max_delay, targetAlphSize, sourceAlphSize, emProbs);
	std::cout << "Emission model (tropical semiring): " << tropicalEm.fst.NumStates() << " states, "
			<< tropicalEm.fst.getNumArcs() << " arcs\n";

	if (!no_save) {
		std::string emission_outfile = output_dir + "/emission.fst";
		std::cout << "Saving the trained emission model to: " << emission_outfile << std::endl;
		tropicalEm.fst.Write(emission_outfile);
	}

	return tropicalEm;
}

#endif /* EMISSION_H_ */

