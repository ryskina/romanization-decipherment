/*
 * model.h
 * Unsupervised training of the main WFST model
 *
 *  Created on: Nov 20, 2019
 *      Author: Maria Ryskina
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <list>
#include <math.h>
#include <ctime>
#include <dirent.h>

#include <fst/fstlib.h>

#include "data_utils.h"
#include "fst_utils.h"
#include "base_fst.h"
#include "emission.h"
#include "lm.h"

// Base class for the unsupervised WFST model
class Base {
public:
	VectorFst<StdArc> lmStd;
	EmissionFst<StdArc> emStd;
	EditDistanceFst evalFst;
	int origAlphSize;
	int latinAlphSize;
	int orig_epsilon;
	int latin_epsilon;
	int max_delay;
	bool no_epsilons = false;
	float freeze_at = -1;

	// Marking symbols not seen in training for deletion
	std::vector<bool> latinDelMask;

	Base(int md, int oa, int la, bool ne = false, float fa = -1) :
		emStd(md, oa, la, VecWeight::One(), fa), evalFst(oa), latinDelMask(la, true) {

		origAlphSize = oa;
		latinAlphSize = la;
		orig_epsilon = oa + 1;
		latin_epsilon = la + 1;
		max_delay = md;
		no_epsilons = ne;
		freeze_at = fa;
	}

	// Decode an indexed Latin string into an indexed original script string
	std::vector<int> decode(std::vector<int> latinIndices, LmComposeType composeType = PHI_MATCH,
			bool verbose = false) {
		VectorFst<StdArc> output;
		if (!no_epsilons) {
			output = constructOutput<StdArc>(latinIndices, latin_epsilon);
		} else {
			output = constructAcceptor<StdArc>(latinIndices);
		}
		VectorFst<StdArc> lattice;

		Compose<StdArc>(emStd, output, &lattice);
		if (!composeCheck(&lattice, "emission", verbose)) return std::vector<int>();

		if (composeType == PHI_MATCH) {
			lmPhiCompose<StdArc>(lmStd, lattice, &lattice);
		} else {
			Compose<StdArc>(lmStd, lattice, &lattice);
			RmEpsilon(&lattice);
			Determinize(lattice, &lattice);
		}
		if (!composeCheck(&lattice, "lm rescored", verbose)) return std::vector<int>();

		VectorFst<StdArc> path;
		ShortestPath(lattice, &path);
		if (!pathCheck(&path)) return std::vector<int>();

		std::vector<int> out;
		int curState = path.Start();

		while (true) {
			assert (path.NumArcs(curState) <= 1);
			if (path.NumArcs(curState) == 0) {
				assert (path.Final(curState) != TropicalWeight::Zero());
				break;
			}
			for (ArcIterator<VectorFst<StdArc>> aiter(path, curState); !aiter.Done(); aiter.Next()) {
				const StdArc &arc = aiter.Value();
				curState = arc.nextstate;
				if (arc.ilabel != 0 && arc.ilabel != orig_epsilon) out.push_back(arc.ilabel);
			}
		}
		return out;
	}

	// Decode all strings the test set and compute CER
	float test(IndexedStrings testData, LmComposeType composeType = PHI_MATCH, bool verbose = false,
			std::string outfile_path = "") {
		float distSum = 0;
		float goldLensSum = 0;
		int failCount = 0;

		std::ofstream outfile;
		if (outfile_path != "") outfile.open(outfile_path);

		for (int i = 0; i < testData.latinIndices.size(); i++) {

			std::vector<int> origIndices = testData.origIndices[i];
			std::vector<int> latinIndices = testData.latinIndices[i];

			if (verbose) {
				std::cout << std::endl;
				std::cout << "Latin:      " <<
						testData.latinIndexerPtr->encode(latinIndices) << std::endl;
			}

			int l = latinIndices.size();
			for (int pos = l - 1; pos >= 0 ; pos--) {
				if (!latinDelMask[latinIndices[pos]]) latinIndices.erase(latinIndices.begin() + pos);
			}
			if (latinIndices.size() < l && verbose) {
				std::cout << "Filtered:   " <<
						testData.latinIndexerPtr->encode(latinIndices) << std::endl;
			}

			std::vector<int> predicted = decode(latinIndices, composeType, verbose);
			if (predicted.size() == 0) failCount++;

			distSum += DataUtils::eval(predicted, origIndices, &evalFst);
			goldLensSum += origIndices.size();
			std::string out = testData.origIndexerPtr->encode(predicted);

			if (verbose) {
				std::cout << "Prediction: " << out << std::endl;
				std::cout << "Gold:       " << testData.origIndexerPtr->encode(origIndices) << std::endl;
			}

			if (outfile.is_open()) {
				outfile << out << std::endl;
			}

		}
		if (outfile.is_open()) outfile.close();
		std::cout << "Failed to compose " << failCount << " out of " << testData.latinIndices.size() << std::endl;
		return distSum / goldLensSum;
	}
};

// Unsupervised WFST model trainer class
class Trainer : public Base{
public:
	VectorFst<ExpVecArc> lmExp;
	EmissionFst<ExpVecArc> emExp;
	std::vector<VectorFst<StdArc>> lmStdArray;

	Indexer* origIndexerPtr;
	Indexer* latinIndexerPtr;

	VecWeight prior = VecWeight::Zero();
	VecWeight mu;
	float alpha = 0.9;
	float pseudoCount = 0;

	Trainer(std::vector<VectorFst<StdArc>> lmFstArray, int md, Indexer* oIPtr, Indexer *lIPtr, int seed,
			std::vector<std::pair<int, int>> priorMappings, bool no_epsilons = false, float freeze_at = -1) :
		Base(md, oIPtr->getSize(), lIPtr->getSize(), no_epsilons, freeze_at),
		emExp(md, oIPtr->getSize(), lIPtr->getSize(), VecWeight::One(), freeze_at) {

		origIndexerPtr = oIPtr;
		latinIndexerPtr = lIPtr;

		lmStdArray = lmFstArray;

		mu.SetDefaultValue(pseudoCount);
		// Increasing the elements of mu corresponding to the mappings encoded in priors
		for (int i = 0; i < emExp.arcIndexer.size(); i++) {
			for (std::pair<int, int> &symbolPair : priorMappings) {
				if (emExp.arcIndexer[i].first == symbolPair.first && emExp.arcIndexer[i].second == symbolPair.second) {
					mu.SetValue(i, Plus(mu.Value(i), LogWeight(pseudoCount)));
					prior.SetValue(i, LogWeight(pseudoCount));
				}
			}
		}

		lmStd = lmStdArray[0];
		Map(lmStd, &lmExp, WeightConvertMapper<StdArc, ExpVecArc>());

		VecWeight emProbs = addNoise(emExp.NumArcs(0), mu, seed);
		emExp = EmissionFst<ExpVecArc>(max_delay, origAlphSize, latinAlphSize, emProbs);
		emStd = EmissionFst<StdArc>(max_delay, origAlphSize, latinAlphSize, emProbs);
	}

	void train(std::vector<std::vector<int>> latinIndicesVector, IndexedStrings devData, IndexedStrings testData,
			std::string output_dir, int batchSize, int upgrade_lm_every, int upgrade_lm_by,
			LmComposeType composeType = PHI_MATCH, bool verbose=false, bool no_save=false) {

		if (batchSize > latinIndicesVector.size()) batchSize = latinIndicesVector.size();

		int order = 2;
		int k = 0;
		float prevDevCer = 100;

		std::string emission_outfile = output_dir + "/emission.fst";
		std::string model_outfile = output_dir + "/model.fst";
		std::string test_prediction_file = output_dir + "/test_prediction.txt";
		std::string dev_prediction_file = output_dir + "/dev_prediction.txt";

		float numTokens = 0;
		float mll = 0;

		std::clock_t start;
		double elapsed;
		start = std::clock();

		Adder<VecWeight> final;

		for (int i = 0; i < latinIndicesVector.size(); i++) {
			std::vector<int> latinIndices = latinIndicesVector[i];
			numTokens += latinIndices.size();

			VectorFst<ExpVecArc> output;
			if (!no_epsilons) {
				output = constructOutput<ExpVecArc>(latinIndices, latin_epsilon);
			} else {
				output = constructAcceptor<ExpVecArc>(latinIndices);
			}

			VectorFst<ExpVecArc> lattice;
			Compose<ExpVecArc>(emExp, output, &lattice);
			if (!composeCheck(&lattice, "emission")) continue;

			if (composeType == PHI_MATCH) {
				lmPhiCompose<ExpVecArc>(lmExp, lattice, &lattice);
			} else {
				Compose<ExpVecArc>(lmExp, lattice, &lattice);
				RmEpsilon(&lattice);
			}

			if (!composeCheck(&lattice, "lm rescore")) continue;
			if (verbose) printStats(&lattice, "Lattice: ");

			VecWeight unnormCounts;
			LogWeight ll;

			std::vector<ExpVecWeight> dist;
			ShortestDistance(lattice, &dist, true);
			if (dist.size() == 0) {
				if (verbose) std::cout << "FAILED TO COMPUTE COUNTS" << std::endl;
				continue;
			}
			// Collecting expected counts in the expectation semiring
			unnormCounts = dist[0].Value2();
			ll = dist[0].Value1();
			mll += ll.Value();
			final.Add(Divide(unnormCounts, ll));

			if (((i+1) % batchSize == 0) || i+1 == latinIndicesVector.size()) {
				// Performing a stepwise EM update after each batch
				final.Add(prior);
				float logEta = -alpha * log(k + 2);
				float logCoef = log(1 - exp(logEta));
				k++;
				mu = Plus(Times(LogWeight(logCoef), mu), Times(LogWeight(logEta), final.Sum()));
				VecWeight emProbs;
				if (freeze_at >= 0) {
					emProbs = emExp.normalizeFrozen(mu, freeze_at, orig_epsilon, latin_epsilon);
				} else {
					emProbs = emExp.normalize(mu);
				}

				// Pruning the low-probability emission arcs
				int pruned = 0;
				float minThr = 4.5;
				float thr = 5 - 0.05 * k;
				if (thr < minThr) thr = minThr;
				for (int arcIdx = 0; arcIdx < emExp.arcIndexer.size(); arcIdx++) {
					if (emExp.arcIndexer[arcIdx].first != orig_epsilon && emExp.arcIndexer[arcIdx].second != latin_epsilon) {
						if (emProbs.Value(arcIdx) == LogWeight::Zero() || emProbs.Value(arcIdx).Value() > thr) {
							emProbs.SetValue(arcIdx, LogWeight::Zero());
							mu.SetValue(arcIdx, LogWeight::Zero());
							pruned++;
						}
					}
				}
				std::cout << "Pruned " << pruned << " arcs out of " << emExp.arcIndexer.size() << "; thr = " << thr << std::endl;

				// Normalizing mu to get emission probabilities
				if (freeze_at >= 0) {
					emProbs = emExp.normalizeFrozen(mu, freeze_at, orig_epsilon, latin_epsilon);
				} else {
					emProbs = emExp.normalize(mu);
				}

				emExp = EmissionFst<ExpVecArc>(max_delay, origAlphSize, latinAlphSize, emProbs);
				emStd = EmissionFst<StdArc>(max_delay, origAlphSize, latinAlphSize, emProbs);
				std::cout << "Log-likelihood of training mini-batch: " << mll << std::endl;

				elapsed = (std::clock() - start) / (double) CLOCKS_PER_SEC;
				if (verbose) std::cout << "String pairs processed: " << i + 1  << "; time elapsed: " << elapsed << std::endl;
				mll = 0;
				numTokens = 0;

				if ((i+1) % batchSize == 0 || i+1 == latinIndicesVector.size()) {
					// Evaluating on validation data after every batch
					float devCer = test(devData, composeType, false, dev_prediction_file);
					std::cout << "Validation data CER: " << devCer << std::endl;

					// Evaluating on test data if the validation CER is less than the previous best value
					if (testData.latinIndices.size() > 0 && devCer <= prevDevCer) {
						if (!no_save) {
							std::cout << "Saving the trained emission model to: " << emission_outfile << std::endl;
							emStd.Write(emission_outfile);
							std::cout << "Composing and saving the base lattice to: " << model_outfile << std::endl;
							VectorFst<StdArc> allStatesLattice;
							lmPhiCompose<StdArc>(lmStd, emStd, &allStatesLattice);
							allStatesLattice.Write(model_outfile);
						}
						std::cout << "Evaluating on test data\n";
						float testCer = test(testData, composeType, false, test_prediction_file);
						std::cout << "Test data CER: " << testCer << std::endl;
						prevDevCer = devCer;
					}
				}

				// Increasing the language model order
				if ((i+1) % (upgrade_lm_every * batchSize) == 0) {
					int current_order = order;
					// If insertion/deletion probabilities are frozen, we unfreeze them when increasing LM order
					if (current_order == 2 && freeze_at >= 0) {
						std::cout << "Unfreezing insertions and deletions\n";
						freeze_at = -1;
						emProbs = emExp.normalize(mu);
						emExp = EmissionFst<ExpVecArc>(max_delay, origAlphSize, latinAlphSize, emProbs);
						emStd = EmissionFst<StdArc>(max_delay, origAlphSize, latinAlphSize, emProbs);
					}
					order += upgrade_lm_by;
					if (order > 6) order = 6;
					if (order > current_order) {
						std::cout << "Increasing LM order to " << order << std::endl;
						lmStd = lmStdArray[order - 2];
						Map(lmStd, &lmExp, WeightConvertMapper<StdArc, ExpVecArc>());
					}
				}
			}
		}

		if (verbose) {
			elapsed = (std::clock() - start) / (double) CLOCKS_PER_SEC;
			std::cout<<"Time elapsed: "<< elapsed << "; tokens per second: "<< numTokens / elapsed << std::endl;
		}
	}

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
		mu = lp;
		if (freeze_at >= 0) return emExp.normalizeFrozen(lp, freeze_at, orig_epsilon, latin_epsilon);
		return emExp.normalize(lp);
	}
};

class Model : public Base {
public:
	Model(VectorFst<StdArc> lmFst, EmissionTropicalSemiring em) :
		Base(em.fst.max_delay, em.origAlphSize, em.latinAlphSize) {

		latin_epsilon = em.latinAlphSize + 1;
		orig_epsilon = em.origAlphSize + 1;

		latinDelMask = em.getOIndices();

		this->lmStd = lmFst;
		this->emStd = em.fst;
	}

	Model(VectorFst<StdArc> lmFst, EmissionFst<StdArc> emFst, int oa, int la) :
		Base(emFst.max_delay, oa, la) {

		this->lmStd = lmFst;
		this->emStd = emFst;
	}
};

#endif /* MODEL_H_ */
