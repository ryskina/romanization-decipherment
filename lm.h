/*
 * lm.h
 * Training the language model WFST
 *
 *  Created on: Oct 15, 2019
 *      Author: Maria Ryskina
 */

#ifndef LM_H_
#define LM_H_

#include <list>
#include <math.h>

#include <fst/fstlib.h>
#include <ngram/ngram.h>

#include "base_fst.h"
#include "fst_utils.h"
#include "data_utils.h"

using namespace fst;
using namespace ngram;

// Specifies the way to compose LM with the lattice
enum LmComposeType {
  PHI_MATCH = 1,      // treating backoff arcs as failure arcs
  EPSILON_MATCH = 2,  // treating backoff arcs as epsilon transitions
};

VectorFst<StdArc> trainLmOpenGRM(IndexedStrings trainData, int targetAlphSize, int order, std::string output_dir,
		bool no_save = false, bool no_epsilons = false, int trainDataSize = 1000000) {
	std::clock_t start;
	double elapsed;
	start = std::clock();

	if (trainDataSize > trainData.targetIndices.size()) {
		trainDataSize = trainData.targetIndices.size();
	}

	int step = div(trainDataSize, 10).quot;
	NGramCounter<LogWeight> counter(order);

	for (int i = 0; i < trainDataSize; i++) {
		std::vector<int> indices = trainData.targetIndices[i];
		VectorFst<LogArc> input = constructAcceptor<LogArc>(indices);

		counter.Count(input);
		if ((i + 1) % step == 0 || i + 1 == trainDataSize) {
			elapsed = (std::clock() - start) / (double) CLOCKS_PER_SEC;
			std::cout << "String pairs processed: " << i + 1 << "; time elapsed: " << elapsed << std::endl;
		}

	}
	VectorFst<LogArc> countFst;
	counter.GetFst(&countFst);
	ArcSort(&countFst, ILabelCompare<LogArc>());

	VectorFst<StdArc> convCounts;
	Map(countFst, &convCounts, WeightConvertMapper<LogArc, StdArc>());
	NGramWittenBell lm(&convCounts, false, 0, kNormEps, true, 10);
	lm.MakeNGramModel();
	VectorFst<StdArc> lmFst = (VectorFst<StdArc>)lm.GetFst();

	int numArcs = 0;
	for (int state = 0; state < lmFst.NumStates(); state++) numArcs += lmFst.NumArcs(state);
	std::cout << "Done training a " << order << "-gram language model\n";
	std::cout << "Before pruning: " << lmFst.NumStates() << " states, " << numArcs << " arcs\n";

	if (order == 3) {
		NGramRelEntropy pruning(&lmFst, 1e-5);
		pruning.ShrinkNGramModel();
	} else if (order > 3) {
		NGramRelEntropy pruning(&lmFst, 2e-5);
		pruning.ShrinkNGramModel();
	}

	numArcs = 0;
	for (int state = 0; state < lmFst.NumStates(); state++) numArcs += lmFst.NumArcs(state);
	std::cout << "After pruning: " << lmFst.NumStates() << " states, " << numArcs << " arcs\n";

	if (!no_save) {
		std::string lm_outfile = output_dir + "/lm_" + std::to_string(order) + ".fst";
		std::cout << "Saving the pruned language model to: " << lm_outfile << std::endl;
		lmFst.Write(lm_outfile);
	}

	if (!no_epsilons) {
		// Adding target_epsilon (insertion) loop for each LM state
		for (int i = 0; i < lmFst.NumStates(); i++) {
			lmFst.AddArc(i, StdArc(targetAlphSize+1, targetAlphSize+1, TropicalWeight::One(), i));
		}
	}

	numArcs = 0;
	for (int state = 0; state < lmFst.NumStates(); state++) numArcs += lmFst.NumArcs(state);
	std::cout << "Final: " << lmFst.NumStates() << " states, " << numArcs << " arcs\n";

	// Based on the OpenGRM "ngramapply" command
	lmFst.SetInputSymbols(new SymbolTable());
	NGramOutput ngram(&lmFst);
	ngram.MakePhiMatcherLM(kSpecialLabel);
	lmFst = ngram.GetFst();

	return lmFst;
}

// Reimplementation of OpenGRM Phi-composition
template<class A>
void lmPhiCompose(VectorFst<A> lmFst, VectorFst<A> infst, VectorFst<A> *ofst) {
	FLAGS_fst_compat_symbols = false;

	using NGPhiMatcher = PhiMatcher<Matcher<Fst<A>>>;
	ComposeFstOptions<A, NGPhiMatcher> opts(
	    CacheOptions(), new NGPhiMatcher(lmFst, MATCH_OUTPUT, kSpecialLabel, 1, MATCHER_REWRITE_NEVER),
				new NGPhiMatcher(infst, MATCH_NONE, kNoLabel));
    *ofst = ComposeFst<A>(lmFst, infst, opts);
}

#endif /* LM_H_ */
