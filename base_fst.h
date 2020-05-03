/*
 * base_fst.h
 * Parameterization of the emission WFST
 *
 *  Created on: Oct 27, 2019
 *      Author: Maria Ryskina
 */

#ifndef BASE_FST_H_
#define BASE_FST_H_

#include <assert.h>

#include <fst/fstlib.h>

#include "fst_utils.h"

using namespace fst;

using ExpVecArc = ExpectationArc<LogArc, SparsePowerWeight<LogWeight>>;
using ExpVecWeight = ExpectationWeight<LogWeight, SparsePowerWeight<LogWeight>>;
using VecWeight = SparsePowerWeight<LogWeight>;


// Arc weight converter from LogWeight to W
// arcIndex is required for conversion to ExpVecWeight
template<class W>
W wrapArcWeight(LogWeight arcProb, int arcIndex) {};

template<>
inline ExpVecWeight wrapArcWeight<ExpVecWeight>(LogWeight arcProb, int arcIndex) {
	SparsePowerWeight<LogWeight> basisVector =
			SparsePowerWeight<LogWeight>(arcIndex, arcProb, LogWeight::Zero());
	return ExpVecWeight(arcProb, basisVector);
}

template<>
inline TropicalWeight wrapArcWeight<TropicalWeight>(LogWeight arcProb, int arcIndex) {
	WeightConvert<LogWeight, TropicalWeight> conv;
	return conv(arcProb);
}

template<>
inline LogWeight wrapArcWeight<LogWeight>(LogWeight arcProb, int arcIndex) {
	return arcProb;
}

// Base class for the emission WFSTs
template<class A>
class BaseFst : public VectorFst<A> {
using W = typename A::Weight;
public:
	std::vector<std::pair<int, int>> arcIndexer; // stores input-output label pairs for each arc
	VecWeight logProbs; 	                     // vector of arc weights, stored in order of indexing

	W getArcWeight(int arcIndex, float frozenProb = -1) {
		LogWeight arcProb = logProbs.Value(arcIndex);
		if (frozenProb >= 0) {
			arcProb = LogWeight(frozenProb);
			logProbs.SetValue(arcIndex, arcProb);
		}
		return wrapArcWeight<W>(arcProb, arcIndex);
	}

	void printVector(VecWeight out) {
		for (int i=0; i < arcIndexer.size(); i++) {
			// Not printing zero values
			if (out.Value(i) != LogWeight::Zero()) {
				std::cout << "arc "<< i << ": " << arcIndexer[i].first << "->" << arcIndexer[i].second;
				std::cout << "; negative log count " << out.Value(i) << std::endl;
			}
		}
	}

	void printProbsWithLabels(VecWeight out, Indexer *oIPtr, Indexer *lIPtr,
			int o_eps, int l_eps, float min_value = 0) {
		std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> cv;

		for (int i=0; i < arcIndexer.size(); i++) {
			// Not printing zero values
			if (out.Value(i) != LogWeight::Zero()) {
				float prob = exp(-out.Value(i).Value());
				// Not printing values under min_value
				if (prob > min_value) {
					std::cout << "arc "<< i << ": ";
					if (arcIndexer[i].first == o_eps) {
						std::cout << "<o_eps>->";
					} else {
						std::cout << cv.to_bytes(oIPtr->lookup[arcIndexer[i].first]) << "->";
					}
					if (arcIndexer[i].second == l_eps) {
						std::cout << "<l_eps>";
					} else {
						std::cout << cv.to_bytes(lIPtr->lookup[arcIndexer[i].second]);
					}
					std::cout << "; log probability " << exp(-out.Value(i).Value()) << std::endl;
				}
			}
		}
	}

	// Normalize expected counts by input label
	VecWeight normalize(VecWeight counts) {
		VecWeight out;
		Adder<VecWeight> z; //normalizer

		for (int i = 0; i < arcIndexer.size(); i++) {
			int idxFrom = arcIndexer[i].first;
			VecWeight convertedCount = VecWeight(idxFrom, counts.Value(i), LogWeight::Zero());
			z.Add(convertedCount);
		}

		for (int i = 0; i < arcIndexer.size(); i++) {
			int idxFrom = arcIndexer[i].first;
			if (z.Sum().Value(idxFrom) == LogWeight::Zero()) {
				out.SetValue(i, LogWeight::Zero());
			} else {
				LogWeight val = Divide(counts.Value(i), z.Sum().Value(idxFrom));
				out.SetValue(i, val);
			}
		}
		return out;
	}

	// Normalize expected counts by input label,
	// with freezing insertion and deletion probabilities at freezeProb
	VecWeight normalizeFrozen(VecWeight counts, float freezeProb, int o_eps, int l_eps) {
		VecWeight out;
		Adder<VecWeight> z; //normalizer
		LogWeight normalizerAdjustment = LogWeight(-log1p(-exp(-freezeProb)));

		for (int i = 0; i < arcIndexer.size(); i++) {
			int idxFrom = arcIndexer[i].first;
			int idxTo = arcIndexer[i].second;
			if (idxFrom == o_eps || idxTo == l_eps) continue;
			VecWeight convertedCount = VecWeight(idxFrom, counts.Value(i), LogWeight::Zero());
			z.Add(convertedCount);
		}

		for (int i = 0; i < arcIndexer.size(); i++) {
			int idxFrom = arcIndexer[i].first;
			int idxTo = arcIndexer[i].second;
			if (idxFrom == o_eps || idxTo == l_eps){
				out.SetValue(i, LogWeight(freezeProb));
			} else if (z.Sum().Value(idxFrom) == LogWeight::Zero()) {
				out.SetValue(i, LogWeight::Zero());
			} else {
				LogWeight val = Divide(counts.Value(i), z.Sum().Value(idxFrom));
				val = Times(val, normalizerAdjustment);
				out.SetValue(i, val);
			}
		}
		return out;
	}

	VecWeight getLogProbs() {return logProbs;}

	void setLogProbs(VecWeight lp) {logProbs = lp;}

	int getNumArcs() {
		int out = 0;
		for (int state = 0; state < this->NumStates(); state++) {
			out += this->NumArcs(state);
		}
		return out;
	}
};

// Class for setting arc configurations of the emission WFST
template<class A>
class EmissionFst : public BaseFst<A> {
using W = typename A::Weight;
using VecWeight = SparsePowerWeight<LogWeight>;
public:
	int max_delay;     // maximum allowed delay of a path
	int latin_epsilon; // special deletion symbol index
	int orig_epsilon;  // special insertion symbol index

	EmissionFst(int max_delay, size_t origAlphSize, size_t latinAlphSize, VecWeight lp,
			float freezeProb = -1) {
		this->setLogProbs(lp);
		latin_epsilon = latinAlphSize + 1;
		orig_epsilon = origAlphSize + 1;
		this->max_delay = max_delay;

		// Creating states
		for (int state = 0; state < 2 * max_delay + 1; state++) {
			this->AddState();
			this->SetFinal(state, W::One());
		}

		this->SetStart(max_delay);

		// Creating the emission arcs
		for (int i = 1; i < origAlphSize + 1; i++) {
			for (int j = 1; j < latinAlphSize + 1; j++) {
				addEmissionArc(i, j);
			}
			// Creating deletion arcs (original symbol -> latin_epsilon)
			addDeletionArc(i, freezeProb);
		}

		// Creating insertion arcs (original_epsilon -> Latin symbol)
		for (int j = 1; j < latinAlphSize + 1; j++) {
			addInsertionArc(j, freezeProb);
		}
	}

protected:
	void addEmissionArc(int ilabel, int olabel) {
		// Restricted symbols (punctuation) can only be substituted with their equivalents
		if (ilabel != olabel && (ilabel <= TO_RESTRICT || olabel <= TO_RESTRICT)) return;
		// Creating the corresponding emission arc for each state simultaneously
		this->arcIndexer.push_back({ilabel, olabel});
		int arcIndex = this->arcIndexer.size() - 1;
		W weight = this->getArcWeight(arcIndex);
		if (weight != W::Zero()) {
			for (int state = 0; state < 2 * max_delay + 1; state++) {
				this->AddArc(state, A(ilabel, olabel, weight, state));
			}
		}
	}

	void addDeletionArc(int ilabel, float freezeProb = -1) {
		// Creating the corresponding deletion arc for each state simultaneously
		this->arcIndexer.push_back({ilabel, latin_epsilon});
		int arcIndex = this->arcIndexer.size() - 1;
		W weight = this->getArcWeight(arcIndex, freezeProb);
		if (weight != W::Zero()) {
			if (max_delay == 0) {
				this->AddArc(0, A(ilabel, latin_epsilon, weight, 0));
			} else {
				for (int state = 0; state < 2 * max_delay; state++) {
					this->AddArc(state, A(ilabel, latin_epsilon, weight, state+1));
				}
			}
		}
	}

	void addInsertionArc(int olabel, float freezeProb = -1) {
		// Creating the corresponding insertion arc for each state simultaneously
		this->arcIndexer.push_back({orig_epsilon, olabel});
		int arcIndex = this->arcIndexer.size() - 1;
		W weight = this->getArcWeight(arcIndex, freezeProb);
		if (weight != W::Zero()) {
			if (max_delay == 0) {
				this->AddArc(0, A(orig_epsilon, olabel, weight, 0));
			} else {
				for (int state = 0; state < 2 * max_delay; state++) {
					this->AddArc(state+1, A(orig_epsilon, olabel, weight, state));
				}
			}
		}
	}
};

// Implementation of weight converters for the special weight type
namespace fst {
	template<>
	struct WeightConvert<TropicalWeight, ExpVecWeight> {
		ExpVecWeight operator()(TropicalWeight w1) const {
			return ExpVecWeight(w1.Value(), VecWeight::Zero());
		}
	};

	template<>
	struct WeightConvert<ExpVecWeight, TropicalWeight> {
		TropicalWeight operator()(ExpVecWeight w1) const {
			return TropicalWeight(w1.Value1().Value());
		}
	};
}

#endif /* BASE_FST_H_ */
