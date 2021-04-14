/*
 * fst_utils.h
 *
 *  Created on: Oct 24, 2019
 *      Author: Maria Ryskina
 */

#ifndef FST_UTILS_H_
#define FST_UTILS_H_

#include <vector>

#include "fst/fstlib.h"

using namespace fst;

const int NUM_EPS_TOTAL = 10; // total number of epsilons to use if delay is not restricted

template <class A, int n>
class EpsilonTotalFilter : public VectorFst<A> {
using W = typename A::Weight;
public:
	int alphSize;

	EpsilonTotalFilter(int a, int epsilon_label) {
		alphSize = a;

		// Adding main state
		this->AddState();
		this->SetStart(0);
		this->SetFinal(0, W::One());

		// Adding *->* arcs
		addStarArcs(0, 0);

		// Adding epsilon states
		for (int i=1; i < n+1; i++) {
			this->AddState();
			this->SetFinal(i, W::One());
			// Adding i-th epsilon-arc
			this->AddArc(i-1, A(epsilon_label, epsilon_label, W::One(), i));
			// Adding loop *->* arcs
			addStarArcs(i, i);
		}
	}

private:
	void addStarArcs(int stateFrom, int stateTo) {
		for (int i = 1; i < alphSize + 1; i++) {
			this->AddArc(stateFrom, A(i, i, W::One(), stateTo));
		}
	}
};

// Constructing an acceptor of an indexed string
template<class A>
VectorFst<A> constructAcceptor(std::vector<int> indices, typename A::Weight w = A::Weight::One()) {
	using W = typename A::Weight;
	VectorFst<A> acceptor;
	acceptor.AddState();
	acceptor.SetStart(0);

	for (int& i : indices) {
		acceptor.AddState();
		int s = acceptor.NumStates() - 1;
		acceptor.AddArc(s-1, A(i, i, w, s));
	}
	acceptor.SetFinal(acceptor.NumStates() - 1, W::One());
	return acceptor;
};

// Construction an input FST (acceptor w/ insertion loops) for an indexed string
template<class A>
VectorFst<A> constructInput(std::vector<int> indices, int loop_olabel) {
	using W = typename A::Weight;
	VectorFst<A> acceptor;
	acceptor.AddState();
	acceptor.SetStart(0);
	acceptor.AddArc(0, A(0, loop_olabel, W::One(), 0));

	for (int& i : indices) {
		acceptor.AddState();
		int s = acceptor.NumStates() - 1;
		acceptor.AddArc(s-1, A(i, i, W::One(), s));
		acceptor.AddArc(s, A(0, loop_olabel, W::One(), s));
	}
	acceptor.SetFinal(acceptor.NumStates() - 1, W::One());
	return acceptor;
};

// Construction an output FST (acceptor w/ deletion loops) for an indexed string
template<class A>
VectorFst<A> constructOutput(std::vector<int> indices, int loop_ilabel) {
	using W = typename A::Weight;
	VectorFst<A> acceptor;
	acceptor.AddState();
	acceptor.SetStart(0);

	for (int& i : indices) {
		acceptor.AddState();
		int s = acceptor.NumStates() - 1;
		acceptor.AddArc(s-1, A(i, i, W::One(), s));
		acceptor.AddArc(s-1, A(loop_ilabel, 0, W::One(), s-1));
	}

	acceptor.AddArc(acceptor.NumStates() - 1, A(loop_ilabel, 0, W::One(), acceptor.NumStates() - 1));
	acceptor.SetFinal(acceptor.NumStates() - 1, W::One());
	return acceptor;
};

// Functions to verify composition results
template<class A>
bool composeCheck(VectorFst<A> *cfst, std::string name = "", bool verbose = true) {
	if (cfst->NumStates() == 0) {
		if (verbose) std::cout << "FAILED TO COMPOSE: "  << name << std::endl;
		return false;
	}
	return true;
}

template<class A>
bool pathCheck(VectorFst<A> *cfst, bool verbose = false) {
	if (cfst->NumStates() == 0) {
		if (verbose) std::cout << "FAILED TO FIND PATH" << std::endl;
		return false;
	}
	return true;
}

template<class A>
void printStats(VectorFst<A> *cfst, std::string name = "") {
	int numArcs = 0;
	for (int state = 0; state < cfst->NumStates(); state++) numArcs += cfst->NumArcs(state);
	std::cout << name << cfst->NumStates() << " states, " << numArcs << " arcs" << std::endl;
}

#endif /* FST_UTILS_H_ */
