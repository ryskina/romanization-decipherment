/*
 * data_utils.h
 * Methods for reading files and indexing data
 * Classes for storing symbol tables
 *
 *  Created on: Oct 24, 2019
 *      Author: Maria Ryskina
 */

#ifndef DATA_UTILS_H_
#define DATA_UTILS_H_

#include <list>
#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <locale>
#include <codecvt>
#include <algorithm>
#include <assert.h>
#include <dirent.h>
#include <stdlib.h>

#include <fst/fstlib.h>

#include "fst_utils.h"

using namespace fst;

// Symbol indexer and lookup map to store the alphabets
class Indexer {
public:
	std::map<char16_t, int> indexer;
	std::map<int, char16_t> lookup;
	bool locked = false;

	/*
	 * This implementation makes the following assumptions about indexing:
	 *    * symbols indexed 1 through 29 are punctuation marks that can only be
	 *          inserted, deleted or substituted with their equivalent
	 *    * symbols indexed 30 through alphabetSize are the remaining characters
	 *    * symbol indexed alphabetSize + 1 corresponds to an epsilon-transition
	 *          (insertion for Latin, deletion for original)
	 *
	 * Index 0 is reserved in OpenFST for epsilon and is not used in this implementation.
	 * We replace it with separate insertion and deletion symbols described above to avoid composition issues
	 * (compare to \epsilon_1 and \epsilon_2 in Mohri, "Weighted automata algorithms", section 5.1)
	 *
	 */

	// Initializing symbol indexer from file
	Indexer(std::string symbol_table_path) {
		std::ifstream symbol_table_file;
		symbol_table_file.open(symbol_table_path);

		std::string line;
		if (symbol_table_file.is_open()) {
			while (getline(symbol_table_file, line)) {
				std::u16string wline = std::wstring_convert<
						std::codecvt_utf8_utf16<char16_t>, char16_t>{}.from_bytes(line);;
				wchar_t c = wline.at(wline.size() - 1);
				int v;
				sscanf(line.c_str(), "%d %*s", &v);
				indexer.insert({c, v});
			}
			symbol_table_file.close();
			lock();
		} else {
			std::cout << "Unable to open symbol table file: " << symbol_table_path << std::endl;
			exit(-1);
		}
	}

	// Initializing symbol indexer with punctuation
	Indexer(bool arabic = false) {
		if (!arabic) {
			index(" :_()[|]-\"!.*=@/+&$~–—^");
			index(";?,<>%");
		} else {
			// For Arabic, some punctuation marks are substituted with special equivalents
			index(" :_()[|]-\"!.*=@/+&$~–—^");
			index("؛");
			index("؟");
			index("٬");
			index("><");
			index("٪");
		}
	}

	std::vector<int> index(std::string str) {
		std::u16string wstr = std::wstring_convert<
				std::codecvt_utf8_utf16<char16_t>, char16_t>{}.from_bytes(str);

		std::vector<int> res(wstr.length());
		std::map<char16_t,int>::iterator it;

		for (int pos = 0; pos < wstr.length(); pos++) {
			wchar_t c = wstr.at(pos);
			it = indexer.find(c);
			if (it == indexer.end()) {
				if (!locked) {
					indexer.insert({c, indexer.size() + 1});
				} else continue;
			}
			int val = indexer.at(c);
			res[pos] = val;
		}
		return res;
	}

	void getLookup() {
		std::map<char16_t, int>::iterator it = indexer.begin();
		while (it != indexer.end()) {
			char16_t c = it->first;
			int v = it->second;
			lookup.insert({v, c});
			it++;
		}
	}

	std::string encode(std::vector<int> indices) {
		std::u16string wstr;
		for (int &i: indices) {
			wstr.push_back(lookup.at(i));
		}
		return std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t>{}.to_bytes(wstr);
	}

	int getSize() {
		return indexer.size();
	}

	void lock() {
		locked = true;
		getLookup();
	}

	void exportSymbols(std::string outPath) {
		std::ofstream outfile;
		outfile.open(outPath);
		std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> cv;
		for (int pos = 1; pos < lookup.size() + 1; pos++) {
			outfile << pos << " " << cv.to_bytes(lookup.at(pos)) << std::endl;
		}
		outfile.close();
	}
};

// Data structure storing parallel data in indexed form
class IndexedStrings {
public:
	Indexer *latinIndexerPtr;
	Indexer *origIndexerPtr;
	std::vector<std::vector<int>> latinIndices;
	std::vector<std::vector<int>> origIndices;

	IndexedStrings(Indexer *lIPtr, Indexer *oIPtr) {
		latinIndexerPtr = lIPtr;
		origIndexerPtr = oIPtr;
	}

	void readLatinString(std::string latinString) {
		std::vector<int> indices = latinIndexerPtr->index(latinString);
		latinIndices.push_back(indices);
	}

	void readOrigString(std::string origString) {
		std::vector<int> indices = origIndexerPtr->index(origString);
		origIndices.push_back(indices);
	}

	void append(const IndexedStrings& a) {
		latinIndices.insert(latinIndices.end(), a.latinIndices.begin(), a.latinIndices.end());
		origIndices.insert(origIndices.end(), a.origIndices.begin(), a.origIndices.end());
	}

	int size() {
		return latinIndices.size();
	}

};

// An FST used to compute edit distance between strings in the original alphabet
class EditDistanceFst : public VectorFst<StdArc> {
public:
	EditDistanceFst(int origAlphSize) {
		this->AddState();
		this->SetStart(0);
		this->SetFinal(0, TropicalWeight::One());

		for (int i = 1; i < origAlphSize + 1; i++) {
			this->AddArc(0, StdArc(0, i, TropicalWeight(1), 0));
		}

		for (int i = 1; i < origAlphSize + 1; i++) {
			for (int j = 0; j < i; j++) {
				this->AddArc(0, StdArc(i, j, TropicalWeight(1), 0));
			}
			this->AddArc(0, StdArc(i, i, TropicalWeight(0), 0));
			for (int j = i+1; j < origAlphSize + 1; j++) {
				this->AddArc(0, StdArc(i, j, TropicalWeight(1), 0));
			}
		}
	}
};

class DataUtils {
public:
	static void printIndices(std::vector<int> indices) {
		for (int j = 0; j < indices.size(); j++) {
			std::cout << indices[j] + 1 << ' ';
		}
		std::cout << std::endl;
	}

	static void readAndIndex(std::string path, IndexedStrings *out, int max_len = 10000) {
		std::string latinString;
		std::string origString;

		std::ifstream myfile(path);

		if (myfile.is_open()) {
			while (getline(myfile,latinString)) {
				getline(myfile,origString);
				latinString.erase(std::remove(latinString.begin(), latinString.end(), '#'), latinString.end());
				origString.erase(std::remove(origString.begin(), origString.end(), '#'), origString.end());
				if (latinString.size() <= max_len) {
					out->readLatinString(latinString);
					out->readOrigString(origString);
				} else continue;
			}
			myfile.close();
		} else  {
			std::cout << "Unable to open file: " << path << std::endl;
			exit(-1);
		}
	}

	static float eval(std::vector<int> predicted, std::vector<int> gold, EditDistanceFst *evalFst) {
		VectorFst<StdArc> predFst = constructAcceptor<StdArc>(predicted);
		VectorFst<StdArc> goldFst = constructAcceptor<StdArc>(gold);
		VectorFst<StdArc> res;
		Compose(predFst, (VectorFst<StdArc>)*evalFst, &res);
		Compose(res, goldFst, &res);
		std::vector<TropicalWeight> dist;
		ShortestDistance(res, &dist, true);
		return dist[0].Value();
	}

	static std::vector<std::pair<int, int>> readPrior(std::string path, Indexer *lIPtr, Indexer *oIPtr) {
		std::string line;
		std::ifstream myfile(path);
		std::vector<std::pair<int, int>> res;

		if (myfile.is_open()) {
			while (getline(myfile,line)) {
				line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
				std::vector<int> lIndices = lIPtr->index(line.substr(0, 1));
				assert(lIndices.size() == 1);
				std::vector<int> oIndices = oIPtr->index(line.substr(1));
				for (int &oIndex : oIndices) {
					res.push_back({oIndex, lIndices[0]});
				}
			}
			myfile.close();
		} else {
			std::cout << "Unable to open file: " << path << std::endl;
			exit(-1);
		}
		return res;
	}
};

#endif /* DATA_UTILS_H_ */
