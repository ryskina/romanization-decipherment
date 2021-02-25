/*
 * main.cpp
 *
 *  Created on: Oct 24, 2019
 *      Author: Maria Ryskina
 */

#include <list>
#include <ctime>
#include <math.h>
#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>

#include <fst/fstlib.h>
#include <ngram/ngram.h>

#include "data_utils.h"
#include "fst_utils.h"
#include "base_fst.h"
#include "emission.h"
#include "lm.h"
#include "model.h"

using namespace fst;

// Default hyperparameter settings
int seed = 0;
int batch_size = 10;
int upgrade_lm_every = 100;
int upgrade_lm_by = 1;
int max_delay = 2;
float freeze_at = -1; // no freezing

std::string dataset;
std::string prior = "uniform";
bool run_supervised = false;
bool no_epsilons = false;
bool no_test = false;
bool no_save = false;


/*
 * Argument parsing functions
 */

void PrintHelp() {
    std::cout <<
    	"USAGE (unsupervised):\n"
    	"  ./decipher --dataset {ru|ar} [--seed S] [--batch-size B] [--upgrade-lm-every E] [--upgrade-lm-by U]"
    	" [--prior {phonetic|visual|combined}] [--freeze-at F] [--no-epsilons] [--no-test] [--no-save]\n"
        "USAGE (supervised):\n"
        "  ./decipher --dataset {ru|ar} --supervised [--seed S] [--batch-size B] [--no-test] [--no-save]\n\n"
    	"OPTIONS:\n"
		"--dataset {ru|ar}:                    Dataset (Russian or Arabic; mandatory parameter)\n"
		"--seed S:                             Set random seed to S (int; default 0)\n"
		"--batch-size B:                       Mini-batch size for stepwise EM training "
		                                       "(unsupervised only; int; default 10)\n"
		"--upgrade-lm-every E:                 Increasing language model order after processing every E batches "
		                                       "(unsupervised only; int; default 100)\n"
		"--upgrade-lm-by U:                    Increasing language model order by U each time "
		                                       "(unsupervised only; int; default 1)\n"
		"--prior {phonetic|visual|combined}:   Prior on emission parameters "
		                                       "(unsupervised training only; default = uniform)\n"
		"--freeze-at F:                        Train with freezing insertion and deletion probabilities at F "
		                                       "for the first E batches (unsupervised only; float; "
		                                       "no freezing by default)\n"
		"--supervised:                         Train a supervised model on validation data\n"
		"--no-epsilons:                        Turn off insertions and deletions (unsupervised only)\n"
		"--no-test:                            Turn off testing\n"
		"--no-save:                            Turn off model saving\n"
		"--help:                               Display help\n";
    exit(1);
}

void ProcessArgs(int argc, char* argv[]) {
    const char* const short_opts = "d:n:b:e:u:p:f:srtvh";
    const option long_opts[] = {
		{"dataset", required_argument, nullptr, 'd'},
		{"seed", required_argument, nullptr, 'n'},
		{"batch-size", required_argument, nullptr, 'b'},
		{"upgrade-lm-every", required_argument, nullptr, 'e'},
		{"upgrade-lm-by", required_argument, nullptr, 'u'},
		{"prior", required_argument, nullptr, 'p'},
		{"freeze-at", required_argument, nullptr, 'f'},
		{"supervised", no_argument, nullptr, 's'},
		{"no-epsilons", no_argument, nullptr, 'r'},
		{"no-test", no_argument, nullptr, 't'},
		{"no-save", no_argument, nullptr, 'v'},
		{"help", no_argument, nullptr, 'h'},
		{nullptr, no_argument, nullptr, 0}
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt)
        {
        case 'd':
            dataset = std::string(optarg);
            if (dataset != "ru" && dataset != "ar") {
            	std::cout << "Error: Unknown dataset option: " << dataset << std::endl;
            	PrintHelp();
            	break;
            }
            std::cout << "Dataset: " << dataset << std::endl;
            break;

        case 'n':
            seed = std::stoi(optarg);
            break;

        case 'b':
            batch_size = std::stoi(optarg);
            break;

        case 'e':
            upgrade_lm_every = std::stoi(optarg);
            break;

        case 'u':
            upgrade_lm_by = std::stoi(optarg);
            break;

        case 'p':
            prior = std::string(optarg);
            if (prior != "phonetic" && prior != "visual"  && prior != "combined") {
            	std::cout << "Error: Unknown prior option: " << prior << std::endl;
                PrintHelp();
                break;
            }
            break;

        case 'f':
            freeze_at = std::stof(optarg);
            break;

        case 's':
            run_supervised = true;
            break;

        case 'r':
            no_epsilons = true;
            break;

        case 't':
            no_test = true;
            break;

        case 'v':
            no_save = true;
            break;

        case 'h': // -h or --help
        case '?': // Unrecognized option
        default:
        	PrintHelp();
            break;
        }
    }

    if (dataset.empty()) {
    	std::cout << "Error: No dataset specified\n";
    	PrintHelp();
    } else {
    	if (dataset == "ar") max_delay = 5;
    	std::cout << "Maximum emission model delay: " << max_delay << std::endl;
    }

    if (run_supervised) {
    	std::cout << "Training a supervised model\n";
    } else {
    	std::cout << "Training an unsupervised model with " << prior << " prior\n";
    	std::cout << "Mini-batch size: " << batch_size << std::endl;
    	std::cout << "Language model order will be increased every " << upgrade_lm_every << " batches\n";
    	std::cout << "Language model order will be increased by " << upgrade_lm_by << " each time\n";
    	if (freeze_at > 0) {
        	std::cout << "Insertion and deletion probabilities frozen at " << exp(-freeze_at) <<
        			" (" <<  freeze_at << " in negative log space) for the first " <<
					upgrade_lm_every << " batches\n";
    	}
    }
    std::cout << "Seed: " << seed << std::endl;

}


int main(int argc, char* argv[]) {
	ProcessArgs(argc, argv);
	std::string data_dir = "./data/" + dataset;

	mkdir("./output/", 0777);
	std::string output_dir = "./output/" + dataset;
	if (run_supervised) {
		output_dir = output_dir + "_sup_seed-" + std::to_string(seed);
	} else {
		output_dir = output_dir + "_uns_" + prior + "_seed-" + std::to_string(seed) +
				"_upgrade-lm-every-" + std::to_string(upgrade_lm_every) +
				"-by-" + std::to_string(upgrade_lm_by);
		if (no_epsilons) {
			output_dir += "_no-epsilons";
		}
		if (freeze_at >= 0) {
			output_dir += "_freeze-" + std::to_string(freeze_at);
		}
	}
	mkdir(output_dir.c_str(), 0777);
	std::cout << "Saving models and output files to " << output_dir << std::endl;

	Indexer origIndexer = Indexer(data_dir + "/alphabet_orig.txt");
	Indexer latinIndexer = Indexer(data_dir + "/alphabet_latin.txt");

	IndexedStrings trainData(&latinIndexer, &origIndexer);
	DataUtils::readAndIndex(data_dir + "/data_train.txt", &trainData);
	std::cout << "\nLoaded " << trainData.latinIndices.size() << " training sentences\n";

	IndexedStrings devData(&latinIndexer, &origIndexer);
	DataUtils::readAndIndex(data_dir + "/data_dev.txt", &devData);
	std::cout << "Loaded " << devData.latinIndices.size() << " validation sentence pairs\n";

	IndexedStrings testData(&latinIndexer, &origIndexer);
	if (!no_test) {
		DataUtils::readAndIndex(data_dir + "/data_test.txt", &testData);
		std::cout << "Loaded " << testData.latinIndices.size() << " test sentence pairs\n";
	} else {
		std::cout << "Testing turned off\n";
	}

	IndexedStrings lmTrainData(&latinIndexer, &origIndexer);
	DataUtils::readAndIndex(data_dir + "/data_lm.txt", &lmTrainData);
	std::cout << "Loaded " << lmTrainData.latinIndices.size() << " monolingual LM training sentences\n";

	if (!latinIndexer.locked) latinIndexer.lock();
	if (!origIndexer.locked) origIndexer.lock();

	if (run_supervised) {
		std::cout << "\nTraining the language model of the original orthography...\n";
		VectorFst<StdArc> lmFst = trainLmOpenGRM(lmTrainData, origIndexer.getSize(), 6, output_dir, no_save);
		std::cout << "Done\n";

		std::cout << "\nTraining the emission model on validation data...\n";
		EmissionTropicalSemiring tropicalEm = trainEmission(devData, max_delay, origIndexer.getSize(),
				latinIndexer.getSize(), seed, output_dir, no_save);
		std::cout << "Done\n";

		Model model(lmFst, tropicalEm);

		if (!no_save) {
			std::string model_outfile = output_dir + "/model.fst";
			std::cout << "\nComposing and saving the base lattice to: " << model_outfile << std::endl;
			VectorFst<StdArc> allStatesLattice;
			lmPhiCompose<StdArc>(model.lmStd, model.emStd, &allStatesLattice);
			allStatesLattice.Write(model_outfile);
		}

		if (!no_test) {
			std::cout << "\nTesting...\n";
			std::string prediction_file = output_dir + "/prediction.txt";
			std::cout << "Writing predictions to " << prediction_file << std::endl;
			float cer = model.test(testData, PHI_MATCH, true, prediction_file);
			std::cout << "CER: " << cer << std::endl;
		} else {
			// Printing the emission parameters
			model.emStd.printProbsWithLabels(model.emStd.logProbs, &origIndexer,
					&latinIndexer, model.emStd.orig_epsilon, model.emStd.latin_epsilon);
		}

	} else {
 		std::vector<std::vector<int>> monolingualTrain = trainData.latinIndices;
		std::sort(monolingualTrain.begin(), monolingualTrain.end(),
				[](const std::vector<int> &a, const std::vector<int> &b){ return a.size() < b.size(); });

		std::cout << "\nTraining the language models of the original orthography...\n";
		std::vector<VectorFst<StdArc>> lmFstArray;
		// While insertion/deletion probabilities are kept frozen,
		// the language model disallows epsilons (insertions) to keep the model locally normalized
		for (int order = 2; order <= 6; order++) lmFstArray.push_back(
				trainLmOpenGRM(lmTrainData, origIndexer.getSize(), order, output_dir, no_save,
						no_epsilons || (freeze_at >= 0 && order == 2)));

		std::vector<std::pair<int, int>> priorMappings;
		if (prior != "uniform") {
			std::string priorFname = data_dir + "/prior_" + prior + ".txt";
			priorMappings = DataUtils::readPrior(priorFname, &latinIndexer, &origIndexer);
			std::cout << "Initializing with the " << prior << " prior\n";
		}

		Trainer trainer(lmFstArray, max_delay, &origIndexer, &latinIndexer, seed, priorMappings,
				no_epsilons, freeze_at);
		trainer.train(monolingualTrain, devData, testData, output_dir, batch_size, upgrade_lm_every,
				upgrade_lm_by, PHI_MATCH, true, no_save);

	}

	return 0;
}
