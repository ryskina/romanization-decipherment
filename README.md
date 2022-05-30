This repository contains the code for the [paper](https://www.aclweb.org/anthology/2020.acl-main.737/):

```
Phonetic and Visual Priors for Decipherment of Informal Romanization
Maria Ryskina, Matthew R. Gormley, Taylor Berg-Kirkpatrick
ACL 2020
```

Please contact mryskina@cs.cmu.edu for any questions.

This implementation relies on the [OpenFst library](http://www.openfst.org/) and the [OpenGrm Ngram library](http://www.opengrm.org/twiki/bin/view/GRM/NGramLibrary). 

## Requirements

  * g++ >= 5.4.0
  * OpenFst == 1.7.X (originally implemented with 1.7.4)
  * OpenGrm == 1.3.X (originally implemented with 1.3.8)

You can install the dependencies automatically by building a Conda enviroment from [`environment.yml`](environment.yml), contributed by Michele Corrazza ([@ashmikuz](https://gitlab.com/ashmikuz)).

## Data

The data files for Russian and Arabic must be stored in the `./data/ru/` and `./data/ar/` directories respectively.

**Russian:** [`ru.tgz`](data/ru.tgz) contains the full preprocessed romanized Russian dataset, including the symbol tables and priors. The language model data file is a preprocessed version of the `vktexts.txt` file from the social media segment of the [Taiga Corpus](https://tatianashavrina.github.io/taiga_site/downloads); the rest of the data is collected by the authors.

**Arabic:** [`ar.tgz`](data/ar.tgz) contains only the files for the symbol tables and priors. The [BOLT Egyptian Arabic SMS/Chat and Transliteration](https://catalog.ldc.upenn.edu/LDC2017T07) dataset used in this paper is distributed by LDC; a script to preprocess the LDC data into the required format will be added shortly.

## Usage

Run `make` to build the code. If the OpenFst and OpenGrm libraries are installed in a location other than default (`/usr/local/`), you need to specify the correct include (`-I`) and lib (`-L`) paths in the makefile.

To reproduce the supervised experiments described in the paper, run:
```
./decipher --dataset {ar|ru} --supervised
```

To reproduce the unsupervised experiments:
```
./decipher --dataset {ar|ru} --freeze-at 100 [--prior {phonetic|visual|combined}]
```

To see the full usage statement and the command line option descriptions, run:
```
./decipher --help
```

## Reference
 ```
 @inproceedings{ryskina2020phonetic,
  title={Phonetic and Visual Priors for Decipherment of Informal Romanization},
  author={Ryskina, Maria and Gormley, Matthew R. and Berg-Kirkpatrick, Taylor},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
 ```
