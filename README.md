# deepMiRGene
Deep Recurrent Neural Network-Based Identification of Precursor microRNAs

Park, Seunghyun, et al. "Deep Recurrent Neural Network-Based Identification of Precursor microRNAs." Advances in Neural Information Processing Systems. 2017.

http://papers.nips.cc/paper/6882-deep-recurrent-neural-network-based-identification-of-precursor-micrornas

//

Usage:
  python inference/deepMiRGene.py -i \<input file\> -o \<output file\>

- input file: fasta format
- output file: 0 (true pre-miRNA) or 1 (pseudo pre-miRNA)



Dependencies:
1. biopython http://biopython.org/wiki/Download

2. RNAfold (python version) https://www.tbi.univie.ac.at/RNA/#download

3. sklearn http://scikit-learn.org/stable/install.html

4. Keras https://keras.io/#installation
- theono backended




** Reproduce
1. reproduce/cv.py
- cross-validation results for the human and cross-species dataset (Table 2)

2. reproduce/test.py (human and cross-species)
- test results for the human and cross-species dataset (Table 2)

3. reproduce/test_new.py (new)
- test results for the new dataset (Table 3)



