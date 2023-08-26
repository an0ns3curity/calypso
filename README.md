# calypso

To run the attack, navigate to `calypso/` and run one of **evolutionary_search_multithreaded_pypuf_aeomebic_xor_lppuf.py** (for cross-architectural attacks on LP-PUF) or **evolutionary_search_multithreaded_pypuf_aeomebic_xor.py** (for attacks on XOR PUFs). Tweek the following hyperparameters as suitable:

1. **--challenge-length**: The total size of the challenge set
2. **--cut-length**: The number of stages to be mutated in one generation
3. **--target-degree**: The degree of the underlying XOR PUF to be targeted

In order to mount other cross-architectural attacks, replace the LP-PUF instantiation **targetPUF = lppuf.LPPUFv1(n=CHALLENGE_LENGTH, m=PUF_LENGTH, seed=random.randint(0,100))** with a suitable instantiation.