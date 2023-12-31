## Training on Endgame Tablebases - Opportunities and Challenges

Opportunities of training on EGTB positions include:
- numerous sets of progressively more challenging problems is available for testing (3..7 man, different piece combinations)

- training example generation requires almost no time because random legal chess positions with few men are trivial to generate and the training target (e.g. win/loss status) can be queried from the tablebases in also negligible time. Indeed, training data can be generated with sufficient speed (in a compiled language such as C#) to make on-the-fly training data generation feasible, obviating a preprocessing step to prepare training data.

- the exact ground truth training targets are directly available for the value head via the win/draw/loss information available from tablebases for all positions

- convergence will be achieved with fewer training samples due to the simplified domain (fewer pieces on board and closer proximity to terminal positions)

- absolute performance metrics exist, such as win/draw/loss accuracy rates or Elo rating versus perfect play and can be calculated quickly by playing against an opponent with tablebases enabled (an oracle)

- in theory, training might converge to perfect accuracy for some endgames. This might be of practical use if the neural network were also smaller (requiring less disk space)

Challenges include:
- problem complexity is limited to 7-man positions

- tablebases do not directly provide policy trading targets and these must therefore they be proxied using some heuristics (such as assigning higher probability to wins than losses, higher probability to promotion to queen than to pawn, higher probability to DTZ shortening moves, etc.). Alternatively, the nets could be trained without any policy target.

- tablebase positions sampled purely at random are likely to be mostly "out of distribution" with respect to what is seen in typical chess positions (for example, opposing pawns on the same file which are not blocking one another is a less common condition in actual play)

- related, tablebase sampled at random contain a much higher frequency of "obvious" wins/draws than seen in actual play, thereby distorting error rate metrics

- neural networks trained on tablebases will never perform well in actual game play because neither the value nor the proxy policy targets provide sufficient information to reliably provide the specific sequence of moves to achieve the best possible outcome in the position

- a neural network trained on tablebase positions is of no direct practical use in playing chess because EGTB already provide perfect answers at very low computational cost

-  lessons learned in the simplified domain potentially might fail to transfer to larger networks and the full game of chess. On the other hand, CeresTrain research to date has suggested this is mostly not the case. More generally, it has been observed that already in some extremely simplified domains (such as 1-D MINST) many phenomena can be observed which are still present in extremely large systems.

These disadvantages notwithstanding, the benefits of starting CeresTrain with tablebase training seem compelling:
- correctness of the data generation, training, and network evaluation code is more easily established. For example, it is expected that perfect accuracy should ultimately result from the training of simple endgames such as KPk.

- the time-consuming process of testing of the software and neural network variations is greatly sped up because of the aforementioned increased training speed. 

- it becomes more practical to continue training sessions long enough to reach near-saturation. This may be important because the true value (or not) of variations of network architecture or optimization technique may only becoming apparent after a very long period of training.

- it seems plausible that endgame training can progressively extended to more complex positions which are not covered by EGTB (for example, positions with up to 10 pieces). If it remains the case that specialized networks can achieve higher value accuracy with much lower computational cost, it seems they may be very useful in actual game play as leaf nodes reaching these positions would be more quickly and accurately assessed.


## Baseline performance of existing networks on EGTB positions

An interesting question is how well current general purpose chess neural networks perform on endgame positions. To assess this, we calculated the accuracy of the value head on a random positions draw from a set of common or interesting endgame types. Shown below is the error rate (complement of accuracy) in classifying win/draw/loss for a series of LC0 networks. 

#### Error Rates of LC0 Network Value Head on Random Tablebase Positions

| NetID              | KPk   | KPkp  | KPPkpp | KNPknp | KRPkrp | KQPkqp | KNNkpp |  Avg  |
|--------------------|------:|------:|-------:|-------:|-------:|-------:|-------:|------:|
| T1                 |  0.03 |  0.86 |   2.62 |   2.79 |   3.07 |   3.87 |   6.32 | 2.79  |
| BT2                |  0.11 |  1.25 |   3.48 |   3.34 |   4.09 |   4.70 |   8.41 | 3.63  |
| T1_DISTILL_512     |  0.20 |  1.07 |   3.26 |   3.52 |   4.16 |   5.12 |   7.65 |  3.57 |
| T2                 |  0.02 |  0.79 |   3.12 |   3.08 |   4.14 |   4.94 |   7.34 | 3.35  |
| T78                |  0.14 |  1.41 |   4.22 |   4.42 |   5.36 |   6.22 |   8.97 | 4.39  |
| T81                |  0.13 |  2.19 |   4.86 |   5.52 |   6.99 |   8.27 |   9.06 | 5.29  |
| T80                |  0.30 |  2.48 |   5.35 |   5.66 |   6.74 |   9.58 |   9.18 | 5.61  |
| T60                |  0.38 |  3.51 |   7.23 |   7.54 |   9.57 |  13.56 |  13.33 | 7.87  |
| T1_DISTILL_256     |  1.74 |  5.24 |  10.04 |  10.27 |  10.77 |  11.44 |  18.94 |  9.78 |
| T75                |  1.11 |  5.33 |   9.22 |  11.13 |  13.52 |  19.30 |  15.93 | 10.79 |
| T70                |  4.11 | 11.90 |  17.57 |  16.15 |  19.17 |  28.23 |  20.84 | 16.85 |

We see steady improvement of accuracy with later networks, but the error rate remains fairly high for the more complex endgames. On the one hand, the direct significance of this finding is probably very limited because:

* the LC0 networks might perform better if a history of prior moves were available

* the uniform distribution of samples across all legal endgames tested here differs significantly from the distribution of positions likely to be seen in actual game play 

* neural network evaluations are not needed at all because EGTB are available and commonly used

On the other hand, this does suggest that value accuracy is already quite imperfect at 6 pieces and is likely to be even greater at higher piece counts. It seems worthwhile to test if specialized networks trained specifically on endgames of (say) 10 or less pieces can outperform general networks in both accuracy and speed, thereby adding practical value. This idea of network specialization is loosely related to ideal of "mixture of experts" which has recently been applied with good effect in large language models such as GPT-4 and Mixtral 8x7b. Furthermore, it seems like that in endgames there is often no substitute for search with the consequence that improving inference speed (with smaller specialized nets) may prove critical.

