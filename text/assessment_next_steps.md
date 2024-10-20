
## Improving chess neural networks - assessing the prospects
The LC0 chess project has succeeded in continuously improving the quality of chess networks, with a major improvement in accuracy seen every 6 to 12 months. Most recently, these improvements have been driven by the replacement of convolutions with encoder blocks from the transformer architecture. 

A hope is that CeresTrain will facilitate reseach tasks such as architecture search and training regime refinement. It now becomes easier to implement and test ideas proposed in the academic literature for improvements. 

On the one hand, our expectations for such experiments should probably be modest. The academic literature is littered with many papers which claim to deliver neural net architectural improvements but upon careful scrutiny are found to be marred by flaws such as:
- inability to replicate, no reference implementation provided

- inappropriate baseline for comparison (not state of the art)

- failure to disprove statistical noise as source of outperformance

- unmentioned limitations or drawbacks (such as high training instability or increased latency)

- failure to demonstrate Pareto dominance (e.g. on accuracy vs computational efficiency tradeoff)

- failure to characterize magnitude of potential overfitting of hyperparameters

For example, despite dozens of proposed variants of the transformer architecture, few have stood the test of time and achieved wide adoption (see ["Do Transformer Modifications Transfer Across Implementations and Applications?"](https://arxiv.org/pdf/2102.11972.pdf)). Instead, it seems that the simple scaling of the basic transformer mechanism remains dominant.

On the other hand, the specific nature of the chess problem does provide some cause for optimism. First, the [Smolgen feature](https://github.com/Ergodice/lczero-training/blob/attention-net-body/README.md) developed by Daniel Monroe has already proven that certain domain-specific architectural adaptations can yield major improvements. Second, chess has the property of high fungibility of accuracy and computational effort. Specifically, MCTS search can be used to turn a higher computational budget into higher accuracy, almost without limit. By contrast, other domains such as NLP posses this feature to a much lesser degree (although "chain of thought" and "best of N" answers are steps in that direction). As a consequence, the importance and potential benefits of finding architectural adjustments which increase inference speed are much greater in chess and therefore especially worthy of continued research.


## Next Steps for CeresTrain

Future experiments will extend training to endgames having more than 7 pieces. This will require generation of training data by search using the Ceres engine. Generating the necessary several hundred million training positions is computationally intensive. However generating training data from endgame positions is somewhat less expensive than opening or middlegame positions because search subtrees often terminate early in tablebase positions and the value target (win/draw/loss) can often be determined without deep search.

Other important areas for future work infrastructure work include:
- optimization of inference speed (ONNX backend)
- support for other devices such as CPU, Apple MPS, AMD GPUs
- documentation of the CeresTrain API for programmatic usage
- better support for testing non-transformer architectures (e.g. convolutional)
- more experimentation with 7-man tablebases or even possibly a subset of [8-man tablebases](https://www.youtube.com/watch?v=i06N9WohMqc)
- make available introspection statistics relating to weights and activation

Areas for future research effort include:
- better characterizing the width vs depth tradeoff
- ablation studies on various features (such as Smolgen, non-ReLU activations, etc.) to quantify their importance
- determination of the smallest possible network sizes that deliver strong performance
- training and inference tests with FP8 quantization
- testing variations of training curriculum
- testing impact of label noise
- testing of additional transformer variants (focusing on accuracy vs speed tradeoff)
- testing with mixed tablebase positions (for example, does joint training on KRPkrp and KRPkr endgames perform equally well or better than just KRPkrp)