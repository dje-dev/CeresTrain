
## Methods of Utilizing CeresTrain

CeresTrain can be utilized in one of two different ways. One way involve writing C# code and programming to the API exposed by the software library. This approach offers maximum flexibility. The second approach involves using the CeresTrain executable with command line arguments. This approach is very convenient for rapid experimentation. 

In the next section we'll utilize the command line mode and use a subset of the available commands, which can be displayed with the --help argument:

![Alt text](console_commands.png)


## Running CeresTrain from the Command Line

In this example we will train a neural network to play one of the simpler endgames - King and pawn versus King and pawn. There are 7,436,088
 such unique positions, as can be seen on the [Syzygy Endgame tablebases](https://syzygy-tables.info/?fen=8/p7/8/8/2K5/5k2/2P5/8_w_-_-_0_1) site, showing on this page a composition by Grigoriev (1929) where white can win with exact play.

Each CeresTrain experiment is tagged with an identifying configuration name and an associated set of files which capture the architecture/configuration of the training and also the artifacts resulting from the training session (such as Tensorboard files, summary results files, and the actual binary neural network weights files).

The first step to run an experiment is to create a configuration with a specified name using the init command. For example:

```
CeresTrain init --config=smallnet
```

This command will result in the creation of a set of JSON files which describe several dozen configuration parameters relating to data sources, network architecture, optimization hyperparameters, machine configuration (e.g. which GPUs to use), monitoring options, etc. The full contents of each of these files can be displayed on the console using the info command. For example, the section relating to network architecture looks like the following:
```
CeresTrain info --config=smallnet

e:\cout\configs\smallnet_ceres_net.json
  {
    "ModelDim": 192,
    "NumLayers": 6,
    "NumHeads": 8,
    "PreNorm": false,
    "NormType": "LayerNorm",
    "AttentionMultiplier": 1,
    "FFNMultiplier": 1,
    "FFNActivationType": "ReLUSquared",
    "HeadsActivationType": "ReLU",
    "DeepNorm": false,
    "SmolgenDimPerSquare": 8,
    "SmolgenDim": 64,
    "HeadWidthMultiplier": 4
  }
```

Next we actually train a neural network using 10 million randomly generated training positions with this configuration:
```
CeresTrain train --config=smallnet --pieces=KPkp --num-pos=10000000
```

During the course of the circa 11 minute training session, status information will be continuously logged to the console, showing various training statistics. Note that the value head accuracy attains 98.63% by the end of the session. The policy accuracy is much lower (45.19%) but this is not very meaningful because there are typically many equally good moves and this metric measures tests for exact equality.

![Alt text](live_train_status.png)



Now we can reload the saved network and evaluate the performance of that network on true out of sample random positions, and also compare against the value/policy head accuracies against a representative network trained by the LC0 community (here a late T81 network consisting of 19 layers and about 75 million parameters). Note that the metric used here to assess policy is testing for membership of the top policy move in the set of moves which all have the same best possible outcome. We note that the Ceres net has an error rate on the value target about 1/3 lower than T81 and very similar policy performance.
```
CeresTrain eval --config=smallnet --pieces=KPkp --num-pos=5000 --verbose=true --net-spec=811971

Number of test positions : 5000  KPkp
  Accuracy value        : 98.86%
  Accuracy value (comp) : 97.94% using <weights_run1_811971.pb.gz on GPU 0>

  Accuracy policy       : 99.82%
  Accuracy policy (comp): 99.80% using <weights_run1_811971.pb.gz on GPU 0>
```
Because the verbose option was enabled on the command line, each test position will be logged to the console with the evaluations of the Ceres and LC0 networks shown (with errors flagged in red). For example:
![Alt text](test_positions.png)

It is certainly encouraging that a small neural net can be trained in just a few minutes and outperforms much larger generalized nets. However we must remember that the training and evaluation sampled uniformly from random positions. However the types of positions actually encountered in games are likely to have a significantly different distribution. We confirm this is the case by rerunning the evaluation but specifying a PGN file containing actual KPkp positions that were seen in human games (using the pos-fn argument). In this case, we see the LC0 network now dramatically outperforms, scoring 99.02% on value compared to only 96.70% for the Ceres net.
```
CeresTrain eval --config=smallnet --pieces=KPkp --num-pos=5000 --verbose=true --net-spec=~T81 --pos-fn=e:\cout\data\KPkp.pgn

Number of test positions : 5000  KPkp
  Accuracy value        : 96.70%
  Accuracy value (comp) : 99.02% using <weights_run1_811971.pb.gz on GPU 0>

  Accuracy policy       : 99.26%
  Accuracy policy (comp): 99.94% using <weights_run1_811971.pb.gz on GPU 0>
```


Finally, we can use the UCI command to revisit the composed position mentioned at the beginning of the section and test if the trained network understands the position is winning. We see that it assigns a value of +111 centipawns which does qualify as more likely winning than not. It also proposes Kd4 as the top policy move, which is indeed the winning move (not Kb5).
```
CeresTrain uci --config=smallnet --pieces=KPkp

position fen 8/p7/8/8/2K5/5k2/2P5/8 w - - 0 1
go nodes 1

info depth 1 seldepth 1 time 16 nodes 1 score cp 111 tbhits 0 nps 61 pv c4d4 
```

## Scaling CeresTrain using PyTorch 2.0 with distributed training

We showed in the prior section that a small net (of 5 million parameters) trained for 10mm positions performs reasonably well, but still significantly below that of LC0 nets when tested on typical endgame positions. The next question is if we can train a larger net for longer and thereby achieve much higher accuracy. For this we must turn to...

![Alt text](remote_live_train_status.png)
```
CeresTrain eval --config=KP_256_10_bl --pieces=KPkp --num-pos=5000 --verbose=true --net-spec=~T81 --pos-fn=e:\cout\data\KPkp.pgn

Number of test positions : 5000  KPkp
  Accuracy value        : 99.94%
  Accuracy value (comp) : 99.02% using <weights_run1_811971.pb.gz on GPU 0>

  Accuracy policy       : 99.94%
  Accuracy policy (comp): 99.94% using <weights_run1_811971.pb.gz on GPU 0>
```

## Scaling to harder problems comparing to SOTA networks

Consider now the harder 6-man positions with KQPkqp and a comparison CeresTrain net performance against one of the best available nets from the LC0 project (T2). We trained nets of the same size (256x10x8 FFN1 Smolgen) for 200 million positions against each of 3 representative 6-man tablebase positions and computed the value head accuracy of the nets (both CeresTrain and LC0) on both random positions and actual positions:

*Value head accuracy of CeresTrain versus LC0 T2 network*
|         |  KPPkpp |  KRPkrp |  KQPkqp |
|---------|--------:|--------:|--------:|
| Random  |   1.22% |    1.5% |   3.48% |
| Actual  |  -0.45% |  -0.08% |   0.47% |


We make several observations:
* CeresTrain is clearly superior on random positions
* Results are mixed on actual positions
* CeresTrain relative performance increases with more difficult endgames
* CeresTrain accuracy continues to improve with longer training (for example, the KRP endgame trained with 500mm positions rather than 200mm achieved relative accuracy of 0.26% versus T2 rather than -0.08%)

It is not surprising that CeresTrain nets are less effective on actual game positions, since they represented a much smaller percentage of the training distribution that what was fed to the LC0 networks (100% from actual games). One possible remedy would be to filter the CeresTrain training positions to only those which are "more typical" of actual game positions. Another or complementary strategy would be to train bigger networks, or train them for longer. 


