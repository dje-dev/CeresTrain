
## Philosopical underpinnings
A few philosophical ideas underlie the approach taken with the CeresTrain project:

* **reductionism** - research on hard problems best begins in a simplified and more manageable setting. Using chess as the ["Drosophila of Artificial Intelligence"](https://doi.org/10.1007/978-1-4613-9080-0_14) is already one application of this idea, and the idea of using only endgames within chess is a further recursive application.

* **empiricism** - the history of deep learning shows that empirical findings almost always leads theory, so rapid experimentation is justifiable to build an intuition for the behavior of these systems and their possibilities.

* **investment in software infrastructure** - spending time up front on software infrastructure to facilitate research can often pay big dividends. For example, deep learning libraries such as Pytorch represented a massive investment and delivers no direct research insights, but has ultimately proven indispensable for progress in deep learning  

* **scaling dominance** - the history of deep learning shows that simple methods, scaled across data and compute, are often ultimately the most successful, as pointed out in ["The Bitter Lesson"](http://www.incompleteideas.net/IncIdeas/BitterLesson.html). Thus research infrastrurcture should be designed from the ground up with scaling as a focus.

* **focus on execution speed** - a consequence of both the empiricism and scaling observations is that that research productivity will be gated by the speed at which experiments can be set up and performed. Therefore the system has been designed and implemented to deliver high performance, thereby enabling the researcher to do more experiments per unit of time. Concretely, this translates into use of a modern strongly typed compiled language (C#), frequent use of multithreading, intense runtime performance optimization, and heavy reliance upon state-of-the-art third-party libraries such as PyTorch. 
* **power of open source** - building on the community collaboration across data, code, and research findings has proven remarkably successful, as for example demonstrated by the [astonishing rate of progress](https://www.semianalysis.com/p/google-we-have-no-moat-and-neither) in generative AI

