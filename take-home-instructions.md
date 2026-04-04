Here's our current takehome question. You'll notice that the linked pre-print uses CV/some LLMs to try to digitize KM curves. I think there's a better way to solve this however. Solutions such as https://github.com/TheJaeLal/LineFormer perform decent despite not being trained specifically for KM curves.

That being said, ultimately we only care about performance (accuracy) when benchmarking your solution.

## takehome task
Kaplan Meier (KM) plots are a common type of monotonically-decreasing plot in biomedical/clinical papers to show the likelihood of a population remaining event-free over time.

When conducting systematic reviews, KM plots in included papers can be a valuable source of data for conducting an analysis (ie. a KM plot might tell you “how many patients were still alive 6 months after receiving treatment”, when survival statistics are not otherwise reported explicitly in the paper). The challenge is that these plots must be digitised to extract these values. This digitisation is typically done by humans using digitisation software, and is a task that LLMs alone and other document parsing software (ie. reducto) completely fail at.

The wikipedia page gives a good overview / formal definition of the KM estimator: https://en.wikipedia.org/wiki/Kaplan–Meier_estimator.

Here’s an interesting pre-print that describes one approach to programmatically digitise KM curves using a combination of image processing and LLMs: https://www.biorxiv.org/content/10.1101/2025.09.15.676421v1.full.pdf+html.

The take-home problem is simple: use any approach you can come up with to digitise KM curves and some way to measure the performance of your algorithm. Include some type of benchmark on real or synthetic data to measure/demonstrate the accuracy of your solution. Feel free to use LLMs both as part of your solution and to help develop it.

This is a deceptively simple problem; in practice there are many edge-cases which make a robust solution difficult to achieve. Internally, our implementation of KM curve digitization currently achieves performance similar to the results in the above preprint.

We’re looking for 
Evidence of strong research design, including adequately powered training and evaluation benchmarks. This also includes good benchmark selection (i.e., including low resolution images, multiple overlapping curves, truncated y-axis, etc.)
High empirical performance (can you achieve ~95% accuracy on the curve fidelity metrics reported in this pre-print, or find another way to measure accuracy?)

