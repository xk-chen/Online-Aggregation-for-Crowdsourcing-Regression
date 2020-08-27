# Online-Aggregation-for-Crowdsourcing-Regression
Experimental codes for the algorithm Online Aggregation for Crowdsourcing Regression

Data stream mining widely exists in practical applications such as online advertising, quantitative investment and crowdsourcing data analysis. For these tasks, the existing methods such as deep neural networks and support vector machines, which are based on batch learning scheme in general, have non-satisfactory performances in terms of efficiency and timeliness. As a new learning paradigm, online learning has become an ideal vehicle for large-scale data stream mining due to low computational complexity. 

To address the ubiquitous issues of labor burden as well as algorithm selection in data stream mining tasks, in this paper, we propose Ensemble Online Learning algorithm with the following characteristics:
1. This algorithm only needs to fetch the labels of partial observations that are determined by influence function. 
2. Utilizing weighted majority method, this algorithm aggregates base learning algorithms effectively through the trust degree that is adjusted dynamically. 
