#A simple tutorial for FLGC

### Requirement ###

- Pytorch >= 1.7.0
- Torch-Geometric >= 1.6.3
- numpy
- scikit-learn


### Run ###

    # reproduce semi-supervised node classification on Cora
    python demo_ssl_graph.py

    # reproduce un-supervised clustering on Iris
    python demo_clustering.py 



### Citation ###
    @article{FLGC-ACM-2023,
	    author = {Cai, Yaoming and Zhang, Zijia and Ghamisi, Pedram and Cai, Zhihua and Liu, Xiaobo and Ding, Yao},
	    title = {Fully Linear Graph Convolutional Networks for Semi-Supervised and Unsupervised Classification},
	    year = {2023},
	    issue_date = {June 2023},
	    publisher = {Association for Computing Machinery},
	    address = {New York, NY, USA},
	    volume = {14},
	    number = {3},
	    doi = {10.1145/3579828},
	    journal = {ACM Transactions on Intelligent Systems and Technology},
	    month = {apr},
	    articleno = {40},
	    numpages = {23}
    }
