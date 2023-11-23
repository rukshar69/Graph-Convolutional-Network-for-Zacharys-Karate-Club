# Graph-Convolutional-Network-for-Zacharys-Karate-Club
We train a **semi-supervised GCN model** on **Zacharys karate club** dataset. We use the semi-supervised GCN model to predict the community of each node in Zachary's karate club network to predict club members' community.

# Introduction: Hands-on Graph Neural Networks


**Graph Neural Networks (GNNs)** aim to generalize classical deep learning concepts to irregularly structured data (in contrast to images or texts).

This is done by following a simple **neural message passing scheme**, where node features $\mathbf{x}_v^{(\ell)}$ of all nodes $v \in \mathcal{V}$ in a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ are iteratively updated by aggregating localized information from their neighbors $\mathcal{N}(v)$:


$x_v^{(\ell + 1)} = f^{(\ell + 1)}_{\theta}\left( x_v^{(\ell)}, \left( x_w^{(\ell)} : w \in \mathcal{N}(v) \right) \right)$



We implement Graph Neural Networks based on the **[PyTorch Geometric (PyG) library](https://github.com/rusty1s/pytorch_geometric)**, an extension library to the popular deep learning framework [PyTorch](https://pytorch.org/).

Let's dive into the world of GNNs by looking at a simple graph-structured example, the well-known [**Zachary's karate club network**](https://en.wikipedia.org/wiki/Zachary%27s_karate_club). This graph describes a social network of 34 members of a karate club divided into 4 communities and documents links between members. Here, we are interested in detecting communities that arise from the members' interaction.

To solve the Karate Club community detection problem using GNN, we specifically use **Graph Convolutional Network(GCN)**

Graph Convolutional Networks (GCNs) are a type of neural network specifically designed to handle data structured as graphs. They're particularly useful for tasks involving relational data, such as social networks, citation networks, biological networks, etc. The mechanism behind GCNs involves applying convolutional operations on graph-structured data.

Here's a breakdown of the mechanism:

1. **Graph Representation:**
   - Graphs consist of nodes (representing entities) and edges (representing relationships between entities).
   - Each node typically has features associated with it (like node attributes or embeddings).

2. **Graph Convolution:**
   - GCNs perform convolutional operations on the graph data. However, unlike regular convolutions used in image data, GCNs adapt convolutional operations for graphs.

3. **Neighborhood Aggregation:**
   - GCNs operate by aggregating information from a node's neighborhood, incorporating features from neighboring nodes.
   - Convolution in GCNs involves propagating information from neighboring nodes to update the features of the central node.

4. **Learnable Parameters:**
   - GCNs learn parameters for the convolutional filters, allowing them to adapt and capture complex relationships between nodes in the graph.

Here's a simple example:

Consider a social network where nodes represent users, and edges represent friendships. Each user has attributes like age, interests, and connection strength with other users.

1. **Graph Representation:**
   - Each node (user) is represented as a feature vector containing information about the user (age, interests, etc.).
   - Edges (friendships) connect nodes in the graph.

2. **Graph Convolution:**
   - For each node, a GCN collects information from its neighboring nodes.
   - The features of a node are updated by aggregating information from its neighbors, weighted by the strength of the connections (edges).
   - The updated node features capture not only its own attributes but also information from its friends.

3. **Learnable Parameters:**
   - The GCN learns parameters (weights) for the convolutional filters, adjusting them during training to better capture relationships in the graph.

4. **Task Execution:**
   - The output of the GCN can be used for various tasks such as node classification (predicting node labels), link prediction (predicting missing edges), recommendation systems, etc.

This process continues for multiple layers in deep GCNs, allowing the model to capture increasingly complex relationships and higher-level abstractions across the graph structure.

GCNs have proven effective in various domains due to their ability to handle and learn from graph-structured data efficiently.

In Graph Convolutional Networks (GCNs), the aggregation of information from a node's neighbors weighted by the value of edges is typically done through a mathematical operation known as message passing. This process involves the application of convolutional filters on the graph data to update node representations.

Let's break it down mathematically:

1. **Graph Structure:**
   - Consider a graph $G = (V, E)$ where $V$ represents nodes and $E$ represents edges.
   - Let $A$ be the adjacency matrix representing the connections between nodes, where $A_{ij}$ is the weight of the edge between nodes $i$ and $j$.

2. **Node Representations:**
   - Each node $i$ has an initial feature representation $h_i^{(0)}$. This could be represented as a matrix $H^{(0)}$ where each row corresponds to the feature vector of a node.

3. **Message Passing:**
   - To update the node representations in a GCN, we apply a convolutional operation based on the graph structure.
   - The update rule for a single layer of GCN can be expressed as:

$$
h_i^{(l+1)} = \sigma\left(\sum_{j \in \text{neighbors}(i)} \frac{1}{{\sqrt{d_i \cdot d_j}}}\cdot h_j^{(l)} \cdot W^{(l)}\right)
$$

Here:
- $h_i^{(l)}$ is the representation of node $i$ at layer $l$.
- $\text{neighbors}(i)$ represents the neighbors of node $i$.
- $W^{(l)}$ is the weight matrix for the $l$th layer of the GCN.
- $d_i$ and $d_j$ represent the degrees of nodes $i$ and $j$ respectively.

4. **Explanation of the Equation:**
   - $h_i^{(l+1)}$ is updated by aggregating the information from its neighbors ($h_j^{(l)}$) multiplied by the learnable weight matrix $W^{(l)}$.
   - The summation involves neighbors $j$ of node $i$ with the division by $\sqrt{d_i \cdot d_j}$ as a normalization term (for scaling based on node degrees).
   - The activation function $\sigma$ introduces non-linearity (often ReLU or similar).

5. **Iteration and Stacking:**
   - This process can be repeated for multiple layers of GCN by stacking these operations sequentially to capture higher-order relationships and refine node representations.

In summary, the mathematical operation in GCNs involves updating node representations by aggregating information from neighboring nodes based on edge weights (as normalized by node degrees) and applying learnable weights through a convolutional operation, leading to refined representations capturing the graph structure.

*Multiple layers of Graph Convolutional Networks (GCNs) enable the capturing of higher-order relationships and the refinement of node representations* by allowing information to propagate across the graph in a hierarchical manner. Here's how:

1. **Local and Global Information Aggregation:**
   - Each layer of the GCN aggregates information from a node's immediate neighbors in the previous layer.
   - As the layers progress, information from further neighbors (2-hop, 3-hop, and so on) gets incorporated.
   - This allows the model to capture more extended and complex relationships beyond direct connections.

2. **Non-Linear Transformations:**
   - Each layer typically involves non-linear transformations, such as activation functions (e.g., ReLU), allowing the model to capture more complex patterns and relationships.
   - These transformations help in capturing nonlinearities in the graph structure.

3. **Hierarchical Representation Learning:**
   - Higher layers in the GCN capture more abstract and higher-level features based on the input data.
   - Lower layers might capture more local, node-specific features, while higher layers capture more global and abstract information about the entire graph.

4. **Feature Refinement:**
   - As information passes through multiple layers, the node representations undergo refinement, incorporating information from a broader neighborhood.
   - This refinement process helps nodes to acquire more comprehensive and context-rich representations, capturing nuanced relationships and features.

5. **Adaptive Learning of Weights:**
   - Each layer in the GCN has its own set of learnable parameters (weights) that are adjusted during training.
   - These weights adapt to capture the most relevant information for the task at hand, allowing the model to learn and refine representations specific to the dataset and the target task.

By stacking multiple layers, a GCN effectively leverages these mechanisms to capture increasingly intricate relationships, learn more abstract features, and refine node representations. This hierarchical learning process enables the model to understand and leverage the complex structure inherent in graph data, making it effective for various tasks involving relational data.

# EDA

After initializing the [`KarateClub`](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.KarateClub) dataset, we first can inspect some of its properties.
For example, we can see that this dataset holds exactly **one graph**, and that each node in this dataset is assigned a **34-dimensional feature vector** (which uniquely describes the members of the karate club).
Furthermore, the graph holds exactly **4 classes**, which represent the community each node belongs.

Each graph in PyTorch Geometric is represented by a single [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) object, which holds all the information to describe its graph representation.
We can print the data object anytime via `print(data)` to receive a short summary about its attributes and their shapes:
```
Data(edge_index=[2, 156], x=[34, 34], y=[34], train_mask=[34])
```
We can see that this `data` object holds 4 attributes:
- The `edge_index` property holds the information about the **graph connectivity**, *i.e.*, a tuple of source and destination node indices for each edge.
PyG further refers to
- **node features** as `x` (each of the 34 nodes is assigned a 34-dim feature vector)
- **node labels** as `y` (each node is assigned to exactly one class).
- There also exists an additional attribute called `train_mask`, which describes for which nodes we already know their community assigments.
In total, we are only aware of the ground-truth labels of 4 nodes (one for each community), and the task is to infer the community assignment for the remaining nodes.

The `data` object also provides some **utility functions** to infer some basic properties of the underlying graph.
For example, we can easily infer whether there exists isolated nodes in the graph (*i.e.* there exists no edge to any node), whether the graph contains self-loops (*i.e.*, $(v, v) \in \mathcal{E}$), or whether the graph is undirected (*i.e.*, for each edge $(v, w) \in \mathcal{E}$ there also exists the edge $(w, v) \in \mathcal{E}$).

A summary of the graph dataset:

```
Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
==================================================
Number of nodes: 34
Number of edges: 156
Average node degree: 4.59
Number of training nodes: 4
Training node label rate: 0.12
Has isolated nodes: False
Has self-loops: False
Is undirected: True
```

Let us now inspect the `edge_index` property in more detail:

```
tensor([[ 0,  1],
        [ 0,  2],
        [ 0,  3],
        [ 0,  4],
        [ 0,  5],
        ....
        ....
```

By printing `edge_index`, we can understand how PyG represents graph connectivity internally.
We can see that for each edge, `edge_index` holds a tuple of two node indices, where the first value describes the node index of the source node and the second value describes the node index of the destination node of an edge.

This representation is known as the **COO format (coordinate format)** commonly used for representing sparse matrices.
Instead of holding the adjacency information in a dense representation $\mathbf{A} \in \{ 0, 1 \}^{|\mathcal{V}| \times |\mathcal{V}|}$, PyG represents graphs sparsely, which refers to only holding the coordinates/values for which entries in $\mathbf{A}$ are non-zero.

Importantly, PyG does not distinguish between directed and undirected graphs and treats undirected graphs as a special case of directed graphs in which reverse edges exist for every entry in `edge_index`.

We can further visualize the graph by converting it to the `networkx` library format, which implements, in addition to graph manipulation functionalities, powerful tools for visualization:

![true_graph](https://github.com/rukshar69/Graph-Convolutional-Network-for-Zacharys-Karate-Club/blob/main/blog_images/true_graph.png)

# Implementing Graph Neural Networks

For this, we will use one of the most simple GNN operators, the **GCN layer** ([Kipf et al. (2017)](https://arxiv.org/abs/1609.02907)), which is defined as

![gcn_layer](https://github.com/rukshar69/Graph-Convolutional-Network-for-Zacharys-Karate-Club/blob/main/blog_images/gcn_formula.png)

where $\mathbf{W}^{(\ell + 1)}$ denotes a trainable weight matrix of shape `[num_output_features, num_input_features]` and $c_{w,v}$ refers to a fixed normalization coefficient for each edge.

PyG implements this layer via [`GCNConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv), which can be executed by passing in the node feature representation `x` and the COO graph connectivity representation `edge_index`.

With this, we are ready to create our first Graph Neural Network by defining our network architecture in a `torch.nn.Module` class:

```
GCN(
  (conv1): GCNConv(34, 4)
  (conv2): GCNConv(4, 4)
  (conv3): GCNConv(4, 2)
  (classifier): Linear(in_features=2, out_features=4, bias=True)
)

```

Here, we first initialize all of our building blocks in `__init__` and define the computation flow of our network in `forward`.
We first define and stack **three graph convolution layers**, which correspond to aggregating 3-hop neighborhood information around each node (all nodes up to 3 "hops" away).
In addition, the `GCNConv` layers reduce the node feature dimensionality to $2$, *i.e.*, $34 \rightarrow 4 \rightarrow 4 \rightarrow 2$. Each `GCNConv` layer is enhanced by a [tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html?highlight=tanh#torch.nn.Tanh) non-linearity.

After that, we apply a single linear transformation ([`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear)) that acts as a classifier to map our nodes to 1 out of the 4 classes/communities.

We return both the output of the final classifier as well as the final node embeddings produced by our GNN.
We proceed to initialize our final model via `GCN()`, and printing our model produces a summary of all its used sub-modules.

## Embedding the Karate Club Network

Let's take a look at the node embeddings produced by our GNN.
Here, we pass in the initial node features `x` and the graph connectivity information `edge_index` to the model and visualize its 2-dimensional embedding.

![initial_node_embedding](https://github.com/rukshar69/Graph-Convolutional-Network-for-Zacharys-Karate-Club/blob/main/blog_images/initial_node_embedding.png)

We can see that our GNN has produced a 2-dimensional embedding for each of the 34 nodes in the karate club network.

## Training on the Karate Club Network

But can we do better? Let's look at an example on how to train our network parameters based on the knowledge of the community assignments of 4 nodes in the graph (one for each community):

We make use of a semi-supervised or transductive learning procedure: We simply train against one node per class but are allowed to make use of the complete input graph data.

Training our model is very similar to any other PyTorch model.
In addition to defining our network architecture, we define a loss criterion (here, [`CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)) and initialize a stochastic gradient optimizer (here, [`Adam`](https://pytorch.org/docs/stable/optim.html?highlight=adam#torch.optim.Adam)).
After that, we perform multiple rounds of optimization, where each round consists of a forward and backward pass to compute the gradients of our model parameters w.r.t. to the loss derived from the forward pass.

Note that our semi-supervised learning scenario is achieved by the following line:
```
loss = criterion(out[data.train_mask], data.y[data.train_mask])
```
While we compute node embeddings for all of our nodes, we **only make use of the training nodes for computing the loss**.
Here, this is implemented by filtering the output of the classifier `out` and ground-truth labels `data.y` to only contain the nodes in the `train_mask`.

We train our model and observe how our node embeddings evolve. Below is the YouTube video of the training process in action where we can see the evolution of our node embeddings along with the decrease of the loss.

[![training GCN node embeddings](https://img.youtube.com/vi/Q4wh2NlgpSE/0.jpg)](https://www.youtube.com/watch?v=Q4wh2NlgpSE)

A graph of the decrease in the loss per epoch can be seen below.

![loss_vs_epoch](https://github.com/rukshar69/Graph-Convolutional-Network-for-Zacharys-Karate-Club/blob/main/blog_images/loss_vs_epoch.png)

Precision, recall, and F1-score for each class label(0,1,2,3)

![score_per_class](https://github.com/rukshar69/Graph-Convolutional-Network-for-Zacharys-Karate-Club/blob/main/blog_images/score_per_class.png)

Distribution of labels for true and predicted classes.

![label_dist](https://github.com/rukshar69/Graph-Convolutional-Network-for-Zacharys-Karate-Club/blob/main/blog_images/label_dist.png)

As one can see, our 3-layer GCN model manages to linearly separate the communities and classifies most of the nodes correctly. The loss has also gradually decreased to a near plateau meaning the training is progressing towards the right direction. The overall *accuracy* is over **82** percent. While looking at the class-based metrics(f1-score, precision, and recall), **label 1** has the least *f1-score* of *75%*. By observing the class distribution of the true and predicted labels in the bar graphs, we can see some of the label 1 points have been classified as label 0.

Comparing graph visualizations of prediction labels and true labels, we find, at the fuzzy border region between label 0 and label 1 (green and blue respectively), some blue label 1 nodes are misclassified as green label 0. Hence, label 1 had less f1-score.

![predict_true_graph_comparison](https://github.com/rukshar69/Graph-Convolutional-Network-for-Zacharys-Karate-Club/blob/main/blog_images/predict_true_graph_comparison.png)
# References


