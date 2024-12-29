# Homework 5 - Algorithmic Methods of Data Mining
## USA Airport Flight Analysis

This repository contains the solution for Homework 5 of the course Algorithmic Methods of Data Mining (Data Science, Università di Roma "La Sapienza")

## Description of the Homework

Air travel connects cities across the United States, forming a dense and dynamic network of routes that play a critical role in transportation, commerce, and connectivity. In this assignment, we will analyze the USA Airport Dataset, which includes detailed information about airports, routes, and traffic patterns. Our task is to uncover meaningful insights such as identifying the busiest hubs, analyzing flight routes. Additionally, we will explore the network structure to detect communities of interconnected airports, evaluate their centrality, and assess the impact of disruptions. This analysis will require the application of data visualization, network analysis, and optimization techniques to deliver actionable findings.

The dataset contains information about flights between airports in the United States. You can download and explore the dataset from this link: <https://www.kaggle.com/datasets/flashgordon/usa-airport-dataset> . In this task, we will analyze the basic features of the flight network graph, such as its size, density, and degree distribution.

**IMPORTANT**: we can use packages like **NetworkX** or similar tools solely for handling the data structure. However, when implementing the algorithm for your functionalities, we must write it entirely **from scratch** without relying on pre-implemented functions from libraries that perform any steps on our behalf.

### [Q1]

After implementing some functions requested, we have these results:

- The number of nodes and edges.
- The graph density.
- Degree distribution plots for in-degree and out-degree.
- A table of identified hubs.
- Top routes by passenger flow (table and bar chart).
- Top routes by passenger efficiency (table and bar chart).
- An interactive map showing flight routes.

### [Q2]

In any network, certain nodes (airports, in this case) play a critical role in maintaining connectivity and flow. Centrality measures are used to identify these nodes.

We have implemented some functions to find centrality measures and then we have:

- Asked LLM (eg. ChatGPT) to suggest alternative centrality measures that might be relevant to this task.
- Implemented one of these measures suggested by the LLM, compared its results to the centralities we've had already computed, and analyzed whether it adds any new insights.

### [Q3]

Whenever you plan to fly to a specific city, your goal is to find the most efficient and fastest flight to reach your destination. In the system we are designing, the best route is defined as the one that minimizes the total distance flown to the greatest extent possible. So, we have implemented a function that, given an origin and destination city, determines the best possible route between them. To simplify, the focus has been be limited to flights operating on a specific day.

### [Q4]

Imagine all these flights are currently managed by a single airline. However, this airline is facing bankruptcy and plans to transfer the management of part of its operations to a different airline. The airline is willing to divide the flight network into two distinct partitions, p_1 and p_2, such that no flights connect any airport in p_1 to any airport in
p_2. The flights in p_1 will remain under the management of the original airline, while those in p_2 will be handed over to the second airline. Any remaining flights needed to connect these two partitions will be determined later.

In graph theory, this task is known as a graph disconnection problem. Our goal was to write a function that removes the minimum number of flights between airports to separate the original flight network into two disconnected subgraphs.

### [Q5]

Airlines can optimize their operations by identifying communities within a flight network. These communities represent groups of airports with strong connections, helping airlines pinpoint high-demand regions for expansion or underserved areas for consolidation. By analyzing these communities, airlines can improve resource allocation, reduce costs, and enhance service quality.

In this task, we were asked to analyze the graph and identify the communities based on the flight network provided. For the airline, the primary focus is on the cities, so our communities reflects the connectivity between cities through the flights that link them.

### [BONUS QUESTION]

MapReduce is ideal for network analysis as it enables parallel processing of large graph datasets, making it scalable and efficient. By breaking tasks into map and reduce steps, it allows for distributed analysis of networks, which is essential for handling large-scale graph problems like connected components.

In this task, we were required to use PySpark and the MapReduce paradigm to identify the connected components in a flight network graph. The focus was on airports rather than cities.

Note: we were not allowed to use pre-existing packages or functions in PySpark; instead, we had to implement the algorithm from scratch using the MapReduce paradigm.

### [AQ]

Arya needs to travel between cities using a network of flights. Each flight has a fixed cost (in euros), and she wants to find the cheapest possible way to travel from her starting city to her destination city. However, there are some constraints on the journey:

- Arya can make at most *k* stops during her trip (this means up to *k+1* flights).
- If no valid route exists within these constraints, the result should be *-1*.
  
Given a graph of cities connected by flights, our job was to find the minimum cost for Arya to travel between two specified cities (src to dst) while following the constraints.

## Google Colab Viewer

In case of difficulties viewing the rendered outputs and graphs on the merged_notebook.ipynb file, we provide this link: [merged_notebook.ipynb](https://colab.research.google.com/drive/148AIps5ENUIE7BEkTKiPdnFpKHLtbe6T?usp=sharing) to be able to view everything.

## Structure of the Repository

The repository contains different files, such as:

- LICENSE
- README (this file)
- functions.py (file containing some functions used during the homework 5)
- merged_notebook.ipynb (notebook where you can find all the questions done together in a unique place)
- pic1.png / pic2.png / pic3.png (in case you can't open the fle named top_routes_map.html)
- top_routes_map.html
- usa_airports_map_with_routes.html
- usa_airports_map_with_routes.png (in case you can't open the fle named usa_airports_map_with_routes.html)

