import os
from pathlib import Path

import yaml

def load_yaml(yaml_path):
	"""
	Loads YAML  file
	
	Parameters:
		yaml_path (str): Path to the yaml file.

	Returns: 
		dict: Content of the input yaml file.
	"""
	with open(yaml_path, 'r') as f: 
		content = yaml.safe_load(f)
	
	return content


def generate_dag(pipeline_config):
	"""
	Generates a Directed Acyclic Graph (DAG) representation of the configuration file.
	
	Parameters:
		pipeline_config (str): Path of the configuration file. 

	"""
	nodes = {}
	edges = []
	
	for task in pipeline_config: 
		parent_node = task['dependency']
		child_node = task['unique_id']
		
		nodes[child_node] = {
			"name": task['task'],
			"unique_id": task['unique_id'],
			"parameters": task['parameters'],
			"dependency": task['dependency']
		}

		edge = (parent_node, child_node)
		edges.append(edge)

	return nodes, edges


def get_sink_nodes(edges):
	"""
	Returns the sink nodes of a DAG.

	Parameters:
		edges (list): List of edges of a DAG.
	
	Returns: 
		list: Sink nodes of a DAG.
	"""
	sink_nodes = []

	for edge in edges:
		parent_node, child_node = edge

		if parent_node in sink_nodes:
			sink_nodes.remove(parent_node)

		sink_nodes.append(child_node)

	return sink_nodes


def parse_pipeline_config(config_file):
	"""
	Parses a configuration file.
	
	Parameters:	
		config_file (str): Path to configuration file.
	
	Returns:
		tuple: Tuple of (nodes, edges, sink_nodes).
	"""

	# Loads the YAML configuration file as dict
	config = load_yaml(config_file)

	# Gets the pipeline config from the dict
	pipeline_config = config['pipeline']

	# Generates DAG from the pipeline config
	nodes, edges = generate_dag(pipeline_config)

	# Get the sinks from the DAG 
	sink_nodes = get_sink_nodes(edges)

	return (nodes, edges, sink_nodes)


def main():
	print('Test: get_sink_nodes')
	assert get_sink_nodes([(1, 2)]) == [2]
	assert get_sink_nodes(((1, 2), (2, 3), (3, 4), (3, 5), (1, 6))) == [4, 5, 6]
	print('Test passed')


if __name__ == '__main__':
	main()