import os
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from base.ptg_datasets import is_graph_directed
from src.aux.utils import GRAPHS_DIR


def ptg_to_networkx(ptg_graph):
    node_attrs = None
    if ptg_graph.x != None:
        node_attrs = ["x"]
    edge_attrs = None
    if ptg_graph.edge_attr != None:
        edge_attrs = ["edge_attr"]

    # "graph_attrs" parameter is not working in this torch_geometric version
    nx_graph = to_networkx(ptg_graph, node_attrs=node_attrs, edge_attrs=edge_attrs,
                           to_undirected=not is_graph_directed(ptg_graph))

    return nx_graph


def networkx_to_ptg(nx_graph):
    node_attribute_names = set()
    edge_attribute_names = set()

    # Iterating through the nodes and collect unique attribute names
    for node, data in nx_graph.nodes(data=True):
        for attribute_name in data:
            node_attribute_names.add(attribute_name)

    # Checking that attributes have numeric types
    for attr in node_attribute_names:
        attr_name = nx.get_node_attributes(nx_graph, attr)
        for name in attr_name:
            if not isinstance(attr_name[name], (int, float, complex, list)):
                raise RuntimeError("Wrong NODE attribute type!!!")

    # Iterating through the edges and collect unique attribute names
    for u, v, data in nx_graph.edges(data=True):
        for attribute_name in data:
            edge_attribute_names.add(attribute_name)

    # Checking that attributes have numeric types
    for attr in edge_attribute_names:
        attr_name = nx.get_edge_attributes(nx_graph, attr)
        for name in attr_name:
            if not isinstance(attr_name[name], (int, float, complex)):
                raise RuntimeError("Wrong EDGE attribute type!!!", nx_graph)

    # Converting the set of unique attribute names to a list
    node_attribute_names_list = list(node_attribute_names)
    edge_attribute_names_list = list(edge_attribute_names)

    if len(node_attribute_names_list) < 1:
        node_attribute_names_list = None
    if len(edge_attribute_names_list) < 1:
        edge_attribute_names_list = None

    ptg_graph = from_networkx(nx_graph, group_node_attrs=node_attribute_names_list,
                              group_edge_attrs=edge_attribute_names_list)
    return ptg_graph


def read_nx_graph(data_format, path):
    # FORMATS THAT ARE NOT SUPPORTED:
    # gexf, multiline_adjlist, weighted_edgelist
    if data_format == ".adjlist":  # This format does not store graph or node attributes.
        return nx.read_adjlist(path)
    elif data_format == ".edgelist":
        return nx.read_edgelist(path)
    elif data_format == ".gml":  # Only works with graphs that have node, edge attributes
        return nx.read_gml(path)
    # # GRAPHML DOESN'T WORK WITH from_networkx()
    # elif data_format == "graphml":
    #     return nx.read_graphml(path)
    # # LEDA format is not supported as it stores edge attributes as strings
    # elif data_format == "leda":
    #     return nx.read_leda(path)
    elif data_format == ".g6":
        return nx.read_graph6(path)
    elif data_format == ".s6":
        return nx.read_sparse6(path)
    # # PAJEK format is not supported as it stores node attributes as strings
    # elif data_format == "pajek": # Only works with graphs that have node labels
    #     return nx.read_pajek(path)
    else:
        raise RuntimeError("the READING format is NOT SUPPORTED!!!")


def write_nx_graph(graph, data_format, path):
    if data_format == ".adjlist":
        return nx.write_adjlist(graph, path)
    # elif data_format == "multiline_adjlist":
    #     return nx.write_multiline_adjlist(graph, path)
    elif data_format == ".edgelist":
        return nx.write_edgelist(graph, path)
    # elif data_format == "weighted_edgelist":
    #     return nx.write_weighted_edgelist(graph, path)
    # elif data_format == "gexf":
    #     return nx.write_gexf(graph, path)
    elif data_format == ".gml":
        return nx.write_gml(graph, path)
    # elif data_format == "graphml":
    #     return nx.write_graphml(graph, path)
    # elif data_format == "leda":
    #     return nx.write_leda(graph, path)
    elif data_format == ".g6":
        return nx.write_graph6(graph, path)
    elif data_format == ".s6":
        return nx.write_sparse6(graph, path)
    # elif data_format == "pajek":
    #     return nx.write_pajek(graph, path)
    else:
        raise RuntimeError("the WRITING format is NOT SUPPORTED!!!")


# Reading NX graphs from files in given formats
def read_nx_graphs(format_list, path_list):
    # Creating a dict where keys are formats and values are lists of graphs
    nx_graphs_dict = {format: [] for format in format_list}
    for format in format_list:
        for path in path_list:
            if path.endswith(format):
                nx_graph = read_nx_graph(format, path)
                if (type(nx_graph) == list):
                    for graph in nx_graph:
                        nx_graphs_dict[format].append(graph)
                    break
                nx_graphs_dict[format].append(nx_graph)
                break
    return nx_graphs_dict


def converting_func(nx_graphs_dict):
    ptg_graphs_dict = {format: [] for format in nx_graphs_dict.keys()}
    for format in nx_graphs_dict.keys():
        for graph in nx_graphs_dict[format]:
            ptg_graph = networkx_to_ptg(graph)
            ptg_graphs_dict[format].append(ptg_graph)

    new_nx_graphs_dict = {format: [] for format in ptg_graphs_dict.keys()}
    for format in ptg_graphs_dict.keys():
        for graph in ptg_graphs_dict[format]:
            nx_graph = ptg_to_networkx(graph)
            new_nx_graphs_dict[format].append(nx_graph)

    return new_nx_graphs_dict


# Writing graphs from graphs_dict into given directory, file names will be "./output_graph<num>.<format>"
def write_nx_graphs(graphs_dict, output_dir):  # "output_dir" should be a str
    for format in graphs_dict.keys():
        i = 0
        for graph in graphs_dict[format]:
            write_nx_graph(graph, format, f"{output_dir}/output_graph{i}{format}")
            i += 1
    return


# Extracting paths from a given directory
def path_handler(dir_path):
    path_list = []
    for file_path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file_path)):
            path_list.append(os.path.join(dir_path, file_path))
    return path_list


formats_list = [".adjlist", ".edgelist", ".gml", ".g6", ".s6"]

input_dir = GRAPHS_DIR / 'networkx-graphs' / 'input'
output_dir = GRAPHS_DIR / 'networkx-graphs' / 'output'

if __name__ == '__main__':
    input_paths = path_handler(input_dir)
    nx_graphs_dict = read_nx_graphs(formats_list, input_paths)
    new_nx_graphs_dict = converting_func(nx_graphs_dict)

    write_nx_graphs(new_nx_graphs_dict, str(output_dir))
