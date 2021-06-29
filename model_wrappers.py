import torch
import numpy as np
import os
import shutil

class model_wrapper():
    def __init__(self, model, save_dir=None):
        self.model = model
        self.save_dir = save_dir

    def save_torch_output(self, block_name, var_dict):
        dest_top = self.save_dir+"/torch_intermediates/%s/"%block_name
        if not os.path.isdir(dest_top):
            os.makedirs(dest_top, exist_ok=True)
        for k,v in var_dict.items():
            dest_full = dest_top + k + ".csv"
            np.savetxt(dest_full, v, delimiter=",", fmt='%.3f')

    def move_hls_output(self, block_name, var_names):
        src_top = self.save_dir + "/firmware/%s_"%block_name
        dest_top = self.save_dir+"/hls_intermediates/%s/"%block_name
        if not os.path.isdir(dest_top):
            os.makedirs(dest_top, exist_ok=True)
        for v in var_names:
            src_full = src_top + v + ".csv"
            dest_full = dest_top + v + ".csv"
            shutil.move(src_full, dest_full)

    def EdgeBlock_forward(self, block, node_attr, edge_attr, edge_index, out_dim):
        raise NotImplementedError

    def NodeBlock_forward(self, block, node_attr, edge_attr_aggr, out_dim):
        # dimensions
        n_node, node_dim = node_attr.shape
        edge_dim = edge_attr_aggr.shape[1]

        # block layers
        layers = block.layers._modules
        fc1 = layers['0']
        relu1 = layers['1']
        fc2 = layers['2']
        relu2 = layers['3']
        fc3 = layers['4']

        # input, output
        phi_inputs = torch.zeros((n_node, node_dim + edge_dim))
        node_update = torch.zeros((n_node, out_dim))

        # intermediates
        fc1_out = torch.zeros((n_node, fc1.out_features))
        relu1_out = torch.zeros((n_node, fc1.out_features))
        fc2_out = torch.zeros((n_node, fc2.out_features))
        relu2_out = torch.zeros((n_node, fc2.out_features))
        fc3_out = torch.zeros((n_node, fc3.out_features))

        for i in range(n_node):
            node_i = node_attr[i]
            edge_aggr_i = edge_attr_aggr[i]
            phi_input_i = torch.cat((node_i, edge_aggr_i))

            phi_inputs[i] = phi_input_i
            fc1_out[i] = fc1(phi_input_i)
            relu1_out[i] = relu1(fc1_out[i])
            fc2_out[i] = fc2(relu1_out[i])
            relu2_out[i] = relu2(fc2_out[i])
            fc3_out[i] = fc3(relu2_out[i])
            node_update[i] = block(phi_input_i)

        if self.save_dir is not None:
            block_name = "O"

            # save torch intermediates
            torch_var_dict = {"out": node_update.detach().cpu().numpy(),
                              "block_inputs": phi_inputs.detach().cpu().numpy(),
                              "fc1_out": fc1_out.detach().cpu().numpy(),
                              "relu1_out": relu1_out.detach().cpu().numpy(),
                              "fc2_out": fc2_out.detach().cpu().numpy(),
                              "relu2_out": relu2_out.detach().cpu().numpy(),
                              "fc3_out": fc3_out.detach().cpu().numpy()}
            self.save_torch_output(block_name, torch_var_dict)

            # save hls intermediates
            hls_var_names = ["out", "block_inputs", "fc1_out", "relu1_out", "fc2_out", "relu2_out", "fc3_out"]
            self.move_hls_output(block_name, hls_var_names)

        return node_update

    def forward(self, data):
        node_attr = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        edge_update_1, edge_update_aggr_1 = self.EdgeBlock_forward(self.model.R1, node_attr, edge_attr, edge_index, out_dim=4)  # IN_edge_module_1
        node_update = self.NodeBlock_forward(self.model.O, node_attr, edge_update_aggr_1, out_dim=3)  # IN_node_module
        edge_update_2, edge_update_aggr_2 = self.EdgeBlock_forward(self.model.R2, node_update, edge_update_1, edge_index, out_dim=1)  # IN_edge_module
        out = torch.sigmoid(edge_update_2)
        return out

class model_add_wrapper(model_wrapper):
    def __init__(self, model, save_dir=None):
        super().__init__(model, save_dir)

    def EdgeBlock_forward(self, block, node_attr, edge_attr, edge_index, out_dim):
        # dimensions
        n_node, node_dim = node_attr.shape
        n_edge, edge_dim = edge_attr.shape

        # block layers
        layers = block.layers._modules
        fc1 = layers['0']
        relu1 = layers['1']
        fc2 = layers['2']
        relu2 = layers['3']
        fc3 = layers['4']

        # input, output
        phi_inputs = torch.zeros((n_edge, 2*node_dim+edge_dim))
        edge_update = torch.zeros((n_edge, out_dim))
        edge_update_aggr = torch.zeros((n_node, out_dim))

        # intermediates
        fc1_out = torch.zeros((n_edge, fc1.out_features))
        relu1_out = torch.zeros((n_edge, fc1.out_features))
        fc2_out = torch.zeros((n_edge, fc2.out_features))
        relu2_out = torch.zeros((n_edge, fc2.out_features))
        fc3_out = torch.zeros((n_edge, fc3.out_features))

        for i in range(n_edge):
            s = edge_index[0,i]
            r = edge_index[1,i]

            edge_i = edge_attr[i,:] #edge
            node_s = node_attr[s,:]  #edge sender
            node_r = node_attr[r,:] #edge receiver

            node_concat = torch.cat((node_r, node_s), dim=0)
            phi_input_i = torch.cat((node_concat, edge_i), dim=0)

            phi_inputs[i] = phi_input_i
            fc1_out[i] = fc1(phi_input_i)
            relu1_out[i] = relu1(fc1_out[i])
            fc2_out[i] = fc2(relu1_out[i])
            relu2_out[i] = relu2(fc2_out[i])
            fc3_out[i] = fc3(relu2_out[i])
            edge_update[i] = block(phi_input_i)

            for j in range(out_dim):
                edge_update_aggr[r,j] += edge_update[i,j]

        if self.save_dir is not None:

            if out_dim==4:
                block_name="R1"
            else:
                block_name="R2"

            # save torch intermediates
            torch_var_dict = {
                "out": edge_update.detach().cpu().numpy(),
                "out_aggr": edge_update_aggr.detach().cpu().numpy(),
                "block_inputs": phi_inputs.detach().cpu().numpy(),
                "fc1_out": fc1_out.detach().cpu().numpy(),
                "relu1_out": relu1_out.detach().cpu().numpy(),
                "fc2_out": fc2_out.detach().cpu().numpy(),
                "relu2_out": relu2_out.detach().cpu().numpy(),
                "fc3_out": fc3_out.detach().cpu().numpy()
            }
            self.save_torch_output(block_name, torch_var_dict)

            # save hls intermediates
            hls_var_names = ["out", "out_aggr", "block_inputs", "fc1_out", "relu1_out", "fc2_out", "relu2_out", "fc3_out"]
            self.move_hls_output(block_name, hls_var_names)

        return edge_update, edge_update_aggr

class model_mean_wrapper(model_wrapper):
    def __init__(self, model, save_dir=None):
        super().__init__(model, save_dir)

    def EdgeBlock_forward(self, block, node_attr, edge_attr, edge_index, out_dim):
        # dimensions
        n_node, node_dim = node_attr.shape
        n_edge, edge_dim = edge_attr.shape

        # block layers
        layers = block.layers._modules
        fc1 = layers['0']
        relu1 = layers['1']
        fc2 = layers['2']
        relu2 = layers['3']
        fc3 = layers['4']

        # input, output
        phi_inputs = torch.zeros((n_edge, 2 * node_dim + edge_dim))
        edge_update = torch.zeros((n_edge, out_dim))
        edge_update_aggr = torch.zeros((n_node, out_dim))

        # intermediates
        fc1_out = torch.zeros((n_edge, fc1.out_features))
        relu1_out = torch.zeros((n_edge, fc1.out_features))
        fc2_out = torch.zeros((n_edge, fc2.out_features))
        relu2_out = torch.zeros((n_edge, fc2.out_features))
        fc3_out = torch.zeros((n_edge, fc3.out_features))
        num_edge_per_node = torch.zeros((n_node,))

        for i in range(n_edge):
            s = edge_index[0,i]
            r = edge_index[1,i]
            num_edge_per_node[r] += 1

            edge_i = edge_attr[i,:] #edge
            node_s = node_attr[s,:]  #edge sender
            node_r = node_attr[r,:] #edge receiver

            node_concat = torch.cat((node_r, node_s), dim=0)
            phi_input_i = torch.cat((node_concat, edge_i), dim=0)

            phi_inputs[i] = phi_input_i
            fc1_out[i] = fc1(phi_input_i)
            relu1_out[i] = relu1(fc1_out[i])
            fc2_out[i] = fc2(relu1_out[i])
            relu2_out[i] = relu2(fc2_out[i])
            fc3_out[i] = fc3(relu2_out[i])
            edge_update[i] = block(phi_input_i)

            for j in range(out_dim):
                edge_update_aggr[r,j] += edge_update[i,j]

        for i in range(n_node):
            n_edge_i = num_edge_per_node[i]
            if (n_edge_i > 1):
                edge_update_aggr[i] = edge_update_aggr[i]/num_edge_per_node[i]

        if self.save_dir is not None:

            if out_dim == 4:
                block_name = "R1"
            else:
                block_name = "R2"

            # save torch intermediates
            torch_var_dict = {
                "out": edge_update.detach().cpu().numpy(),
                "out_aggr": edge_update_aggr.detach().cpu().numpy(),
                "block_inputs": phi_inputs.detach().cpu().numpy(),
                "fc1_out": fc1_out.detach().cpu().numpy(),
                "relu1_out": relu1_out.detach().cpu().numpy(),
                "fc2_out": fc2_out.detach().cpu().numpy(),
                "relu2_out": relu2_out.detach().cpu().numpy(),
                "fc3_out": fc3_out.detach().cpu().numpy(),
                "num_edge_per_node": num_edge_per_node.detach().cpu().numpy()
            }
            self.save_torch_output(block_name, torch_var_dict)

            # save hls intermediates
            var_names = ["out", "out_aggr", "num_edge_per_node", "block_inputs", "fc1_out", "relu1_out", "fc2_out", "relu2_out", "fc3_out"]
            self.move_hls_output(block_name, var_names)

        return edge_update, edge_update_aggr

class model_max_wrapper(model_wrapper):
    def __init__(self, model, save_dir=None):
        super().__init__(model, save_dir)

    def EdgeBlock_forward(self, block, node_attr, edge_attr, edge_index, out_dim):
        # dimensions
        n_node, node_dim = node_attr.shape
        n_edge, edge_dim = edge_attr.shape

        # block layers
        layers = block.layers._modules
        fc1 = layers['0']
        relu1 = layers['1']
        fc2 = layers['2']
        relu2 = layers['3']
        fc3 = layers['4']

        # input, output
        phi_inputs = torch.zeros((n_edge, 2 * node_dim + edge_dim))
        edge_update = torch.zeros((n_edge, out_dim))
        edge_update_aggr = torch.zeros((n_node, out_dim))

        # intermediates
        fc1_out = torch.zeros((n_edge, fc1.out_features))
        relu1_out = torch.zeros((n_edge, fc1.out_features))
        fc2_out = torch.zeros((n_edge, fc2.out_features))
        relu2_out = torch.zeros((n_edge, fc2.out_features))
        fc3_out = torch.zeros((n_edge, fc3.out_features))

        for i in range(n_edge):
            s = edge_index[0,i]
            r = edge_index[1,i]

            edge_i = edge_attr[i,:] #edge
            node_s = node_attr[s,:]  #edge sender
            node_r = node_attr[r,:] #edge receiver

            node_concat = torch.cat((node_r, node_s), dim=0)
            phi_input_i = torch.cat((node_concat, edge_i), dim=0)

            phi_inputs[i] = phi_input_i
            fc1_out[i] = fc1(phi_input_i)
            relu1_out[i] = relu1(fc1_out[i])
            fc2_out[i] = fc2(relu1_out[i])
            relu2_out[i] = relu2(fc2_out[i])
            fc3_out[i] = fc3(relu2_out[i])
            edge_update[i] = block(phi_input_i)

            for j in range(out_dim):
                if edge_update_aggr[r,j] < edge_update[i,j]:
                    edge_update_aggr[r,j] = edge_update[i,j]

        if self.save_dir is not None:

            if out_dim == 4:
                block_name = "R1"
            else:
                block_name = "R2"

            # save torch intermediates
            torch_var_dict = {
                "out": edge_update.detach().cpu().numpy(),
                "out_aggr": edge_update_aggr.detach().cpu().numpy(),
                "block_inputs": phi_inputs.detach().cpu().numpy(),
                "fc1_out": fc1_out.detach().cpu().numpy(),
                "relu1_out": relu1_out.detach().cpu().numpy(),
                "fc2_out": fc2_out.detach().cpu().numpy(),
                "relu2_out": relu2_out.detach().cpu().numpy(),
                "fc3_out": fc3_out.detach().cpu().numpy()
            }
            self.save_torch_output(block_name, torch_var_dict)

            # save hls intermediates
            var_names = ["out", "out_aggr", "block_inputs", "fc1_out", "relu1_out", "fc2_out", "relu2_out", "fc3_out"]
            self.move_hls_output(block_name, var_names)


        return edge_update, edge_update_aggr

