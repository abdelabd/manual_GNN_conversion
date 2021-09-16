//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t node_attr[N_NODE*NODE_DIM], input2_t edge_attr[N_EDGE*EDGE_DIM], input3_t edge_index[N_EDGE*TWO],
    result_t layer8_out[N_EDGE*LAYER7_OUT_DIM],
    unsigned short &const_size_in_1, unsigned short &const_size_in_2, unsigned short &const_size_in_3,
    unsigned short &const_size_out_1
) {
    int edge_dim = EDGE_DIM;
    int node_dim = NODE_DIM;
    int two = TWO;
    //hls-fpga-machine-learning insert IO
    if(RESOURCE_LIMIT)
    {
        #pragma HLS ARRAY_RESHAPE variable=node_attr cyclic factor = node_dim dim=1
        #pragma HLS ARRAY_RESHAPE variable=edge_attr cyclic factor = edge_dim dim=1
        #pragma HLS ARRAY_RESHAPE variable=edge_index cyclic factor = two  dim=1
        #pragma HLS INTERFACE ap_vld port=node_attr,edge_attr,edge_index,layer8_out 
        #pragma HLS DATAFLOW
    }
    else
    {
        #pragma HLS ARRAY_RESHAPE variable=node_attr complete dim=0
        #pragma HLS ARRAY_RESHAPE variable=edge_attr complete dim=0
        #pragma HLS ARRAY_RESHAPE variable=edge_index complete dim=0
        #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
        #pragma HLS INTERFACE ap_vld port=node_attr,edge_attr,edge_index,layer8_out 
        #pragma HLS PIPELINE 
    }

    const_size_in_1 = N_NODE*NODE_DIM;
    const_size_in_2 = N_EDGE*EDGE_DIM;
    const_size_in_3 = N_EDGE*TWO;
    const_size_out_1 = N_EDGE*LAYER7_OUT_DIM;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 80>(R1_w0, "R1_w0.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(R1_w1, "R1_w1.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(R1_w2, "R1_w2.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(R1_w3, "R1_w3.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(R1_b0, "R1_b0.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(R1_b1, "R1_b1.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(R1_b2, "R1_b2.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(R1_b3, "R1_b3.txt");
        nnet::load_weights_from_txt<model_default_t, 56>(O_w0, "O_w0.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(O_w1, "O_w1.txt");
        nnet::load_weights_from_txt<model_default_t, 24>(O_w2, "O_w2.txt");
        nnet::load_weights_from_txt<model_default_t, 24>(O_w3, "O_w3.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(O_b0, "O_b0.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(O_b1, "O_b1.txt");
        nnet::load_weights_from_txt<model_default_t, 3>(O_b2, "O_b2.txt");
        nnet::load_weights_from_txt<model_default_t, 3>(O_b3, "O_b3.txt");
        nnet::load_weights_from_txt<model_default_t, 80>(R2_w0, "R2_w0.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(R2_w1, "R2_w1.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(R2_w2, "R2_w2.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(R2_w3, "R2_w3.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(R2_b0, "R2_b0.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(R2_b1, "R2_b1.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(R2_b2, "R2_b2.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(R2_b3, "R2_b3.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers
    if(RESOURCE_LIMIT)
    {
        input_t node_attr_1[N_NODE*NODE_DIM];
        #pragma HLS ARRAY_PARTITION variable=node_attr_1 cyclic factor=node_dim dim=1
        input_t node_attr_2[N_NODE*NODE_DIM];
        #pragma HLS ARRAY_PARTITION variable=node_attr_2 cyclic factor=node_dim dim=1   
        nnet::replicate<input_t,N_NODE,NODE_DIM>(node_attr,node_attr_1,node_attr_2);
        input3_t edge_index_1[N_EDGE*TWO];
        #pragma HLS ARRAY_PARTITION variable=edge_index_1 cyclic factor=2 dim=1
        input3_t edge_index_2[N_EDGE*TWO];
        #pragma HLS ARRAY_PARTITION variable=edge_index_2 cyclic factor=2 dim=1   
        nnet::replicate<input3_t,N_EDGE,TWO>(edge_index,edge_index_1,edge_index_2);
        input3_t edge_index_3[N_EDGE*TWO];
        #pragma HLS ARRAY_PARTITION variable=2 cyclic factor=node_dim dim=1
        input3_t edge_index_4[N_EDGE*TWO];
        #pragma HLS ARRAY_PARTITION variable=2 cyclic factor=node_dim dim=1  
        nnet::replicate<input3_t,N_EDGE,TWO>(edge_index_2,edge_index_3,edge_index_4);
        layer4_t layer4_out[N_EDGE*LAYER4_OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=layer4_out cyclic factor=edge_dim dim=1
        nnet::edgeblock<input2_t, input3_t, layer4_t, config4>(node_attr_1, edge_attr, edge_index_1, layer4_out, R1_w0, R1_b0, R1_w1, R1_b1, R1_w2, R1_b2, R1_w3, R1_b3); // R1
        layer4_t layer4_out_1[N_EDGE*LAYER4_OUT_DIM];
        layer4_t layer4_out_2[N_EDGE*LAYER4_OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=layer4_out_1 cyclic factor=edge_dim dim=1
        #pragma HLS ARRAY_PARTITION variable=layer4_out_2 cyclic factor=edge_dim dim=1
        nnet::replicate<layer4_t,N_EDGE,LAYER4_OUT_DIM>(layer4_out,layer4_out_1,layer4_out_2);

        layer5_t layer5_out[N_NODE*LAYER5_OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=layer5_out cyclic factor=edge_dim dim=1
        nnet::aggregate<input2_t, input3_t, layer5_t, aggregation_config5>(layer4_out_1, edge_index_3, layer5_out); 

        layer6_t layer6_out[N_NODE*LAYER6_OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=layer6_out cyclic factor=node_dim dim=1
        nnet::nodeblock<input_t, layer6_t, config6>(node_attr_2, layer5_out, layer6_out, O_w0, O_b0, O_w1, O_b1, O_w2, O_b2, O_w3, O_b3); // O
        layer7_t layer7_out[N_EDGE*LAYER7_OUT_DIM];
        nnet::edgeblock<input2_t, input3_t, layer7_t, config7>(layer6_out, layer4_out_2, edge_index_4, layer8_out, R2_w0, R2_b0, R2_w1, R2_b1, R2_w2, R2_b2, R2_w3, R2_b3); // R2
    }
    else 
    {
        layer4_t layer4_out[N_EDGE*LAYER4_OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
        nnet::edgeblock<input2_t, input3_t, layer4_t, config4>(node_attr, edge_attr, edge_index, layer4_out, R1_w0, R1_b0, R1_w1, R1_b1, R1_w2, R1_b2, R1_w3, R1_b3); // R1

        layer5_t layer5_out[N_NODE*LAYER5_OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
        nnet::aggregate<input2_t, input3_t, layer5_t, aggregation_config5>(layer4_out, edge_index, layer5_out); // aggr5

        layer6_t layer6_out[N_NODE*LAYER6_OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
        nnet::nodeblock<input_t, layer6_t, config6>(node_attr, layer5_out, layer6_out, O_w0, O_b0, O_w1, O_b1, O_w2, O_b2, O_w3, O_b3); // O

        layer7_t layer7_out[N_EDGE*LAYER7_OUT_DIM];
        #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
        nnet::edgeblock<input2_t, input3_t, layer7_t, config7>(layer6_out, layer4_out, edge_index, layer7_out, R2_w0, R2_b0, R2_w1, R2_b1, R2_w2, R2_b2, R2_w3, R2_b3); // R2

        nnet::sigmoid<layer7_t, result_t, sigmoid_config8>(layer7_out, layer8_out); // final_act

    }


}
