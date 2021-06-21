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
    input_t edge_attr[N_EDGE*EDGE_DIM], input2_t node_attr[N_NODE*NODE_DIM], input3_t edge_index[TWO*N_EDGE],
    layer6_t layer6_out_L[N_EDGE*LAYER6_OUT_DIM],
    unsigned short &const_size_in_1, unsigned short &const_size_in_2, unsigned short &const_size_in_3,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=edge_attr complete dim=0
    #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
    #pragma HLS ARRAY_PARTITION variable=edge_index complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer6_out_L complete dim=0
    #pragma HLS INTERFACE ap_vld port=edge_attr,node_attr,edge_index,layer6_out_L 
    #pragma HLS PIPELINE 

    const_size_in_1 = N_EDGE*EDGE_DIM;
    const_size_in_2 = N_NODE*NODE_DIM;
    const_size_in_3 = TWO*N_EDGE;
    const_size_out_1 = N_EDGE*LAYER6_OUT_DIM;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 400>(R1_w0, "R1_w0.txt");
        nnet::load_weights_from_txt<model_default_t, 1600>(R1_w1, "R1_w1.txt");
        nnet::load_weights_from_txt<model_default_t, 160>(R1_w2, "R1_w2.txt");
        nnet::load_weights_from_txt<model_default_t, 160>(R1_w3, "R1_w3.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(R1_b0, "R1_b0.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(R1_b1, "R1_b1.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(R1_b2, "R1_b2.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(R1_b3, "R1_b3.txt");
        nnet::load_weights_from_txt<model_default_t, 280>(O_w0, "O_w0.txt");
        nnet::load_weights_from_txt<model_default_t, 1600>(O_w1, "O_w1.txt");
        nnet::load_weights_from_txt<model_default_t, 120>(O_w2, "O_w2.txt");
        nnet::load_weights_from_txt<model_default_t, 120>(O_w3, "O_w3.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(O_b0, "O_b0.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(O_b1, "O_b1.txt");
        nnet::load_weights_from_txt<model_default_t, 3>(O_b2, "O_b2.txt");
        nnet::load_weights_from_txt<model_default_t, 3>(O_b3, "O_b3.txt");
        nnet::load_weights_from_txt<model_default_t, 400>(R2_w0, "R2_w0.txt");
        nnet::load_weights_from_txt<model_default_t, 1600>(R2_w1, "R2_w1.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(R2_w2, "R2_w2.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(R2_w3, "R2_w3.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(R2_b0, "R2_b0.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(R2_b1, "R2_b1.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(R2_b2, "R2_b2.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(R2_b3, "R2_b3.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer4_t layer4_out_L[N_EDGE*LAYER4_OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=layer4_out_L complete dim=0
    layer4_t layer4_out_Q[N_NODE*LAYER4_OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=layer4_out_Q complete dim=0
    nnet::IN_edge_module<input_t, input3_t, layer4_t, config4>(edge_attr, node_attr, edge_index, layer4_out_L, layer4_out_Q, R1_w0, R1_b0, R1_w1, R1_b1, R1_w2, R1_b2, R1_w3, R1_b3); // R1
    
    layer5_t layer5_out_P[N_NODE*LAYER5_OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=layer5_out_P complete dim=0
    nnet::IN_node_module<input2_t, layer5_t, config5>(node_attr, layer4_out_Q, layer5_out_P, O_w0, O_b0, O_w1, O_b1, O_w2, O_b2, O_w3, O_b3); // O

    layer6_t layer6_out_Q[N_NODE*LAYER6_OUT_DIM];
    #pragma HLS ARRAY_PARTITION variable=layer6_out_Q complete dim=0
    nnet::IN_edge_module<input_t, input3_t, layer6_t, config6>(layer4_out_L, layer5_out_P, edge_index, layer6_out_L, layer6_out_Q, R2_w0, R2_b0, R2_w1, R2_b1, R2_w2, R2_b2, R2_w3, R2_b3); // R2
    
    std::ofstream E_save;
    E_save.open("edge_attr.csv");
    for(int i=0; i < N_EDGE*EDGE_DIM; i++){
      E_save << edge_attr[i] << std::endl;
    } 
    E_save << "edge_attr" << std::endl;
    E_save.close();

    std::ofstream index_save;
    index_save.open("edge_index.csv");
    for(int i=0; i < TWO*N_EDGE; i++){
      index_save << edge_index[i] << std::endl;
    } 
    index_save << "edge_index" << std::endl;
    index_save.close();

    std::ofstream N_save;
    N_save.open("node_attr.csv");
    for(int i=0; i < N_NODE*NODE_DIM; i++){
      N_save << node_attr[i] << std::endl;
    }
    N_save.close();

    std::ofstream L1_out;
    L1_out.open("edge_update_1.csv");
    for(int i=0; i<N_EDGE*LAYER4_OUT_DIM; i++){
      L1_out << layer4_out_L[i] << std::endl;
    }
    L1_out.close();
 
    std::ofstream Q1_out;
    Q1_out.open("edge_update_aggr_1.csv");
    for(int i=0; i<N_NODE*LAYER4_OUT_DIM; i++){
      Q1_out << layer4_out_Q[i] << std::endl;
    }
    Q1_out.close();

    std::ofstream P_out;
    P_out.open("node_update.csv");
    for(int i=0; i<N_NODE*LAYER5_OUT_DIM; i++){
      P_out << layer5_out_P[i] << std::endl;
    }
    P_out.close();

    std::ofstream L2_out;
    L2_out.open("edge_update_2.csv");
    for(int i=0; i<N_EDGE*LAYER6_OUT_DIM; i++){
      L2_out << layer6_out_L[i] << std::endl;
    }
    L2_out.close();

    std::ofstream Q2_out;
    Q2_out.open("edge_update_aggr_2.csv");
    for(int i=0; i<N_NODE*LAYER6_OUT_DIM; i++){
      Q2_out << layer6_out_Q[i] << std::endl;
    }
    Q2_out.close();
}
