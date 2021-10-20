#ifndef NNET_GRAPH_H_
#define NNET_GRAPH_H_

#include "nnet_common.h"
#include "nnet_merge.h"
#include "nnet_dense.h"
#include "nnet_dense_resource.h"
#include "nnet_activation.h"
#include "nnet_array.h"
#include <math.h>

namespace nnet {
  enum flow {source_to_target=0, target_to_source=1};
  enum aggr {aggr_sum=0, aggr_mean=1, aggr_max=2};
  
  struct graph_config
  {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float table_t;
    
    // Layer Sizes
    static const unsigned n_node = 10;
    static const unsigned n_edge = 20;
    static const unsigned n_features = 3;
    static const unsigned e_features = 4;
    static const unsigned n_out = 4;
    static const unsigned n_layers = 3;

    // message-passing parameters
    static const unsigned aggr = aggr_sum;
    static const unsigned flow = source_to_target;
    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool activate_final = false;
    static const bool resource_limit = false;
    static const bool no_aggr = false; 
  };

  struct aggregate_config
  {
     typedef float table_t;
     static const unsigned n_node = 10;
     static const unsigned n_edge = 20;
     static const unsigned edge_dim = 4;
     static const unsigned aggr = aggr_sum;
     static const unsigned flow = source_to_target;
  };

  // division-LUT for mean-aggregation
  inline float division(float input){
    return 1.0/input;
  }
  template<typename CONFIG_T, int N_TABLE>
  void init_div_table(typename CONFIG_T::table_t table_out[N_TABLE]){
    int j = 0;
    typename CONFIG_T::table_t k = 1;
    table_out[j] = k;
    for(int i=1; i<N_TABLE; i++){
      float in_val = float(i);
      typename CONFIG_T::table_t reciprocal = nnet::division(in_val);
      table_out[i] = reciprocal;
    }
  }
  template<class data_T, class index_T, class res_T, typename CONFIG_T>
  void edge_divide(data_T edge_sum_i, index_T n_edges_i, res_T &edge_mean_i){
    // initialize LUT
  #ifdef __HLS_SYN__
      bool initialized=false;
      typename CONFIG_T::table_t div_table[CONFIG_T::n_edge];
  #else
      static bool initialized=false;
      static typename CONFIG_T::table_t div_table[CONFIG_T::n_edge];
  #endif

      if(!initialized){
        nnet::init_div_table<CONFIG_T, CONFIG_T::n_edge>(div_table);
        initialized=true;
      }

      if(CONFIG_T::io_type==io_parallel){
        #pragma HLS PIPELINE
      }

      data_T reciprocal;
      reciprocal = div_table[n_edges_i];
      edge_mean_i = edge_sum_i*reciprocal;
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_1lyr(
       data_T data[CONFIG_T::dense_config1::n_in],
       res_T res[CONFIG_T::dense_config1::n_out],
       typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
       typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out])
  {
    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config1>(data, res, weights0, biases0);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_2lyr(
       data_T data[CONFIG_T::dense_config1::n_in],
       res_T res[CONFIG_T::dense_config2::n_out],
       typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
       typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
       typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
       typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config2>(data0, res, weights1, biases1);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_3lyr(
       data_T data[CONFIG_T::dense_config1::n_in],
       res_T res[CONFIG_T::dense_config3::n_out],
       typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
       typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
       typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
       typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
       typename CONFIG_T::dense_config3::weight_t weights2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
       typename CONFIG_T::dense_config3::bias_t   biases2[CONFIG_T::dense_config3::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);
    
     nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config3>(data1, res, weights2, biases2);


  
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_4lyr(
       data_T data[CONFIG_T::dense_config1::n_in],
       res_T res[CONFIG_T::dense_config4::n_out],
       typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
       typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
       typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
       typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
       typename CONFIG_T::dense_config3::weight_t weights2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
       typename CONFIG_T::dense_config3::bias_t   biases2[CONFIG_T::dense_config3::n_out],
       typename CONFIG_T::dense_config4::weight_t weights3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
       typename CONFIG_T::dense_config4::bias_t   biases3[CONFIG_T::dense_config4::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);  
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    data_T data2_logits[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config3>(data1, data2_logits, weights2, biases2);
    data_T data2[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config3>(data2_logits, data2);

    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config4>(data2, res, weights3, biases3);
  }
  //////////////////////////////dataflow/////////////////////////////////////
  template<class data_T, int row, int col>
  void replicate(
    data_T     IN  [row*col],
    data_T     OUT1[row*col],
    data_T     OUT2[row*col]
  ){
    for(int i=0; i<row; i++){
      #pragma HLS UNROLL
      for(int j=0; j<col; j++){
        #pragma HLS UNROLL
        OUT1[i*col+j] =  IN[i*col+j];
        OUT2[i*col+j] =  IN[i*col+j];
      }
    }
  }
  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void aggregate_dataflow(
            data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
            index_T   edge_index_1D[CONFIG_T::n_edge*2],
            res_T     edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim])
  {
    #pragma HLS INLINE
    res_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0

    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      receiver_col = 1;
    }
    else{
      receiver_col = 0;
    }     

    if(CONFIG_T::aggr==aggr_max){
      ap_uint<1> edge_aggr_mask[CONFIG_T::n_node];
      #pragma HLS ARRAY_PARTITION variable=edge_aggr_mask complete dim=0        
      for(int i=0;i<CONFIG_T::n_node;i++){
        #pragma HLS UNROLL
        edge_aggr_mask[i]=0;
      }
      res_T most_negative_num;
      most_negative_num.V = 1;
      most_negative_num.V <<= res_T::width - 1;

      for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = most_negative_num;
        }
      }
      for(int i=0; i < CONFIG_T::n_edge; i++){
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        index_T r = edge_index_1D[i*2+receiver_col];
        edge_aggr_mask[r]=1;
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          edge_attr_aggr[r][j] = edge_attr_1D[i*CONFIG_T::edge_dim+j] > edge_attr_aggr[r][j] ? edge_attr_1D[i*CONFIG_T::edge_dim+j] : edge_attr_aggr[r][j];
        }
      }
      for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = edge_aggr_mask[i]*edge_attr_aggr[i][j];
        }
      }
    }
    if(CONFIG_T::aggr==aggr_mean){
      index_T num_edge_per_node[CONFIG_T::n_node];
      #pragma HLS ARRAY_PARTITION variable=num_edge_per_node complete dim=0
      for(int i=0;i<CONFIG_T::n_node;i++){
        #pragma HLS UNROLL
        num_edge_per_node[i]=0;
      }
      for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = 0;
        }
      }
      for(int i=0; i < CONFIG_T::n_edge; i++){
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        index_T r = edge_index_1D[i*2+receiver_col];
        num_edge_per_node[r]+=1;
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          edge_attr_aggr[r][j] += edge_attr_1D[i*CONFIG_T::edge_dim+j];
        }
      }
      for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        for (int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          res_T edge_mean_j;
          nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_attr_aggr[i][j], num_edge_per_node[i], edge_mean_j);
          edge_attr_aggr[i][j] = edge_mean_j;
        }
      }
    }
    if(CONFIG_T::aggr==aggr_sum){
      for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = 0;
        }
      }
      for(int i=0; i < CONFIG_T::n_edge; i++){
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        index_T r = edge_index_1D[i*2+receiver_col];
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          edge_attr_aggr[r][j] += edge_attr_1D[i*CONFIG_T::edge_dim+j];
        }
      }
    }
    //output array --> output vec
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr, edge_attr_aggr_1D);
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void edgeblock_dataflow(
            data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
      data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
      index_T   edge_index_1D[CONFIG_T::n_edge*2],
      res_T     edge_update_1D[CONFIG_T::n_edge*CONFIG_T::out_dim],
      typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
      typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
      typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
      typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
      typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
      typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
      typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
      typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])
  {
    #pragma HLS INLINE
    int sender_col;
    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      sender_col = 0;
      receiver_col = 1;
    }
    else{
      sender_col = 1;
      receiver_col = 0;
    }

    
    edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) { //for each edge
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
      data_T edge_attr[CONFIG_T::edge_dim];
      #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
      trans_loop_1: for (int c=0; c < CONFIG_T::edge_dim; c++){
        #pragma HLS UNROLL
        edge_attr[c] = edge_attr_1D[i*CONFIG_T::edge_dim+c];
      }
      
      index_T s = edge_index_1D[i*2+sender_col];
      index_T r = edge_index_1D[i*2+receiver_col];
      data_T node_attr_r[CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_attr_r complete dim=0
      
      trans_loop_3: for (int c=0; c < CONFIG_T::node_dim; c++){
        #pragma HLS UNROLL
        node_attr_r[c] = node_attr_1D[r*CONFIG_T::node_dim+c];
      }

      data_T node_attr_s[CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_attr_s complete dim=0
      trans_loop_4: for (int c=0; c < CONFIG_T::node_dim; c++){
        #pragma HLS UNROLL
        node_attr_s[c] = node_attr_1D[s*CONFIG_T::node_dim+c];
      }
      // construct NN input: <receiver, sender, edge>
      data_T node_concat[2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_concat complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr_r, node_attr_s, node_concat);
      data_T phi_input[CONFIG_T::edge_dim + 2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config2>(node_concat, edge_attr, phi_input);
      res_T edge_update[CONFIG_T::out_dim];
      #pragma HLS ARRAY_PARTITION variable=edge_update complete dim=0
      // send it through NN
      if(CONFIG_T::n_layers == 1){
        nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update, core_edge_w0, core_edge_b0);
      }
      else if(CONFIG_T::n_layers == 2){
        nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
      }
      else if(CONFIG_T::n_layers == 3){
        nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
      }
      else if(CONFIG_T::n_layers == 4){
        nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, edge_update, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
      }
      data_T edge_update_sig [CONFIG_T::out_dim];
        if(CONFIG_T::activate_final){        
          #pragma HLS ARRAY_PARTITION variable=edge_update_sig dim=0
          nnet::sigmoid<data_T, res_T, typename CONFIG_T::sigmoid_config1>(edge_update, edge_update_sig);
      }
        trans_loop_5: for (int c=0; c < CONFIG_T::out_dim; c++){
        #pragma HLS UNROLL
        if(CONFIG_T::activate_final)
          edge_update_1D[i*CONFIG_T::out_dim+c] = edge_update_sig[c];
        else
          edge_update_1D[i*CONFIG_T::out_dim+c] = edge_update[c];
      }     
    }
  }
  template<class data_T, class res_T, typename CONFIG_T>
    void nodeblock_dataflow(
      data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
      data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
      res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
      typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
      typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
      typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
      typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
      typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
      typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
      typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
      typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  {
    #pragma HLS INLINE
    node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){ //for each node
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
       data_T node_attr[CONFIG_T::node_dim];
       #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
      trans_loop_1: for (int c=0; c < CONFIG_T::node_dim; c++){
        #pragma HLS UNROLL
        node_attr[c] = node_attr_1D[i*CONFIG_T::node_dim+c];
      }
      data_T edge_attr_aggr[CONFIG_T::edge_dim];
      #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0
      trans_loop_2: for (int c=0; c < CONFIG_T::edge_dim; c++){
        #pragma HLS UNROLL
        edge_attr_aggr[c] = edge_attr_aggr_1D[i*CONFIG_T::edge_dim+c];
      }
      data_T phi_input[CONFIG_T::edge_dim + CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr, edge_attr_aggr, phi_input);
       res_T node_update[CONFIG_T::out_dim];
       #pragma HLS ARRAY_PARTITION variable=node_update complete dim=0
        if(CONFIG_T::n_layers == 1){
          nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update, core_node_w0, core_node_b0);
        }
        else if(CONFIG_T::n_layers == 2){
          nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update, core_node_w0, core_node_b0, core_node_w1, core_node_b1);
        }
        else if(CONFIG_T::n_layers == 3){
          nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(phi_input, node_update, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
        }
        else { // CONFIG_T::n_layers == 4
          nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
        }
        trans_loop_3: for (int c=0; c < CONFIG_T::out_dim; c++){
        #pragma HLS UNROLL
        node_update_1D[i*CONFIG_T::out_dim+c] = node_update[c];
      }
    }
  }

////////////////////////////////////pipeline////////////////////////////////////
template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void aggregate_pipeline(
            data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
            index_T   edge_index_1D[CONFIG_T::n_edge*2],
            res_T     edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim])
  {
    #pragma HLS INLINE
    //initialize arrays
    // 1. edge_attr (input)
    data_T edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_config>(edge_attr_1D, edge_attr);

    // 2. edge_index (input)
    index_T edge_index[CONFIG_T::n_edge][2];
    #pragma HLS ARRAY_PARTITION variable=edge_index complete dim=0
    nnet::vec_to_mat<index_T, index_T, typename CONFIG_T::edge_index_config>(edge_index_1D, edge_index);

    //3. num_edge_per_node (intermediate), 4. edge_aggr_mask (intermediate)
    index_T num_edge_per_node[CONFIG_T::n_node];
    #pragma HLS ARRAY_PARTITION variable=num_edge_per_node complete dim=0
    ap_uint<1> edge_aggr_mask[CONFIG_T::n_node];
    #pragma HLS ARRAY_PARTITION variable=edge_aggr_mask complete dim=0
    for(int i=0; i<CONFIG_T::n_node; i++){
      #pragma HLS UNROLL
      num_edge_per_node[i] = 0;
      if(CONFIG_T::aggr==aggr_max){
        edge_aggr_mask[i] = 0;
      }
    }

    //4. edge_attr_aggr (output)
    res_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_update_aggr complete dim=0
    if((CONFIG_T::aggr==aggr_sum)||(CONFIG_T::aggr==aggr_mean)){
      for(int i=0; i < CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = 0;
        }
      }
    }
    else{ //CONFIG_T:aggr==aggr_max, we want to initialize this with the most negative number we can represent
      // note this only works for ap_ufixed types
      res_T most_negative_num;
      most_negative_num.V = 1;
      most_negative_num.V <<= res_T::width - 1;

      for(int i=0; i < CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = most_negative_num;
        }
      }
    }

    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      receiver_col = 1;
    }
    else{
      receiver_col = 0;
    }

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    for(int i=0; i<CONFIG_T::n_edge; i++){
      #pragma HLS UNROLL
      index_T r = edge_index[i][receiver_col];
      num_edge_per_node[r] += 1;
      edge_aggr_mask[r] = 1;

      if((CONFIG_T::aggr == aggr_sum)||(CONFIG_T::aggr==aggr_mean)){
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[r][j] += edge_attr[i][j];
        }
      }
      else{ //CONFIG_T::aggr==aggr_max
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[r][j] = edge_attr[i][j] > edge_attr_aggr[r][j] ? edge_attr[i][j] : edge_attr_aggr[r][j];
        }
      }
    }

    // sum --> mean
    if(CONFIG_T::aggr == aggr_mean){
      for(int i=0; i < CONFIG_T::n_node; i++){
        for (int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          res_T edge_mean_j;
          nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_attr_aggr[i][j], num_edge_per_node[i], edge_mean_j);
          edge_attr_aggr[i][j] = edge_mean_j;
        }
      }
    }

    // None --> max
    if(CONFIG_T::aggr == aggr_max){ //note: the edge_update_aggr array has been initialized but IS NOT ZEROS
      for(int i=0; i < CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = edge_aggr_mask[i]*edge_attr_aggr[i][j];
        }
      }
    }

    //output array --> output vec
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr, edge_attr_aggr_1D);
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void edgeblock_pipeline(
            data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
			index_T   edge_index_1D[CONFIG_T::n_edge*2],
			res_T     edge_update_1D[CONFIG_T::n_edge*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])
  {
    #pragma HLS INLINE
    //initialize arrays
    // 1. node_attr (input)
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    // 2. edge_attr (input)
    data_T edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_config>(edge_attr_1D, edge_attr);

    // 3. edge_index (input)
    index_T edge_index[CONFIG_T::n_edge][2];
    #pragma HLS ARRAY_PARTITION variable=edge_index complete dim=0
    nnet::vec_to_mat<index_T, index_T, typename CONFIG_T::edge_index_config>(edge_index_1D, edge_index);

    // 4. edge_update (output)
    res_T edge_update[CONFIG_T::n_edge][CONFIG_T::out_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_update complete dim=0

    int sender_col;
    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      sender_col = 0;
      receiver_col = 1;
    }
    else{
      sender_col = 1;
      receiver_col = 0;
    }

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) { //for each edge
      #pragma HLS UNROLL

      // get sender, receiver indices
      index_T s = edge_index[i][sender_col];
      index_T r = edge_index[i][receiver_col];

      // construct NN input: <receiver, sender, edge>
      data_T node_concat[2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_concat complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[r], node_attr[s], node_concat);
      data_T phi_input[CONFIG_T::edge_dim + 2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config2>(node_concat, edge_attr[i], phi_input);

      // send it through NN
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0);
          }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
        }
        else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
        }
        else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
        }
    }

    //output arrays --> output vectors
    // 1. edge_update_1D
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_update_config>(edge_update, edge_update_1D);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void nodeblock_pipeline(
			data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
			res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  {
    #pragma HLS INLINE
    //initialize arrays
    //1. node_attr (input)
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    //2. edge_attr_aggr (input)
    data_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr_1D, edge_attr_aggr);

    // 3. node_update (output)
    res_T node_update[CONFIG_T::n_node][CONFIG_T::out_dim];
    #pragma HLS ARRAY_PARTITION variable=node_update complete dim=0

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){ //for each node
      #pragma HLS UNROLL

      // construct NN input: <node, edge_attr_aggr>
      data_T phi_input[CONFIG_T::edge_dim + CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);

      // send it through NN
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0);
        }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1);
        }
        else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
        }
        else { // CONFIG_T::n_layers == 4
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
        }
    }

    // output array --> output vector
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::node_update_config>(node_update, node_update_1D);

  }
  //////////////////////////////////////////////////////////////////////////
  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void aggregate(
            data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
            index_T   edge_index_1D[CONFIG_T::n_edge*2],
            res_T     edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim])
    {
      if(CONFIG_T::resource_limit){
        aggregate_dataflow<data_T,index_T,res_T,CONFIG_T>(edge_attr_1D,edge_index_1D,edge_attr_aggr_1D);
      }
      else{
        aggregate_pipeline<data_T,index_T,res_T,CONFIG_T>(edge_attr_1D,edge_index_1D,edge_attr_aggr_1D);
      }
    }
  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void edgeblock(
            data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
			index_T   edge_index_1D[CONFIG_T::n_edge*2],
			res_T     edge_update_1D[CONFIG_T::n_edge*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])
    {
      if(CONFIG_T::resource_limit){
        edgeblock_dataflow<data_T,index_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_1D,edge_index_1D,edge_update_1D,core_edge_w0,core_edge_b0,core_edge_w1,core_edge_b1,core_edge_w2,core_edge_b2,core_edge_w3,core_edge_b3);
      }
      else{
        edgeblock_pipeline<data_T,index_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_1D,edge_index_1D,edge_update_1D,core_edge_w0,core_edge_b0,core_edge_w1,core_edge_b1,core_edge_w2,core_edge_b2,core_edge_w3,core_edge_b3);
      }
    }
    template<class data_T, class res_T, typename CONFIG_T>
    void nodeblock(
			data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
			res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
    {
      if(CONFIG_T::resource_limit){
        nodeblock_dataflow<data_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_aggr_1D,node_update_1D,core_node_w0,core_node_b0,core_node_w1,core_node_b1,core_node_w2,core_node_b2,core_node_w3,core_node_b3);
      }
      else{
        nodeblock_pipeline<data_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_aggr_1D,node_update_1D,core_node_w0,core_node_b0,core_node_w1,core_node_b1,core_node_w2,core_node_b2,core_node_w3,core_node_b3);
      }
    }

}
#endif
