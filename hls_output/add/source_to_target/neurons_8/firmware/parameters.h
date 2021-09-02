#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_array.h"
#include "nnet_utils/nnet_common.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_resource.h"
#include "nnet_utils/nnet_graph.h"
#include "nnet_utils/nnet_merge.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/R1_w0.h"
#include "weights/R1_w1.h"
#include "weights/R1_w2.h"
#include "weights/R1_w3.h"
#include "weights/R1_b0.h"
#include "weights/R1_b1.h"
#include "weights/R1_b2.h"
#include "weights/R1_b3.h"
#include "weights/O_w0.h"
#include "weights/O_w1.h"
#include "weights/O_w2.h"
#include "weights/O_w3.h"
#include "weights/O_b0.h"
#include "weights/O_b1.h"
#include "weights/O_b2.h"
#include "weights/O_b3.h"
#include "weights/R2_w0.h"
#include "weights/R2_w1.h"
#include "weights/R2_w2.h"
#include "weights/R2_w3.h"
#include "weights/R2_b0.h"
#include "weights/R2_b1.h"
#include "weights/R2_b2.h"
#include "weights/R2_b3.h"

//hls-fpga-machine-learning insert layer-config
// R1
struct config4: nnet::graph_config{
    typedef layer4_t bias_t;
    typedef layer4_t weight_t;
    typedef layer4_t table_t;
    static const unsigned n_node = N_NODE;
    static const unsigned n_edge = N_EDGE;
    static const unsigned node_dim = NODE_DIM;
    static const unsigned edge_dim = EDGE_DIM;
    static const unsigned out_dim = 4;
    static const unsigned n_layers = 3;
    static const unsigned flow = 0;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool io_stream = false; 

    struct dense_config1 : nnet::dense_config {
        static const unsigned n_in = 10;
        static const unsigned n_out = 8;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer4_t accum_t;
        typedef layer4_t bias_t;
        typedef layer4_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config1 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct dense_config2 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 8;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer4_t accum_t;
        typedef layer4_t bias_t;
        typedef layer4_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config2 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct dense_config3 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 4;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer4_t accum_t;
        typedef layer4_t bias_t;
        typedef layer4_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct dense_config4 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 4;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer4_t accum_t;
        typedef layer4_t bias_t;
        typedef layer4_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config3 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct relu_config4 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct node_attr_config: nnet::matrix_config{
                        static const unsigned n_rows = N_NODE;
                        static const unsigned n_cols = NODE_DIM;
                    };

    struct edge_attr_config: nnet::matrix_config{
                        static const unsigned n_rows = N_EDGE;
                        static const unsigned n_cols = EDGE_DIM;
                    };

    struct edge_index_config: nnet::matrix_config{
                        static const unsigned n_rows = N_EDGE;
                        static const unsigned n_cols = TWO;
                    };

    struct edge_update_config: nnet::matrix_config{
                        static const unsigned n_rows = N_EDGE;
                        static const unsigned n_cols = LAYER4_OUT_DIM;
                    };

    struct merge_config1 : nnet::concat_config {
        static const unsigned n_elem1_0 = NODE_DIM;
        static const unsigned n_elem1_1 = 1;
        static const unsigned n_elem1_2 = 0;
        static const unsigned n_elem2_0 = NODE_DIM;
        static const unsigned n_elem2_1 = 1;
        static const unsigned n_elem2_2 = 0;
    
        static const int axis = 0;
    };
    

    struct merge_config2 : nnet::concat_config {
        static const unsigned n_elem1_0 = 2*NODE_DIM;
        static const unsigned n_elem1_1 = 1;
        static const unsigned n_elem1_2 = 0;
        static const unsigned n_elem2_0 = EDGE_DIM;
        static const unsigned n_elem2_1 = 1;
        static const unsigned n_elem2_2 = 0;
    
        static const int axis = 0;
    };
    
};
// aggr5
struct aggregation_config5: nnet::aggregate_config{
    typedef layer5_t table_t;
    static const unsigned n_node = 28;
    static const unsigned n_edge = 37;
    static const unsigned edge_dim = 4;
    static const unsigned aggr = 0;
    static const unsigned flow = 0;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool io_stream = false;

    struct edge_attr_config: nnet::matrix_config{
                                static const unsigned n_rows = N_EDGE;
                                static const unsigned n_cols = EDGE_DIM;
                            };

    struct edge_index_config: nnet::matrix_config{
                                static const unsigned n_rows = N_EDGE;
                                static const unsigned n_cols = TWO;
                            };

    struct edge_attr_aggr_config: nnet::matrix_config{
                                static const unsigned n_rows = N_NODE;
                                static const unsigned n_cols = LAYER5_OUT_DIM;
                            };

    struct nested_duplicate: nnet::aggregate_config{
        typedef layer5_t table_t;
        static const unsigned n_node = 28;
        static const unsigned n_edge = 37;
        static const unsigned edge_dim = 4;
        static const unsigned aggr = 0;
        static const unsigned flow = 0;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const bool io_stream = false;
    };
};
// O
struct config6: nnet::graph_config{
    typedef layer6_t bias_t;
    typedef layer6_t weight_t;
    typedef layer6_t table_t;
    static const unsigned n_node = N_NODE;
    static const unsigned n_edge = N_EDGE;
    static const unsigned node_dim = NODE_DIM; 
    static const unsigned edge_dim = EDGE_DIM;
    static const unsigned out_dim = 3;
    static const unsigned n_layers = 3;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool io_stream = false; 

    struct dense_config1 : nnet::dense_config {
        static const unsigned n_in = 7;
        static const unsigned n_out = 8;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer6_t accum_t;
        typedef layer6_t bias_t;
        typedef layer6_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config1 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct dense_config2 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 8;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer6_t accum_t;
        typedef layer6_t bias_t;
        typedef layer6_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config2 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct dense_config3 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 3;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer6_t accum_t;
        typedef layer6_t bias_t;
        typedef layer6_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct dense_config4 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 3;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer6_t accum_t;
        typedef layer6_t bias_t;
        typedef layer6_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config3 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct relu_config4 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct node_attr_config: nnet::matrix_config{
                                static const unsigned n_rows = N_NODE;
                                static const unsigned n_cols = NODE_DIM;
                            };

    struct edge_attr_aggr_config: nnet::matrix_config{
                                static const unsigned n_rows = N_NODE;
                                static const unsigned n_cols = LAYER5_OUT_DIM;
                            };

    struct node_update_config: nnet::matrix_config{
                                static const unsigned n_rows = N_NODE;
                                static const unsigned n_cols = LAYER6_OUT_DIM;
                            };

    struct merge_config1 : nnet::concat_config {
        static const unsigned n_elem1_0 = NODE_DIM;
        static const unsigned n_elem1_1 = 1;
        static const unsigned n_elem1_2 = 0;
        static const unsigned n_elem2_0 = EDGE_DIM;
        static const unsigned n_elem2_1 = 1;
        static const unsigned n_elem2_2 = 0;
    
        static const int axis = 0;
    };
    
};
// R2
struct config7: nnet::graph_config{
    typedef layer7_t bias_t;
    typedef layer7_t weight_t;
    typedef layer7_t table_t;
    static const unsigned n_node = N_NODE;
    static const unsigned n_edge = N_EDGE;
    static const unsigned node_dim = NODE_DIM;
    static const unsigned edge_dim = EDGE_DIM;
    static const unsigned out_dim = 1;
    static const unsigned n_layers = 3;
    static const unsigned flow = 0;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool io_stream = false; 

    struct dense_config1 : nnet::dense_config {
        static const unsigned n_in = 10;
        static const unsigned n_out = 8;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer7_t accum_t;
        typedef layer7_t bias_t;
        typedef layer7_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config1 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct dense_config2 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 8;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer7_t accum_t;
        typedef layer7_t bias_t;
        typedef layer7_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config2 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct dense_config3 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 1;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer7_t accum_t;
        typedef layer7_t bias_t;
        typedef layer7_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct dense_config4 : nnet::dense_config {
        static const unsigned n_in = 8;
        static const unsigned n_out = 1;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef layer7_t accum_t;
        typedef layer7_t bias_t;
        typedef layer7_t weight_t;
        static const bool remove_pipeline_pragma = true;
    };
    

    struct relu_config3 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct relu_config4 : nnet::activ_config {
        static const unsigned n_in = 8;
        static const unsigned table_size = 1024;
        static const unsigned io_type = nnet::io_parallel;
    };
    

    struct node_attr_config: nnet::matrix_config{
                        static const unsigned n_rows = N_NODE;
                        static const unsigned n_cols = NODE_DIM;
                    };

    struct edge_attr_config: nnet::matrix_config{
                        static const unsigned n_rows = N_EDGE;
                        static const unsigned n_cols = EDGE_DIM;
                    };

    struct edge_index_config: nnet::matrix_config{
                        static const unsigned n_rows = N_EDGE;
                        static const unsigned n_cols = TWO;
                    };

    struct edge_update_config: nnet::matrix_config{
                        static const unsigned n_rows = N_EDGE;
                        static const unsigned n_cols = LAYER7_OUT_DIM;
                    };

    struct merge_config1 : nnet::concat_config {
        static const unsigned n_elem1_0 = NODE_DIM;
        static const unsigned n_elem1_1 = 1;
        static const unsigned n_elem1_2 = 0;
        static const unsigned n_elem2_0 = NODE_DIM;
        static const unsigned n_elem2_1 = 1;
        static const unsigned n_elem2_2 = 0;
    
        static const int axis = 0;
    };
    

    struct merge_config2 : nnet::concat_config {
        static const unsigned n_elem1_0 = 2*NODE_DIM;
        static const unsigned n_elem1_1 = 1;
        static const unsigned n_elem1_2 = 0;
        static const unsigned n_elem2_0 = EDGE_DIM;
        static const unsigned n_elem2_1 = 1;
        static const unsigned n_elem2_2 = 0;
    
        static const int axis = 0;
    };
    
};
// final_act
struct sigmoid_config8 : nnet::activ_config {
    static const unsigned n_in = N_EDGE*LAYER7_OUT_DIM;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};


#endif
