#ifndef LIBTENSOR_GEN_BTO_H
#define LIBTENSOR_GEN_BTO_H

/** \page gen_bto Generalized block tensors operations

    The generalized block tensors and block tensor operations provide
    building blocks for specialized block tensors and operations which employ
    specific types of tensors and data types. The tensor types and element
    types have to be provided via traits classes.

    The main traits class is the block tensor interface traits class. Any
    implementation of block tensors has to define this class with the
    following member types:
    - element_type -- Type of the data elements
    - template<N> rd_block_type::type -- Type of read-only tensor blocks
    - template<N> wr_block_type::type -- Type of read-write tensor blocks

    The generalized block tensor operations employ an additional traits class
    with member types
    - bti_traits -- Type of block tensor interface traits class
    - temp_block_tensor_type -- Type of a temporary block tensor
    - to_add_type -- Type of tensor operation for addition
    - to_apply_type -- Type of tensor operation for element-wise application
        of a function
    - to_copy_type -- Type of tensor operation for copy
    - to_diag_type -- Type of tensor operation for taking a generalized diagonal
    - to_dirsum_type -- Type of tensor operation for direct sum
    - to_dotprod_type -- Type of tensor operation for dot product
    - to_extract_type -- Type of tensor operation for extraction of sub tensors
    - to_mult_type -- Type of tensor operation for element-wise multiplication
    - to_mult1_type -- Type of tensor operation for element-wise multiplication
    - to_scale_type -- Type of tensor operation for scalar transformation
    - to_scatter_type -- Type of tensor operation for distributing elements
        into a larger tensor
    - to_set_type -- Type of tensor operation for setting a certain value
    - to_trace_type -- Type of tensor operation for taking the trace
    - to_vmpriority_type -- Type of tensor operation for setting and unsetting
        of memory priority
    The traits class also has to have the following functions defined
    - bool is_zero(element_type)
    - element_type zero()
    - element_type identity()

    TODO: add generalized interfaces for tensor operations

    \ingroup libtensor_gen_bto
 **/

#include "gen_block_stream_i.h"
#include "gen_block_tensor_ctrl.h"
#include "gen_block_tensor_i.h"
#include "gen_block_tensor.h"


#include "gen_bto_add.h"
#include "gen_bto_apply.h"
#include "gen_bto_contract2.h"
#include "gen_bto_copy.h"
#include "gen_bto_diag.h"
#include "gen_bto_dirsum.h"
#include "gen_bto_mult.h"
#include "gen_bto_set.h"


#endif // LIBTENSOR_GEN_BTO_H

