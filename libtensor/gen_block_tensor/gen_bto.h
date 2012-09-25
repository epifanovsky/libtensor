#ifndef LIBTENSOR_GEN_BTO_H
#define LIBTENSOR_GEN_BTO_H

/** \page gen_bto Generalized block tensors operations

    The generalized block tensors and block tensor operations provide
    building blocks for specialized block tensors and operations which employ
    specific types of tensors and specific data types. The tensor types
    and element types have to be provided via traits classes.

    The main traits class which is employed is the block tensor interface
    traits class. Any specific implementation of block tensors has to define
    this class with the following member types:
    - element_type -- Type of the data elements
    - template<N> rd_block_type::type -- Type of read-only tensor blocks
    - template<N> wr_block_type::type -- Type of read-write tensor blocks

    The generalized block tensor operations use tensor operations also
    provided via the traits classes. The tensor operation classes are
    expected to have the following interfaces ...

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

