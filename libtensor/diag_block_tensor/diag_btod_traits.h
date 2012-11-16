#ifndef LIBTENSOR_DIAG_BTOD_TRAITS_H
#define LIBTENSOR_DIAG_BTOD_TRAITS_H

#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/diag_tensor/diag_tensor_i.h>
#include <libtensor/diag_tensor/diag_tod_contract2s.h>
#include <libtensor/diag_tensor/diag_tod_copy.h>
#include <libtensor/diag_tensor/diag_tod_set.h>
#include <libtensor/block_tensor/btod_contract2_clst_optimize.h>
#include "diag_block_tensor.h"

namespace libtensor {


struct diag_btod_traits {

    //! Element type
    typedef double element_type;

    //! Block tensor interface traits
    typedef diag_block_tensor_i_traits<double> bti_traits;

    //! Type of temporary block tensor
    template<size_t N>
    struct temp_block_tensor_type {
        typedef diag_block_tensor< N, double, allocator<double> > type;
    };

    //! Type of block of block tensors
    template<size_t N>
    struct block_type {
        typedef diag_tensor_i<N, double> type;
    };

    //! Type of block of block tensors
    template<size_t N>
    struct wr_block_type {
        typedef diag_tensor_i<N, double> type;
    };

//    template<size_t N, typename Functor>
//    struct to_apply_type {
//        typedef tod_apply<N, Functor> type;
//    };

    template<size_t N, size_t M, size_t K>
    struct to_contract2_type {
        typedef diag_tod_contract2s<N, M, K> type;
        typedef btod_contract2_clst_optimize<N, M, K> clst_optimize_type;
    };

    template<size_t N>
    struct to_copy_type {
        typedef diag_tod_copy<N> type;
    };

//    template<size_t N, size_t M>
//    struct to_diag_type {
//        typedef tod_diag<N, M> type;
//    };

//    template<size_t N, size_t M>
//    struct to_dirsum_type {
//        typedef tod_dirsum<N, M> type;
//    };

//    template<size_t N>
//    struct to_dotprod_type {
//        typedef tod_dotprod<N> type;
//    };

//    template<size_t N>
//    struct to_mult_type {
//        typedef tod_mult<N> type;
//    };

    template<size_t N>
    struct to_set_type {
        typedef diag_tod_set<N> type;
    };

//    template<size_t N, size_t M>
//    struct to_scatter_type {
//        typedef tod_scatter<N, M> type;
//    };

//    template<size_t N>
//    struct to_trace_type {
//        typedef tod_trace<N> type;
//    };

//    template<size_t N>
//    struct to_vmpriority_type {
//        typedef tod_vmpriority<N> type;
//    };

    static bool is_zero(double d) {
        return d == 0.0;
    }

    static double zero() {
        return 0.0;
    }

    static double identity() {
        return 1.0;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_TRAITS_H
