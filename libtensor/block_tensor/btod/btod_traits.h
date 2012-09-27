#ifndef LIBTENSOR_BTOD_TRAITS_H
#define LIBTENSOR_BTOD_TRAITS_H

#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_apply.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_diag.h>
#include <libtensor/dense_tensor/tod_dirsum.h>
#include <libtensor/dense_tensor/tod_dotprod.h>
#include <libtensor/dense_tensor/tod_extract.h>
#include <libtensor/dense_tensor/tod_mult.h>
#include <libtensor/dense_tensor/tod_scale.h>
#include <libtensor/dense_tensor/tod_scatter.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/dense_tensor/tod_trace.h>
#include <libtensor/dense_tensor/tod_vmpriority.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/block_tensor_i_traits.h>

namespace libtensor {


template<size_t N, size_t M, size_t K> class tod_contract2;
template<size_t N> class tod_set;


struct btod_traits {

    //! Element type
    typedef double element_type;

    //! Block tensor interface traits
    typedef block_tensor_i_traits<double> bti_traits;

    //! Type of block tensor
    template<size_t N>
    struct block_tensor_type {
        typedef block_tensor_i<N, double> type;
    };

    //! Type of block tensor control
    template<size_t N>
    struct block_tensor_ctrl_type {
        typedef block_tensor_ctrl<N, double> type;
    };

    //! Type of temporary block tensor
    template<size_t N>
    struct temp_block_tensor_type {
        typedef block_tensor< N, double, allocator<double> > type;
    };

    //! Type of block of block tensors
    template<size_t N>
    struct block_type {
        typedef dense_tensor_i<N, double> type;
    };

    //! Type of block of block tensors
    template<size_t N>
    struct wr_block_type {
        typedef dense_tensor_i<N, double> type;
    };

    template<size_t N>
    struct temp_block_type {
        typedef dense_tensor< N, double, allocator<double> > type;
    };

    template<size_t N, typename Functor>
    struct to_apply_type {
        typedef tod_apply<N, Functor> type;
    };

    template<size_t N, size_t M, size_t K>
    struct to_contract2_type {
        typedef tod_contract2<N, M, K> type;
    };

    template<size_t N>
    struct to_copy_type {
        typedef tod_copy<N> type;
    };

    template<size_t N, size_t M>
    struct to_diag_type {
        typedef tod_diag<N, M> type;
    };

    template<size_t N, size_t M>
    struct to_dirsum_type {
        typedef tod_dirsum<N, M> type;
    };

    template<size_t N>
    struct to_dotprod_type {
        typedef tod_dotprod<N> type;
    };

    template<size_t N, size_t M>
    struct to_extract_type {
        typedef tod_extract<N, M> type;
    };

    template<size_t N>
    struct to_mult_type {
        typedef tod_mult<N> type;
    };

    template<size_t N>
    struct to_scale_type {
        typedef tod_scale<N> type;
    };

    template<size_t N>
    struct to_set_type {
        typedef tod_set<N> type;
    };

    template<size_t N, size_t M>
    struct to_scatter_type {
        typedef tod_scatter<N, M> type;
    };

    template<size_t N>
    struct to_trace_type {
        typedef tod_trace<N> type;
    };

    template<size_t N>
    struct to_vmpriority_type {
        typedef tod_vmpriority<N> type;
    };

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

#endif // LIBTENSOR_BTOD_TRAITS_H
