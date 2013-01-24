#ifndef LIBTENSOR_CUDA_BTOD_TRAITS_H
#define LIBTENSOR_CUDA_BTOD_TRAITS_H

#include <libvmm/cuda_allocator.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_contract2.h>
#include <libtensor/block_tensor/btod_contract2_clst_optimize.h>
#include "cuda_block_tensor.h"

namespace libtensor {


struct cuda_btod_traits {

    //! Element type
    typedef double element_type;

    //! Block tensor interface traits
    typedef cuda_block_tensor_i_traits<double> bti_traits;

    //! Type of temporary block tensor
    template<size_t N>
    struct temp_block_tensor_type {
        typedef cuda_block_tensor< N, double, libvmm::cuda_allocator<double> > type;
    };

    template<size_t N>
    struct temp_block_type {
        typedef dense_tensor< N, double, libvmm::cuda_allocator<double> > type;
    };

    template<size_t N>
    struct to_add_type {
        typedef tod_add<N> type;
    };

    template<size_t N, typename Functor>
    struct to_apply_type {
        typedef tod_apply<N, Functor> type;
    };

    template<size_t N>
    struct to_compare_type {
        typedef tod_compare<N> type;
    };

    template<size_t N, size_t M, size_t K>
    struct to_contract2_type {
        typedef cuda_tod_contract2<N, M, K> type;
        typedef btod_contract2_clst_optimize<N, M, K> clst_optimize_type;
    };

    template<size_t N>
    struct to_copy_type {
        typedef cuda_tod_copy<N> type;
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

    template<size_t N, size_t M, size_t K>
    struct to_ewmult2_type {
        typedef tod_ewmult2<N, M, K> type;
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
    struct to_mult1_type {
        typedef tod_mult1<N> type;
    };

    template<size_t N>
    struct to_random_type {
        typedef tod_random<N> type;
    };

    template<size_t N>
    struct to_scale_type {
        typedef tod_scale<N> type;
    };

    template<size_t N, size_t M>
    struct to_scatter_type {
        typedef tod_scatter<N, M> type;
    };

    template<size_t N, typename ComparePolicy>
    struct to_select_type {
        typedef tod_select<N, ComparePolicy> type;
    };

    template<size_t N>
    struct to_set_diag_type {
        typedef tod_set_diag<N> type;
    };

    template<size_t N>
    struct to_set_elem_type {
        typedef tod_set_elem<N> type;
    };

    template<size_t N>
    struct to_set_type {
        typedef tod_set<N> type;
    };

    template<size_t N>
    struct to_size_type {
        typedef tod_size<N> type;
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

#endif // LIBTENSOR_CUDA_BTOD_TRAITS_H
