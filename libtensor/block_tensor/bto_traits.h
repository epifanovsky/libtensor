#ifndef LIBTENSOR_BTO_TRAITS_H
#define LIBTENSOR_BTO_TRAITS_H

#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod.h>
#include <libtensor/block_tensor/btod_contract2_clst_optimize.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/block_tensor_i_traits.h>

namespace libtensor {


//template<size_t N, size_t M, size_t K> class tod_contract2;
//template<size_t N> class tod_set;

template<typename T>
struct bto_traits {

    //! Element type
    typedef T element_type;

    //! Block tensor interface traits
    typedef block_tensor_i_traits<T> bti_traits;

    //! Type of temporary block tensor
    template<size_t N>
    struct temp_block_tensor_type {
        typedef block_tensor< N, T, allocator<T> > type;
    };

    template<size_t N>
    struct temp_block_type {
        typedef dense_tensor< N, T, allocator<T> > type;
    };

    template<size_t N>
    struct to_add_type {
        typedef to_add<N, T> type; 
    };

    template<size_t N, typename Functor>
    struct to_apply_type {
        typedef to_apply<N, Functor, T> type;
    };

    template<size_t N>
    struct to_compare_type {
        typedef to_compare<N, T> type;
    };

    template<size_t N, size_t M, size_t K>
    struct to_contract2_type {
        typedef to_contract2<N, M, K, T> type;
        typedef bto_contract2_clst_optimize<N, M, K, T> clst_optimize_type; 
    };

    template<size_t N>
    struct to_copy_type {
        typedef to_copy<N, T> type;
    };

    template<size_t N, size_t M>
    struct to_diag_type {
        typedef to_diag<N, M, T> type;
    };

    template<size_t N, size_t M>
    struct to_dirsum_type {
        typedef to_dirsum<N, M, T> type; 
    };

    template<size_t N>
    struct to_dotprod_type {
        typedef to_dotprod<N, T> type; 
    };

    template<size_t N, size_t M, size_t K>
    struct to_ewmult2_type {
        typedef to_ewmult2<N, M, K, T> type;
    };

    template<size_t N, size_t M>
    struct to_extract_type {
        typedef to_extract<N, M, T> type;
    };

    template<size_t N>
    struct to_mult_type {
        typedef to_mult<N, T> type;
    };

    template<size_t N>
    struct to_mult1_type {
        typedef to_mult1<N, T> type;
    };

    template<size_t N>
    struct to_random_type {
        typedef to_random<N, T> type;
    };

    template<size_t N>
    struct to_scale_type {
        typedef to_scale<N, T> type; 
    };

    template<size_t N, size_t M>
    struct to_scatter_type {
        typedef to_scatter<N, M, T> type; 
    };

    template<size_t N, typename ComparePolicy>
    struct to_select_type {
        typedef to_select<N, T, ComparePolicy> type;
    };

    template<size_t N>
    struct to_set_diag_type {
        typedef to_set_diag<N, T> type;
    };

    template<size_t N>
    struct to_set_elem_type {
        typedef to_set_elem<N, T> type;
    };

    template<size_t N>
    struct to_set_type {
        typedef to_set<N, T> type;
    };

    template<size_t N>
    struct to_size_type {
        typedef to_size<N, T> type;
    };

    template<size_t N>
    struct to_trace_type {
        typedef to_trace<N, T> type;
    };

    template<size_t N>
    struct to_vmpriority_type {
        typedef to_vmpriority<N, T> type;
    };

    static bool is_zero(T d) {
        return d == 0.0;
    }

    static bool is_zero(const scalar_transf<T> &d) {
        return is_zero(d.get_coeff());
    }

    static T zero() {
        return 0.0;
    }

    static T identity() {
        return 1.0;
    }

};

using btod_traits = bto_traits<double>;

} // namespace libtensor

#endif // LIBTENSOR_BTO_TRAITS_H
